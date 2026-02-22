"""Generate training data: PGN + Stockfish -> stacked tensors on disk.

Produces a .pt file with:
    boards:   Tensor[N, 110, 8, 8]
    policies: Tensor[N, 64, 64]
    values:   Tensor[N, 3]        (win, draw, loss per example)

Stockfish evaluation is parallelized across multiple worker processes,
each owning its own Stockfish subprocess.

The main() entry point uses generate_to_file() which streams results into
disk-backed numpy memory-mapped arrays, keeping RSS at ~4-8 GB even for
tens of millions of examples.
"""

import atexit
import multiprocessing
import os
import argparse
import shutil
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.data.stockfish_oracle import StockfishOracle
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

# -- Per-worker process globals (set by _init_worker) ---------------------

_oracle: StockfishOracle | None = None
_encoder: ExtendedBoardEncoder | None = None


def _cleanup_oracle() -> None:
    if _oracle is None:
        return
    try:
        _oracle.close()
    except Exception:  # noqa: BLE001
        pass


def _init_worker(
    stockfish_path: str,
    stockfish_depth: int,
    policy_temperature: float,
    label_smoothing: float,
) -> None:
    global _oracle, _encoder
    _oracle = StockfishOracle(
        path=stockfish_path,
        depth=stockfish_depth,
        policy_temperature=policy_temperature,
        label_smoothing=label_smoothing,
    )
    _encoder = ExtendedBoardEncoder()
    atexit.register(_cleanup_oracle)


# Return numpy arrays instead of torch tensors to avoid FD-based IPC
# (torch uses sendmsg/recvmsg ancillary data to share tensor storage,
# which exhausts file descriptors at high worker counts).
_EvalResult = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], tuple[float, float, float]]

_MoveSeq = list[tuple[int, int, int | None]]


@dataclass(frozen=True)
class _PositionMeta:
    moves: _MoveSeq
    game_id: int
    eco_code: str | None
    piece_count: int


_EvalResultWithMeta = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    tuple[float, float, float],
    int,          # game_id
    str | None,   # eco_code
    int,          # piece_count
]


def _evaluate_position(meta: _PositionMeta) -> _EvalResultWithMeta:
    if _oracle is None or _encoder is None:
        raise RuntimeError("Worker not initialized")
    board = chess.Board()
    for from_sq, to_sq, promo in meta.moves:
        move = chess.Move(from_sq, to_sq, promo)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move {move.uci()} at ply {len(board.move_stack)}")
        board.push(move)
    board_tensor = _encoder.encode(board)
    policy, value, _ = _oracle.evaluate(board)
    return (
        board_tensor.data.numpy(),
        policy.data.numpy(),
        (value.win, value.draw, value.loss),
        meta.game_id,
        meta.eco_code,
        meta.piece_count,
    )


# -- Streaming helpers ----------------------------------------------------


def _count_positions(pgn_path: Path, max_positions: int) -> int:
    """Fast PGN-only pass that counts available positions (no Stockfish)."""
    streamer = SimplePGNStreamer()
    count = 0
    for record in streamer.stream(pgn_path):
        moves_in_game = len(record.actions)
        remaining = max_positions - count
        count += min(moves_in_game, remaining)
        if count >= max_positions:
            break
    return count


def _stream_positions(pgn_path: Path, max_positions: int) -> Iterator[_PositionMeta]:
    """Yield positions lazily — only ~chunksize*workers buffered at once."""
    streamer = SimplePGNStreamer()
    count = 0
    game_id = 0

    for record in streamer.stream(pgn_path):
        moves_so_far: _MoveSeq = []
        board = chess.Board()
        for action in record.actions:
            moves_so_far.append((action.from_square, action.to_square, action.promotion))
            board.push(chess.Move(action.from_square, action.to_square, action.promotion))
            if count >= max_positions:
                break
            piece_count = bin(board.occupied).count("1")
            yield _PositionMeta(
                moves=list(moves_so_far),
                game_id=game_id,
                eco_code=record.eco_code,
                piece_count=piece_count,
            )
            count += 1

        game_id += 1
        if count >= max_positions:
            break


def generate_to_file(
    pgn_path: Path,
    output_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
    num_workers: int,
    policy_temperature: float = 150.0,
    label_smoothing: float = 0.1,
    chunksize: int = 64,
) -> int:
    """Stream positions through Stockfish workers into disk-backed memmap arrays.

    Returns the number of examples written. RSS stays at ~4-8 GB even for
    tens of millions of examples because the large arrays (boards, policies,
    values) live in memory-mapped files that the OS pages in/out on demand.
    """
    # Pass 1: count positions to pre-allocate exact array sizes
    print("Pass 1: counting positions...")
    total = _count_positions(pgn_path, max_examples)
    print(f"Found {total} positions, starting evaluation with {num_workers} workers")

    if total == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "boards": torch.empty(0, 110, 8, 8),
            "policies": torch.empty(0, 64, 64),
            "values": torch.empty(0, 3),
        }, output_path)
        return 0

    # Create temp directory for memmap files
    tmp_dir = tempfile.mkdtemp(prefix="denoisr_memmap_")
    try:
        # Allocate memmap arrays on disk
        boards_mm = np.memmap(
            os.path.join(tmp_dir, "boards.dat"),
            dtype=np.float32, mode="w+", shape=(total, 110, 8, 8),
        )
        policies_mm = np.memmap(
            os.path.join(tmp_dir, "policies.dat"),
            dtype=np.float32, mode="w+", shape=(total, 64, 64),
        )
        values_mm = np.memmap(
            os.path.join(tmp_dir, "values.dat"),
            dtype=np.float32, mode="w+", shape=(total, 3),
        )

        # Small metadata arrays stay in RAM (~200 MB at 16M examples)
        game_ids = np.empty(total, dtype=np.int64)
        piece_counts = np.empty(total, dtype=np.int32)
        eco_codes: list[str | None] = [None] * total

        # Pass 2: stream positions through worker pool into memmap
        idx = 0
        with multiprocessing.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(stockfish_path, stockfish_depth, policy_temperature, label_smoothing),
        ) as pool:
            positions = _stream_positions(pgn_path, max_examples)
            results = pool.imap_unordered(
                _evaluate_position, positions, chunksize=chunksize,
            )
            for board_np, policy_np, (win, draw, loss), gid, eco, pc in tqdm(
                results, total=total, desc="Evaluating positions", unit="pos",
                smoothing=0.1,
            ):
                boards_mm[idx] = board_np
                policies_mm[idx] = policy_np
                values_mm[idx] = (win, draw, loss)
                game_ids[idx] = gid
                eco_codes[idx] = eco
                piece_counts[idx] = pc
                idx += 1

        # Flush memmap to disk before creating torch views
        boards_mm.flush()
        policies_mm.flush()
        values_mm.flush()

        print(f"Evaluated {idx} positions, saving to {output_path}...")

        # Build output dict with zero-copy torch views on memmap.
        # torch.from_numpy on a memmap shares the backing file —
        # torch.save reads pages on demand via the OS page cache.
        result: dict[str, Any] = {
            "boards": torch.from_numpy(np.asarray(boards_mm)),
            "policies": torch.from_numpy(np.asarray(policies_mm)),
            "values": torch.from_numpy(np.asarray(values_mm)),
            "game_ids": torch.from_numpy(game_ids),
            "piece_counts": torch.from_numpy(piece_counts),
        }
        if any(eco is not None for eco in eco_codes):
            result["eco_codes"] = eco_codes

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
        print(f"Saved {idx} examples to {output_path}")
        return idx
    finally:
        # Clean up temp memmap files
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- Public API -----------------------------------------------------------


def _extract_positions(pgn_path: Path, max_positions: int) -> list[_PositionMeta]:
    streamer = SimplePGNStreamer()
    positions: list[_PositionMeta] = []
    pbar = tqdm(total=max_positions, desc="Extracting positions", unit="pos", smoothing=0.3)
    game_id = 0

    for record in streamer.stream(pgn_path):
        moves_so_far: _MoveSeq = []
        board = chess.Board()
        for action in record.actions:
            moves_so_far.append((action.from_square, action.to_square, action.promotion))
            board.push(chess.Move(action.from_square, action.to_square, action.promotion))
            if len(positions) >= max_positions:
                break
            piece_count = bin(board.occupied).count("1")
            positions.append(_PositionMeta(
                moves=list(moves_so_far),
                game_id=game_id,
                eco_code=record.eco_code,
                piece_count=piece_count,
            ))
            pbar.update(1)

        game_id += 1
        if len(positions) >= max_positions:
            break

    pbar.close()
    return positions


def generate_examples(
    pgn_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
    num_workers: int,
    policy_temperature: float = 150.0,
    label_smoothing: float = 0.1,
) -> list[TrainingExample]:
    positions = _extract_positions(pgn_path, max_examples)
    print(f"Extracted {len(positions)} positions, evaluating with {num_workers} workers")

    examples: list[TrainingExample] = []

    with multiprocessing.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(stockfish_path, stockfish_depth, policy_temperature, label_smoothing),
    ) as pool:
        results = pool.imap_unordered(_evaluate_position, positions)
        for board_np, policy_np, (win, draw, loss), gid, eco, pc in tqdm(
            results, total=len(positions), desc="Evaluating positions", unit="pos",
            smoothing=0.1,
        ):
            examples.append(
                TrainingExample(
                    board=BoardTensor(torch.from_numpy(board_np)),
                    policy=PolicyTarget(torch.from_numpy(policy_np)),
                    value=ValueTarget(win=win, draw=draw, loss=loss),
                    game_id=gid,
                    eco_code=eco,
                    piece_count=pc,
                )
            )

    print(f"Generated {len(examples)} training examples")
    return examples


def stack_examples(
    examples: list[TrainingExample],
) -> dict[str, Any]:
    boards = torch.stack([ex.board.data for ex in examples])
    policies = torch.stack([ex.policy.data for ex in examples])
    values = torch.tensor(
        [[ex.value.win, ex.value.draw, ex.value.loss] for ex in examples],
        dtype=torch.float32,
    )
    result: dict[str, Any] = {
        "boards": boards,
        "policies": policies,
        "values": values,
    }
    if any(ex.game_id is not None for ex in examples):
        result["game_ids"] = torch.tensor(
            [ex.game_id if ex.game_id is not None else -1 for ex in examples],
            dtype=torch.int64,
        )
    if any(ex.eco_code is not None for ex in examples):
        result["eco_codes"] = [ex.eco_code for ex in examples]
    if any(ex.piece_count is not None for ex in examples):
        result["piece_counts"] = torch.tensor(
            [ex.piece_count if ex.piece_count is not None else -1 for ex in examples],
            dtype=torch.int32,
        )
    return result


def unstack_examples(
    data: dict[str, Any],
) -> list[TrainingExample]:
    boards = data["boards"]
    policies = data["policies"]
    values = data["values"]
    game_ids = data.get("game_ids")
    eco_codes = data.get("eco_codes")
    piece_counts = data.get("piece_counts")
    n = boards.shape[0]
    return [
        TrainingExample(
            board=BoardTensor(boards[i]),
            policy=PolicyTarget(policies[i]),
            value=ValueTarget(
                win=values[i, 0].item(),
                draw=values[i, 1].item(),
                loss=values[i, 2].item(),
            ),
            game_id=(
                int(game_ids[i].item())
                if game_ids is not None and game_ids[i].item() >= 0
                else None
            ),
            eco_code=(
                eco_codes[i]
                if eco_codes is not None
                else None
            ),
            piece_count=(
                int(piece_counts[i].item())
                if piece_counts is not None and piece_counts[i].item() >= 0
                else None
            ),
        )
        for i in range(n)
    ]


def _default_num_workers() -> int:
    return (os.cpu_count() or 1) * 2 + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data from PGN games with Stockfish evaluation"
    )
    parser.add_argument("--pgn", required=True, help="Path to .pgn or .pgn.zst file")
    parser.add_argument(
        "--stockfish",
        default=None,
        help="Path to Stockfish binary (auto-detected from PATH if omitted)",
    )
    parser.add_argument("--stockfish-depth", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_num_workers(),
        help=f"Worker processes (default: cpu_count*2+1 = {_default_num_workers()})",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/training_data.pt"
    )
    parser.add_argument("--policy-temperature", type=float, default=150.0,
        help="Softmax temperature for policy targets (default: 150.0)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
        help="Label smoothing epsilon (default: 0.1)")
    parser.add_argument("--chunksize", type=int, default=64,
        help="imap_unordered chunksize for worker batching (default: 64)")
    args = parser.parse_args()

    stockfish_path = args.stockfish or shutil.which("stockfish")
    if stockfish_path is None:
        print(
            "Error: Stockfish not found. Install it or pass --stockfish /path/to/stockfish",
            file=sys.stderr,
        )
        sys.exit(1)

    count = generate_to_file(
        pgn_path=Path(args.pgn),
        output_path=Path(args.output),
        stockfish_path=stockfish_path,
        stockfish_depth=args.stockfish_depth,
        max_examples=args.max_examples,
        num_workers=args.workers,
        policy_temperature=args.policy_temperature,
        label_smoothing=args.label_smoothing,
        chunksize=args.chunksize,
    )
    print(f"Done: {count} examples generated.")


if __name__ == "__main__":
    main()
