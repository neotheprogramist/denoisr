"""Generate training data: PGN + Stockfish -> stacked tensors on disk.

Produces a .pt file with:
    boards:   Tensor[N, 110, 8, 8]
    policies: Tensor[N, 64, 64]
    values:   Tensor[N, 3]        (win, draw, loss per example)

Stockfish evaluation is parallelized across multiple worker processes,
each owning its own Stockfish subprocess.
"""

import atexit
import multiprocessing
import os
import argparse
import shutil
import sys
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
    args = parser.parse_args()

    stockfish_path = args.stockfish or shutil.which("stockfish")
    if stockfish_path is None:
        print(
            "Error: Stockfish not found. Install it or pass --stockfish /path/to/stockfish",
            file=sys.stderr,
        )
        sys.exit(1)

    examples = generate_examples(
        Path(args.pgn),
        stockfish_path,
        args.stockfish_depth,
        args.max_examples,
        args.workers,
        policy_temperature=args.policy_temperature,
        label_smoothing=args.label_smoothing,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stack_examples(examples), output_path)
    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
