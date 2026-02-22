"""Generate training data: .games binary files + Stockfish -> stacked tensors on disk.

Produces a .pt file with:
    boards:   Tensor[N, C, 8, 8]  (C = num encoder planes, default 122)
    policies: Tensor[N, 64, 64]
    values:   Tensor[N, 3]        (win, draw, loss per example)

Stockfish evaluation is parallelized across multiple worker processes,
each owning its own Stockfish subprocess.

The main() entry point uses generate_to_file() which streams results into
disk-backed numpy memory-mapped arrays, keeping RSS at ~4-8 GB even for
tens of millions of examples.
"""

import atexit
import logging
import multiprocessing
import os
import argparse
import random
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
from denoisr.data.stockfish_oracle import StockfishOracle
from denoisr.scripts.config import resolve_workers

log = logging.getLogger(__name__)

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


def _is_tactical(board: chess.Board) -> bool:
    """Classify a position as tactical if any piece is en prise or endgame."""
    piece_count = bin(board.occupied).count("1")
    # Endgame: total piece count <= 10 (excluding kings = already counted)
    if piece_count <= 10:
        return True
    # Check for hanging pieces (attacked by opponent, not defended)
    for pt in chess.PIECE_TYPES:
        if pt == chess.KING:
            continue
        for color in chess.COLORS:
            opp = not color
            for sq in board.pieces(pt, color):
                if board.is_attacked_by(opp, sq) and not board.attackers(color, sq):
                    return True
    return False


def _count_positions_from_games(data_dir: Path, max_positions: int) -> int:
    """Count total positions across all .games files via header-only scan."""
    from denoisr.data.game_format import count_positions

    total = 0
    for gf in sorted(data_dir.glob("*.games")):
        total += count_positions(gf)
        if total >= max_positions:
            return max_positions
    return min(total, max_positions)


def _stream_game_files(
    data_dir: Path,
    max_positions: int,
    min_elo: int | None = None,
    tactical_fraction: float = 0.25,
    seed: int | None = None,
) -> Iterator[_PositionMeta]:
    """Stream positions from .games binary files with tactical enrichment."""
    from denoisr.data.game_format import read_game_records

    game_files = sorted(data_dir.glob("*.games"))
    if not game_files:
        return

    if seed is not None:
        random.seed(seed)

    positions_per_bucket = max_positions // len(game_files)
    count = 0
    tactical_count = 0
    game_id = 0

    for gf in game_files:
        bucket_count = 0
        for record in read_game_records(gf):
            if bucket_count >= positions_per_bucket:
                break
            if count >= max_positions:
                break

            # Elo filtering
            if min_elo is not None:
                w, b = record.white_elo, record.black_elo
                if w is not None and b is not None:
                    if min(w, b) < min_elo:
                        game_id += 1
                        continue
                elif w is not None:
                    if w < min_elo:
                        game_id += 1
                        continue
                elif b is not None:
                    if b < min_elo:
                        game_id += 1
                        continue
                else:
                    game_id += 1
                    continue

            moves_so_far: _MoveSeq = []
            board = chess.Board()
            for action in record.actions:
                moves_so_far.append((action.from_square, action.to_square, action.promotion))
                board.push(chess.Move(action.from_square, action.to_square, action.promotion))
                if count >= max_positions or bucket_count >= positions_per_bucket:
                    break

                is_tac = _is_tactical(board)

                # Enforce tactical fraction
                if not is_tac and tactical_fraction > 0 and count > 0:
                    current_frac = tactical_count / count
                    if current_frac < tactical_fraction and random.random() > 0.5:
                        continue

                piece_count = bin(board.occupied).count("1")
                yield _PositionMeta(
                    moves=list(moves_so_far),
                    game_id=game_id,
                    eco_code=record.eco_code,
                    piece_count=piece_count,
                )
                count += 1
                bucket_count += 1
                if is_tac:
                    tactical_count += 1

            game_id += 1
            if count >= max_positions:
                break


def generate_to_file(
    data_dir: Path,
    output_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
    num_workers: int,
    policy_temperature: float = 80.0,
    label_smoothing: float = 0.02,
    chunksize: int = 64,
    min_elo: int | None = None,
    tactical_fraction: float = 0.0,
    seed: int | None = None,
) -> int:
    """Stream positions through Stockfish workers into disk-backed memmap arrays.

    Returns the number of examples written. RSS stays at ~4-8 GB even for
    tens of millions of examples because the large arrays (boards, policies,
    values) live in memory-mapped files that the OS pages in/out on demand.

    Reads positions from .games binary files in *data_dir*, with optional
    Elo filtering and tactical enrichment.
    """
    if seed is not None:
        random.seed(seed)

    # Pass 1: count positions to pre-allocate exact array sizes
    log.info("Pass 1: counting positions...")
    total = _count_positions_from_games(data_dir, max_examples)
    log.info("Found %d positions, starting evaluation with %d workers", total, num_workers)

    num_planes = ExtendedBoardEncoder().num_planes

    if total == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "boards": torch.empty(0, num_planes, 8, 8),
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
            dtype=np.float32, mode="w+", shape=(total, num_planes, 8, 8),
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
            positions = _stream_game_files(
                data_dir, max_examples,
                min_elo=min_elo,
                tactical_fraction=tactical_fraction,
                seed=seed,
            )
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

        log.info("Evaluated %d positions, saving to %s...", idx, output_path)

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
        log.info("Saved %d examples to %s", idx, output_path)
        return idx
    finally:
        # Clean up temp memmap files
        shutil.rmtree(tmp_dir, ignore_errors=True)




def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate training data from .games files with Stockfish evaluation"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory with .games files"
    )
    parser.add_argument(
        "--stockfish",
        default=None,
        help="Path to Stockfish binary (auto-detected from PATH if omitted)",
    )
    parser.add_argument("--stockfish-depth", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=1_000_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (0 = auto: cpu_count*2+1)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/training_data.pt"
    )
    parser.add_argument("--policy-temperature", type=float, default=80.0,
        help="Softmax temperature for policy targets (default: 80.0)")
    parser.add_argument("--label-smoothing", type=float, default=0.02,
        help="Label smoothing epsilon (default: 0.02)")
    parser.add_argument("--chunksize", type=int, default=64,
        help="imap_unordered chunksize for worker batching (default: 64)")
    parser.add_argument(
        "--min-elo", type=int, default=None,
        help="Minimum Elo to include games (min of white/black Elo, default: None)",
    )
    parser.add_argument(
        "--tactical-fraction", type=float, default=0.25,
        help="Target fraction of tactical positions in dataset (default: 0.25)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for sampling (default: None = random)",
    )
    args = parser.parse_args()

    stockfish_path = args.stockfish or shutil.which("stockfish")
    if stockfish_path is None:
        log.error(
            "Stockfish not found. Install it or pass --stockfish /path/to/stockfish",
        )
        sys.exit(1)

    count = generate_to_file(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        stockfish_path=stockfish_path,
        stockfish_depth=args.stockfish_depth,
        max_examples=args.max_examples,
        num_workers=resolve_workers(args.workers),
        policy_temperature=args.policy_temperature,
        label_smoothing=args.label_smoothing,
        chunksize=args.chunksize,
        min_elo=args.min_elo,
        tactical_fraction=args.tactical_fraction,
        seed=args.seed,
    )
    log.info("Done: %d examples generated.", count)


if __name__ == "__main__":
    main()
