"""Generate training data: PGN + Stockfish -> stacked tensors on disk.

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
from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.data.stockfish_oracle import StockfishOracle
from denoisr.scripts.config import resolve_workers
from denoisr.scripts.interrupts import graceful_main

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
@dataclass(frozen=True)
class _PositionMeta:
    fen: str
    game_id: int
    eco_code: str | None
    piece_count: int


_EvalResultWithMeta = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    tuple[float, float, float],
    int,  # game_id
    str | None,  # eco_code
    int,  # piece_count
]


def _evaluate_position(meta: _PositionMeta) -> _EvalResultWithMeta:
    if _oracle is None or _encoder is None:
        raise RuntimeError("Worker not initialized")
    board = chess.Board(meta.fen)
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


def _count_positions(
    pgn_path: Path,
    max_positions: int,
) -> int:
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


def _stream_positions(
    pgn_path: Path,
    max_positions: int,
    sample_rate: float = 1.0,
) -> Iterator[_PositionMeta]:
    """Stream positions from a PGN file with random sub-sampling.

    sample_rate < 1.0 randomly skips positions for uniform-ish sampling
    across the entire PGN file rather than taking the first N sequentially.
    """
    streamer = SimplePGNStreamer()
    count = 0
    game_id = 0

    for record in streamer.stream(pgn_path):
        board = chess.Board()
        for action in record.actions:
            board.push(
                chess.Move(action.from_square, action.to_square, action.promotion)
            )
            if count >= max_positions:
                break
            # Random sampling: skip positions with probability (1 - sample_rate)
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue
            piece_count = bin(board.occupied).count("1")
            yield _PositionMeta(
                fen=board.fen(en_passant="fen"),
                game_id=game_id,
                eco_code=record.eco_code,
                piece_count=piece_count,
            )
            count += 1

        game_id += 1
        if count >= max_positions:
            break


_GIB = 1024 * 1024 * 1024
_CHUNK_FORMAT = "chunked_v1"


def _estimate_dataset_gib(num_examples: int, num_planes: int) -> float:
    bytes_per_example = ((num_planes * 8 * 8) + (64 * 64) + 3) * 4
    return (num_examples * bytes_per_example) / _GIB


def _write_chunk_file(
    *,
    chunk_dir: Path,
    chunk_idx: int,
    count: int,
    boards_buf: npt.NDArray[np.float32],
    policies_buf: npt.NDArray[np.float32],
    values_buf: npt.NDArray[np.float32],
    game_ids_buf: npt.NDArray[np.int64],
    piece_counts_buf: npt.NDArray[np.int32],
    eco_codes_buf: list[str | None],
) -> Path:
    """Write one chunk file and return its path."""
    chunk_path = chunk_dir / f"chunk_{chunk_idx:06d}.pt"
    payload: dict[str, Any] = {
        "boards": torch.from_numpy(boards_buf[:count].copy()),
        "policies": torch.from_numpy(policies_buf[:count].copy()),
        "values": torch.from_numpy(values_buf[:count].copy()),
        "game_ids": torch.from_numpy(game_ids_buf[:count].copy()),
        "piece_counts": torch.from_numpy(piece_counts_buf[:count].copy()),
    }
    eco_codes = eco_codes_buf[:count]
    if any(eco is not None for eco in eco_codes):
        payload["eco_codes"] = eco_codes.copy()
    torch.save(payload, chunk_path)
    return chunk_path


def _generate_to_file_chunked(
    *,
    pgn_path: Path,
    output_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
    num_workers: int,
    policy_temperature: float,
    label_smoothing: float,
    chunksize: int,
    chunk_examples: int,
) -> int:
    """Generate data directly to on-disk chunk files + manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_path.parent / f"{output_path.stem}_chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    num_planes = ExtendedBoardEncoder().num_planes
    boards_buf = np.empty((chunk_examples, num_planes, 8, 8), dtype=np.float32)
    policies_buf = np.empty((chunk_examples, 64, 64), dtype=np.float32)
    values_buf = np.empty((chunk_examples, 3), dtype=np.float32)
    game_ids_buf = np.empty(chunk_examples, dtype=np.int64)
    piece_counts_buf = np.empty(chunk_examples, dtype=np.int32)
    eco_codes_buf: list[str | None] = [None] * chunk_examples

    chunk_records: list[dict[str, Any]] = []
    total_written = 0
    chunk_idx = 0
    fill = 0

    def flush_chunk() -> None:
        nonlocal fill, chunk_idx, total_written, eco_codes_buf
        if fill == 0:
            return
        chunk_path = _write_chunk_file(
            chunk_dir=chunk_dir,
            chunk_idx=chunk_idx,
            count=fill,
            boards_buf=boards_buf,
            policies_buf=policies_buf,
            values_buf=values_buf,
            game_ids_buf=game_ids_buf,
            piece_counts_buf=piece_counts_buf,
            eco_codes_buf=eco_codes_buf,
        )
        rel_path = str(chunk_path.relative_to(output_path.parent))
        chunk_records.append({"path": rel_path, "count": fill})
        total_written += fill
        log.info("Wrote chunk %d (%d examples): %s", chunk_idx, fill, chunk_path)
        chunk_idx += 1
        fill = 0
        eco_codes_buf = [None] * chunk_examples

    with multiprocessing.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(stockfish_path, stockfish_depth, policy_temperature, label_smoothing),
    ) as pool:
        positions = _stream_positions(
            pgn_path, max_positions=max_examples, sample_rate=1.0
        )
        results = pool.imap_unordered(
            _evaluate_position, positions, chunksize=chunksize
        )
        for board_np, policy_np, (win, draw, loss), gid, eco, pc in tqdm(
            results,
            total=max_examples,
            desc="Evaluating positions",
            unit="pos",
            smoothing=0.1,
        ):
            boards_buf[fill] = board_np
            policies_buf[fill] = policy_np
            values_buf[fill] = (win, draw, loss)
            game_ids_buf[fill] = gid
            piece_counts_buf[fill] = pc
            eco_codes_buf[fill] = eco
            fill += 1
            if fill >= chunk_examples:
                flush_chunk()
        flush_chunk()

    manifest = {
        "format": _CHUNK_FORMAT,
        "num_planes": num_planes,
        "total_examples": total_written,
        "chunks": chunk_records,
    }
    torch.save(manifest, output_path)
    log.info(
        "Saved chunked manifest with %d examples across %d chunks to %s",
        total_written,
        len(chunk_records),
        output_path,
    )
    return total_written


def generate_to_file(
    pgn_path: Path,
    output_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
    num_workers: int,
    policy_temperature: float = 80.0,
    label_smoothing: float = 0.02,
    chunksize: int = 64,
    seed: int | None = None,
    scratch_dir: Path | None = None,
    chunk_examples: int = 0,
) -> int:
    """Stream positions through Stockfish workers into disk-backed memmap arrays.

    Returns the number of examples written. RSS stays at ~4-8 GB even for
    tens of millions of examples because the large arrays (boards, policies,
    values) live in memory-mapped files that the OS pages in/out on demand.
    """
    if seed is not None:
        random.seed(seed)

    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if not stockfish_path.strip():
        raise ValueError(
            "stockfish_path must be a non-empty executable path or command"
        )
    if chunk_examples < 0:
        raise ValueError("chunk_examples must be >= 0")

    if chunk_examples > 0:
        log.info(
            "Chunked generation enabled (chunk_examples=%d, workers=%d)",
            chunk_examples,
            num_workers,
        )
        return _generate_to_file_chunked(
            pgn_path=pgn_path,
            output_path=output_path,
            stockfish_path=stockfish_path,
            stockfish_depth=stockfish_depth,
            max_examples=max_examples,
            num_workers=num_workers,
            policy_temperature=policy_temperature,
            label_smoothing=label_smoothing,
            chunksize=chunksize,
            chunk_examples=chunk_examples,
        )

    # Pass 1: count positions to pre-allocate exact array sizes
    log.info("Pass 1: counting positions...")
    total = _count_positions(pgn_path, max_examples)
    if total <= 0:
        total = 0

    log.info(
        "Found %d positions, starting evaluation with %d workers",
        total,
        num_workers,
    )

    num_planes = ExtendedBoardEncoder().num_planes
    estimated_dataset_gib = _estimate_dataset_gib(total, num_planes)
    log.info("Estimated tensor payload size: %.2f GiB", estimated_dataset_gib)

    if total == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "boards": torch.empty(0, num_planes, 8, 8),
                "policies": torch.empty(0, 64, 64),
                "values": torch.empty(0, 3),
            },
            output_path,
        )
        return 0

    # Create temp directory for memmap files
    scratch_root = scratch_dir or output_path.parent
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="denoisr_memmap_", dir=str(scratch_root))
    try:
        # Allocate memmap arrays on disk
        boards_mm = np.memmap(
            os.path.join(tmp_dir, "boards.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(total, num_planes, 8, 8),
        )
        policies_mm = np.memmap(
            os.path.join(tmp_dir, "policies.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(total, 64, 64),
        )
        values_mm = np.memmap(
            os.path.join(tmp_dir, "values.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(total, 3),
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
            initargs=(
                stockfish_path,
                stockfish_depth,
                policy_temperature,
                label_smoothing,
            ),
        ) as pool:
            # Compute sample_rate for random sub-sampling across the PGN.
            # If we found more positions than needed, randomly skip some
            # so we get a uniform sample rather than the first N positions.
            sample_rate = (
                min(1.0, max_examples / max(total, 1)) if total > max_examples else 1.0
            )
            positions = _stream_positions(
                pgn_path,
                max_examples,
                sample_rate=sample_rate,
            )
            results = pool.imap_unordered(
                _evaluate_position,
                positions,
                chunksize=chunksize,
            )
            for board_np, policy_np, (win, draw, loss), gid, eco, pc in tqdm(
                results,
                total=total,
                desc="Evaluating positions",
                unit="pos",
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


@graceful_main("denoisr-generate-data", logger=log)
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
    parser.add_argument("--max-examples", type=int, default=1_000_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (0 = auto: 64)",
    )
    parser.add_argument("--output", type=str, default="outputs/training_data.pt")
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=80.0,
        help="Softmax temperature for policy targets (default: 80.0)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.02,
        help="Label smoothing epsilon (default: 0.02)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="imap_unordered chunksize for worker batching (default: 64)",
    )
    parser.add_argument(
        "--chunk-examples",
        type=int,
        default=0,
        help=(
            "Save data in shard files with this many examples per chunk "
            "(0 = single-file output)"
        ),
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default="outputs/scratch",
        help=("Directory for temporary memmap files (default: outputs/scratch)"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None = random)",
    )
    args = parser.parse_args()

    stockfish_path = args.stockfish or shutil.which("stockfish")
    if stockfish_path is None:
        log.error(
            "Stockfish not found. Install it or pass --stockfish /path/to/stockfish",
        )
        sys.exit(1)

    count = generate_to_file(
        pgn_path=Path(args.pgn),
        output_path=Path(args.output),
        stockfish_path=stockfish_path,
        stockfish_depth=args.stockfish_depth,
        max_examples=args.max_examples,
        num_workers=resolve_workers(args.workers),
        policy_temperature=args.policy_temperature,
        label_smoothing=args.label_smoothing,
        chunksize=args.chunksize,
        seed=args.seed,
        scratch_dir=Path(args.scratch_dir),
        chunk_examples=args.chunk_examples,
    )
    log.info("Done: %d examples generated.", count)


if __name__ == "__main__":
    main()
