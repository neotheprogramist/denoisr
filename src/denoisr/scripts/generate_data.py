"""Generate training data: PGN + Stockfish -> stacked tensors on disk.

Produces a .pt file with:
    format: "chunked_v1"
    chunks: [{path, count}, ...]
and a sibling chunk directory containing:
    boards:   Tensor[N, C, 8, 8]  (C = num encoder planes, default 122)
    policies: Tensor[N, 64, 64]
    values:   Tensor[N, 3]        (win, draw, loss per example)

Stockfish evaluation is parallelized across multiple worker processes,
each owning its own Stockfish subprocess.

The main() entry point uses generate_to_file() which always writes chunked
output (single code path).
"""

import argparse
import atexit
import logging
import multiprocessing
import os
import random
import shutil
import sys
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
from denoisr.scripts.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)

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


_CHUNK_FORMAT = "chunked_v1"
_GIB = float(1024 * 1024 * 1024)
_DEFAULT_MAX_RAM_GIB = 64.0
_CHUNK_BUFFER_RAM_FRACTION = 0.40
_WORKER_BASELINE_RAM_GIB = 0.20
_RUNTIME_RESERVE_RAM_GIB = 4.0
_INFLIGHT_RESULT_FACTOR = 2.0


@dataclass
class _TenStepProgressTracker:
    """Emit at most 10 INFO progress logs at 10% milestones."""

    total: int
    next_step: int = 1

    def maybe_log(self, completed: int) -> None:
        if completed < 0:
            raise ValueError("completed must be >= 0")
        while self.next_step <= 10 and (completed * 10) >= (
            self.total * self.next_step
        ):
            pct = self.next_step * 10
            log.info(
                "Generation progress step %d/10 (%d%%): %d/%d examples",
                self.next_step,
                pct,
                completed,
                self.total,
            )
            self.next_step += 1


def _estimate_chunk_buffer_gib(num_examples: int, num_planes: int) -> float:
    bytes_per_example = ((num_planes * 8 * 8) + (64 * 64) + 3) * 4
    return (num_examples * bytes_per_example) / _GIB


def _plan_chunk_examples_for_memory(
    *,
    requested_chunk_examples: int,
    max_ram_gib: float,
    num_planes: int,
    num_workers: int,
    chunksize: int,
) -> tuple[int, float]:
    """Return RAM-safe chunk_examples and estimated peak RSS in GiB."""
    if max_ram_gib <= 0:
        raise ValueError("max_ram_gib must be > 0")
    if requested_chunk_examples < 1:
        raise ValueError("requested_chunk_examples must be >= 1")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1")

    eval_result_bytes = int(((num_planes * 8 * 8) + (64 * 64) + 3) * 4)
    buffer_example_bytes = eval_result_bytes + 8 + 4  # game_ids + piece_counts

    max_ram_bytes = int(max_ram_gib * _GIB)
    worker_bytes = int(num_workers * _WORKER_BASELINE_RAM_GIB * _GIB)
    inflight_bytes = int(
        num_workers * chunksize * eval_result_bytes * _INFLIGHT_RESULT_FACTOR
    )
    reserve_bytes = int(_RUNTIME_RESERVE_RAM_GIB * _GIB)

    non_chunk_bytes = worker_bytes + inflight_bytes + reserve_bytes
    available_chunk_bytes = max_ram_bytes - non_chunk_bytes
    if available_chunk_bytes <= 0:
        raise ValueError(
            "RAM budget is too low for generation overheads: "
            f"max_ram_gib={max_ram_gib:.1f}, workers={num_workers}, "
            f"chunksize={chunksize}"
        )

    chunk_fraction_cap_bytes = int(max_ram_bytes * _CHUNK_BUFFER_RAM_FRACTION)
    chunk_budget_bytes = min(available_chunk_bytes, chunk_fraction_cap_bytes)
    safe_chunk_examples = max(1, chunk_budget_bytes // buffer_example_bytes)
    effective_chunk_examples = min(requested_chunk_examples, safe_chunk_examples)
    est_peak_gib = (
        non_chunk_bytes + (effective_chunk_examples * buffer_example_bytes)
    ) / _GIB
    return effective_chunk_examples, est_peak_gib


def _resolve_max_ram_gib(max_ram_gib: float | None) -> float:
    if max_ram_gib is not None:
        return max_ram_gib
    raw = os.environ.get("DENOISR_MAX_RAM_GIB", "").strip()
    if not raw:
        return _DEFAULT_MAX_RAM_GIB
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid DENOISR_MAX_RAM_GIB={raw!r}") from exc


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

    def _to_chunk_tensor(arr: npt.NDArray[Any]) -> torch.Tensor:
        if count >= arr.shape[0]:
            # Full chunk: avoid extra allocation and serialize directly.
            return torch.from_numpy(arr)
        # Tail chunk: copy to avoid serializing the full backing storage.
        return torch.from_numpy(arr[:count].copy())

    payload: dict[str, Any] = {
        "boards": _to_chunk_tensor(boards_buf),
        "policies": _to_chunk_tensor(policies_buf),
        "values": _to_chunk_tensor(values_buf),
        "game_ids": _to_chunk_tensor(game_ids_buf),
        "piece_counts": _to_chunk_tensor(piece_counts_buf),
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
    use_tqdm: bool,
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
    processed = 0
    progress_tracker = _TenStepProgressTracker(total=max_examples)

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
            disable=not use_tqdm,
        ):
            boards_buf[fill] = board_np
            policies_buf[fill] = policy_np
            values_buf[fill] = (win, draw, loss)
            game_ids_buf[fill] = gid
            piece_counts_buf[fill] = pc
            eco_codes_buf[fill] = eco
            fill += 1
            processed += 1
            progress_tracker.maybe_log(processed)
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
    chunksize: int = 1_024,
    seed: int | None = None,
    chunk_examples: int = 1_000_000,
    use_tqdm: bool = False,
    max_ram_gib: float | None = None,
) -> int:
    """Generate chunked training data and write a manifest."""
    if seed is not None:
        random.seed(seed)

    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1")
    if not stockfish_path.strip():
        raise ValueError(
            "stockfish_path must be a non-empty executable path or command"
        )
    if chunk_examples < 1:
        raise ValueError("chunk_examples must be >= 1")
    resolved_max_ram_gib = _resolve_max_ram_gib(max_ram_gib)
    if resolved_max_ram_gib <= 0:
        raise ValueError("max_ram_gib must be > 0")

    num_planes = ExtendedBoardEncoder().num_planes
    effective_chunk_examples, est_peak_gib = _plan_chunk_examples_for_memory(
        requested_chunk_examples=chunk_examples,
        max_ram_gib=resolved_max_ram_gib,
        num_planes=num_planes,
        num_workers=num_workers,
        chunksize=chunksize,
    )
    if effective_chunk_examples < chunk_examples:
        log.warning(
            "Reducing chunk_examples from %d to %d to fit RAM budget %.1f GiB "
            "(estimated peak %.2f GiB)",
            chunk_examples,
            effective_chunk_examples,
            resolved_max_ram_gib,
            est_peak_gib,
        )
    chunk_gib = _estimate_chunk_buffer_gib(effective_chunk_examples, num_planes)
    log.info(
        "Chunked generation: max_examples=%d workers=%d chunksize=%d "
        "chunk_examples=%d (~%.2f GiB chunk buffers, est_peak=%.2f/%.2f GiB)",
        max_examples,
        num_workers,
        chunksize,
        effective_chunk_examples,
        chunk_gib,
        est_peak_gib,
        resolved_max_ram_gib,
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
        chunk_examples=effective_chunk_examples,
        use_tqdm=use_tqdm,
    )


@graceful_main("denoisr-generate-data", logger=log)
def main() -> None:
    load_env_file()
    log_path = configure_logging()
    parser = build_parser(
        "Generate training data from PGN games with Stockfish evaluation"
    )
    add_env_argument(
        parser,
        "--pgn",
        env_var="DENOISR_PGN_PATH",
        help="Path to .pgn or .pgn.zst file",
    )
    add_env_argument(
        parser,
        "--stockfish",
        env_var="DENOISR_STOCKFISH_PATH",
        default=None,
        required=False,
        help="Path to Stockfish binary (auto-detected from PATH if omitted)",
    )
    add_env_argument(
        parser,
        "--stockfish-depth",
        env_var="DENOISR_STOCKFISH_DEPTH",
        type=int,
    )
    add_env_argument(
        parser,
        "--max-examples",
        env_var="DENOISR_MAX_EXAMPLES",
        type=int,
    )
    add_env_argument(
        parser,
        "--workers",
        env_var="DENOISR_WORKERS",
        type=int,
        help="Worker processes",
    )
    add_env_argument(
        parser,
        "--output",
        env_var="DENOISR_DATA_OUTPUT",
        type=str,
    )
    add_env_argument(
        parser,
        "--policy-temperature",
        env_var="DENOISR_POLICY_TEMPERATURE",
        type=float,
        help="Softmax temperature for policy targets",
    )
    add_env_argument(
        parser,
        "--label-smoothing",
        env_var="DENOISR_LABEL_SMOOTHING",
        type=float,
        help="Label smoothing epsilon",
    )
    add_env_argument(
        parser,
        "--chunksize",
        env_var="DENOISR_CHUNKSIZE",
        type=int,
        help="imap_unordered chunksize for worker batching",
    )
    add_env_argument(
        parser,
        "--chunk-examples",
        env_var="DENOISR_CHUNK_EXAMPLES",
        type=int,
        help="Save data in shard files with this many examples per chunk",
    )
    add_env_argument(
        parser,
        "--max-ram-gib",
        env_var="DENOISR_MAX_RAM_GIB",
        type=float,
        default=_DEFAULT_MAX_RAM_GIB,
        required=False,
        help="RAM budget used to auto-cap chunk size during generation",
    )
    add_env_argument(
        parser,
        "--seed",
        env_var="DENOISR_SEED",
        type=int,
        default=None,
        required=False,
        help="Random seed for reproducible sampling (default: None = random)",
    )
    add_env_argument(
        parser,
        "--tqdm",
        env_var="DENOISR_TQDM",
        action=argparse.BooleanOptionalAction,
        default=False,
        required=False,
        help="show tqdm progress bars",
    )
    args = parser.parse_args()
    log.info("logging to %s", log_path)

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
        chunk_examples=args.chunk_examples,
        use_tqdm=args.tqdm,
        max_ram_gib=args.max_ram_gib,
    )
    log.info("Done: %d examples generated.", count)


if __name__ == "__main__":
    main()
