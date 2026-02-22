"""Sort PGN games into Elo-stratified binary .games files.

Reads a .pgn or .pgn.zst file and writes separate .games files
per Elo range under the output directory.
"""

import argparse
import logging
from pathlib import Path

from denoisr.data.game_format import GameBatchWriter
from denoisr.data.pgn_streamer import SimplePGNStreamer

log = logging.getLogger(__name__)


def _parse_ranges(raw: str) -> list[tuple[int, int | None]]:
    """Parse '0-1200,1200-1600,2400+' into [(0,1200),(1200,1600),(2400,None)]."""
    ranges: list[tuple[int, int | None]] = []
    for part in raw.split(","):
        part = part.strip()
        if part.endswith("+"):
            ranges.append((int(part[:-1]), None))
        else:
            lo, hi = part.split("-")
            ranges.append((int(lo), int(hi)))
    return ranges


def _bucket_name(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo}+"
    return f"{lo}-{hi}"


def _min_elo(white_elo: int | None, black_elo: int | None) -> int | None:
    if white_elo is not None and black_elo is not None:
        return min(white_elo, black_elo)
    return white_elo or black_elo


def sort_pgn_to_games(
    pgn_path: Path,
    output_dir: Path,
    ranges: list[tuple[int, int | None]],
    max_buffer_bytes: int = 16 * 1024**3,
) -> dict[str, int]:
    """Sort PGN games into binary .games files by Elo range.

    Returns a dict mapping bucket names to game counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, GameBatchWriter] = {}
    for lo, hi in ranges:
        name = _bucket_name(lo, hi)
        path = output_dir / f"{name}.games"
        writer = GameBatchWriter(path, max_buffer_bytes=max_buffer_bytes)
        writer.__enter__()
        writers[name] = writer

    skipped = 0
    streamer = SimplePGNStreamer()
    try:
        for record in streamer.stream(pgn_path):
            elo = _min_elo(record.white_elo, record.black_elo)
            if elo is None:
                skipped += 1
                continue

            bucket: str | None = None
            for lo, hi in ranges:
                if hi is None:
                    if elo >= lo:
                        bucket = _bucket_name(lo, hi)
                        break
                elif lo <= elo < hi:
                    bucket = _bucket_name(lo, hi)
                    break

            if bucket is None:
                skipped += 1
                continue

            writers[bucket].write(record)
    finally:
        for writer in writers.values():
            writer.close()

    counts = {name: writer.count for name, writer in writers.items()}
    log.info("Sorted games by Elo (skipped %d without Elo):", skipped)
    for name, count in counts.items():
        log.info("  %s: %d games", name, count)
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Sort PGN games by Elo into binary .games files"
    )
    parser.add_argument("--input", required=True, help="Path to .pgn or .pgn.zst")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--ranges",
        default="0-1200,1200-1600,1600-2000,2000-2400,2400+",
        help="Comma-separated Elo ranges (default: 0-1200,1200-1600,1600-2000,2000-2400,2400+)",
    )
    args = parser.parse_args()

    ranges = _parse_ranges(args.ranges)
    sort_pgn_to_games(Path(args.input), Path(args.output), ranges)


if __name__ == "__main__":
    main()
