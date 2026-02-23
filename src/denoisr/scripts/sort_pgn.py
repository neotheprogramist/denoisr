"""Sort PGN games into Elo-stratified buckets.

Reads a .pgn or .pgn.zst file and writes separate .pgn.zst files
per Elo range under the output directory.
"""

import argparse
import io
import logging
from pathlib import Path

import chess.pgn
import zstandard as zstd

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Sort PGN games by Elo into separate files"
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open output compressors
    writers: dict[str, tuple[io.BufferedWriter, zstd.ZstdCompressor]] = {}
    file_handles: list[io.BufferedWriter] = []
    for lo, hi in ranges:
        name = _bucket_name(lo, hi)
        path = output_dir / f"{name}.pgn.zst"
        fh = open(path, "wb")  # noqa: SIM115
        file_handles.append(fh)
        compressor = zstd.ZstdCompressor(level=3)
        writers[name] = (fh, compressor)

    # Track stats
    counts: dict[str, int] = {_bucket_name(lo, hi): 0 for lo, hi in ranges}
    move_counts: dict[str, int] = {_bucket_name(lo, hi): 0 for lo, hi in ranges}
    skipped = 0

    streamer = SimplePGNStreamer()
    try:
        for record in streamer.stream(Path(args.input)):
            elo = _min_elo(record.white_elo, record.black_elo)
            if elo is None:
                skipped += 1
                continue

            # Find matching bucket
            bucket_name: str | None = None
            for lo, hi in ranges:
                if hi is None:
                    if elo >= lo:
                        bucket_name = _bucket_name(lo, hi)
                        break
                elif lo <= elo < hi:
                    bucket_name = _bucket_name(lo, hi)
                    break

            if bucket_name is None:
                skipped += 1
                continue

            # Write game as PGN text to compressed stream
            game = chess.pgn.Game()
            game.headers["Result"] = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}.get(
                record.result, "*"
            )
            if record.eco_code:
                game.headers["ECO"] = record.eco_code
            if record.white_elo is not None:
                game.headers["WhiteElo"] = str(record.white_elo)
            if record.black_elo is not None:
                game.headers["BlackElo"] = str(record.black_elo)

            # Replay moves
            node: chess.pgn.GameNode = game
            board = chess.Board()
            for action in record.actions:
                move = chess.Move(action.from_square, action.to_square, action.promotion)
                node = node.add_variation(move)
                board.push(move)

            pgn_text = str(game) + "\n\n"
            fh, compressor = writers[bucket_name]
            fh.write(compressor.compress(pgn_text.encode("utf-8")))
            counts[bucket_name] += 1
            move_counts[bucket_name] += len(record.actions)

    finally:
        for fh in file_handles:
            fh.close()

    # Report
    log.info("Sorted games by Elo (skipped %d without Elo):", skipped)
    for name in counts:
        log.info("  %s: %d games, %d moves", name, counts[name], move_counts[name])


if __name__ == "__main__":
    main()
