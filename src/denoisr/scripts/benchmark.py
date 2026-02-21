"""Elo benchmarking — self-contained parallel match against a reference engine.

Runs the trained Denoisr engine against an opponent (e.g. Stockfish) using
parallel game execution with optional SPRT for statistical confidence.
"""

import argparse
import math
import shutil
import sys
from importlib import resources
from pathlib import Path

from denoisr.engine.types import TimeControl
from denoisr.evaluation.benchmark import (
    BenchmarkConfig,
    _default_concurrency,
    run_benchmark,
)


def _parse_time_control(tc_str: str) -> TimeControl:
    """Parse 'base+increment' string into TimeControl."""
    parts = tc_str.split("+")
    base = float(parts[0])
    increment = float(parts[1]) if len(parts) > 1 else 0.0
    return TimeControl(base_seconds=base, increment=increment)


def _default_openings_path() -> Path:
    """Locate the bundled default.epd opening book."""
    ref = resources.files("denoisr.data.openings").joinpath("default.epd")
    return Path(str(ref))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Denoisr against a reference engine"
    )
    parser.add_argument(
        "--engine-cmd", required=True,
        help="Command to run the Denoisr UCI engine",
    )
    parser.add_argument(
        "--engine-args", default="",
        help="Additional args for Denoisr engine (space-separated)",
    )
    parser.add_argument(
        "--opponent-cmd", default=None,
        help="Command to run the opponent engine (default: auto-detect stockfish)",
    )
    parser.add_argument(
        "--opponent-args", default="",
        help="Additional args for opponent engine (space-separated)",
    )
    parser.add_argument(
        "--opponent-elo", type=int, default=None,
        help="Limit opponent strength via UCI_Elo (e.g. 1200)",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--time-control", default="10+0.1",
        help="Time control as 'base+increment' in seconds (default: 10+0.1)",
    )
    parser.add_argument(
        "--openings", type=str, default=None,
        help="Path to EPD opening book (default: bundled 50-position book)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=_default_concurrency(),
        help=f"Parallel games (default: cpu_count*2+1 = {_default_concurrency()})",
    )
    parser.add_argument("--sprt-elo0", type=float, default=None)
    parser.add_argument("--sprt-elo1", type=float, default=None)
    args = parser.parse_args()

    opponent_cmd = args.opponent_cmd or shutil.which("stockfish")
    if opponent_cmd is None:
        print(
            "Error: Stockfish not found. Install it or pass --opponent-cmd",
            file=sys.stderr,
        )
        sys.exit(1)

    openings_path: Path | None
    if args.openings is not None:
        openings_path = Path(args.openings)
    else:
        openings_path = _default_openings_path()

    tc = _parse_time_control(args.time_control)
    engine_args = tuple(args.engine_args.split()) if args.engine_args else ()
    opponent_args = tuple(args.opponent_args.split()) if args.opponent_args else ()

    config = BenchmarkConfig(
        engine_cmd=args.engine_cmd,
        engine_args=engine_args,
        opponent_cmd=opponent_cmd,
        opponent_args=opponent_args,
        opponent_elo=args.opponent_elo,
        games=args.games,
        time_control=tc,
        openings_path=openings_path,
        sprt_elo0=args.sprt_elo0,
        sprt_elo1=args.sprt_elo1,
        concurrency=args.concurrency,
    )

    sprt_msg = ""
    if config.sprt_elo0 is not None and config.sprt_elo1 is not None:
        sprt_msg = f", SPRT[{config.sprt_elo0:.0f},{config.sprt_elo1:.0f}]"
    elo_msg = f" vs Elo {config.opponent_elo}" if config.opponent_elo else ""
    print(
        f"Benchmark: {config.games} games, {config.concurrency} workers, "
        f"TC {args.time_control}{elo_msg}{sprt_msg}"
    )

    def on_game(played: int, w: int, d: int, l: int) -> None:
        from denoisr.engine.elo import compute_elo
        elo, err = compute_elo(w, d, l)
        elo_str = f"{elo:+.1f} +/- {err:.1f}" if not math.isinf(elo) else "N/A"
        print(f"  Game {played}/{config.games}: +{w} ={d} -{l} | Elo: {elo_str}")

    result = run_benchmark(config, on_game=on_game)

    print(
        f"\nResult: +{result.wins} ={result.draws} -{result.losses}"
        f" ({result.games_played} games)"
    )
    if not math.isinf(result.elo_diff):
        print(f"Elo: {result.elo_diff:+.1f} +/- {result.elo_error:.1f}")
    print(f"LOS: {result.los:.1f}%")
    if result.sprt_result is not None:
        print(f"SPRT: {result.sprt_result}")


if __name__ == "__main__":
    main()
