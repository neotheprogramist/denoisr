"""Elo benchmarking via cutechess-cli.

Runs the trained Denoisr engine against a reference engine (e.g. Stockfish)
using cutechess-cli with optional SPRT for statistical confidence.
"""

import argparse
import subprocess
import sys

from denoisr.evaluation.benchmark import (
    BenchmarkConfig,
    build_cutechess_command,
    parse_cutechess_output,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Denoisr via cutechess-cli")
    parser.add_argument(
        "--engine-cmd", required=True, help="Command to run the Denoisr UCI engine"
    )
    parser.add_argument(
        "--opponent-cmd",
        default="stockfish",
        help="Command to run the opponent UCI engine",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--time-control", default="10+0.1")
    parser.add_argument("--sprt-elo0", type=int, default=None)
    parser.add_argument("--sprt-elo1", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cutechess-cli command without running it",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        engine_cmd=args.engine_cmd,
        opponent_cmd=args.opponent_cmd,
        games=args.games,
        time_control=args.time_control,
        sprt_elo0=args.sprt_elo0,
        sprt_elo1=args.sprt_elo1,
        concurrency=args.concurrency,
    )

    cmd = build_cutechess_command(config)

    if args.dry_run:
        print(cmd)
        return

    print(f"Running: {cmd}")
    proc = subprocess.run(
        cmd.split(), capture_output=True, text=True, check=False
    )
    print(proc.stdout)

    if proc.returncode != 0:
        print(f"cutechess-cli exited with code {proc.returncode}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

    result = parse_cutechess_output(proc.stdout)
    if "elo_diff" in result:
        print(f"\nElo: {result['elo_diff']:.1f} +/- {result['elo_error']:.1f}")
    if "los" in result:
        print(f"LOS: {result['los']:.1f}%")
    if "sprt_result" in result:
        print(f"SPRT: {result['sprt_result']}")


if __name__ == "__main__":
    main()
