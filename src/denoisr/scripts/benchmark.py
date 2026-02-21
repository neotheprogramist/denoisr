"""Elo benchmarking — self-contained parallel match against a reference engine.

Runs the trained Denoisr engine against an opponent (e.g. Stockfish) using
parallel game execution with optional SPRT for statistical confidence.

When --baseline-cmd is provided, both the primary engine and baseline are
benchmarked against the same opponent, and a comparison table is printed.
"""

import argparse
import math
import shlex
import shutil
import sys
from importlib import resources
from pathlib import Path

from denoisr.engine.types import TimeControl
from denoisr.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
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


def _split_cmd(cmd_str: str) -> tuple[str, tuple[str, ...]]:
    """Split a command string into (executable, args_tuple)."""
    parts = shlex.split(cmd_str)
    return parts[0], tuple(parts[1:])


def _format_elo(result: BenchmarkResult) -> str:
    if math.isinf(result.elo_diff):
        return "N/A"
    return f"{result.elo_diff:+.1f} +/- {result.elo_error:.1f}"


def _format_score(result: BenchmarkResult) -> str:
    return f"+{result.wins} ={result.draws} -{result.losses}"


def _print_result(label: str, result: BenchmarkResult) -> None:
    print(f"\n--- {label} ---")
    print(f"Score: {_format_score(result)} ({result.games_played} games)")
    print(f"Elo:   {_format_elo(result)}")
    print(f"LOS:   {result.los:.1f}%")
    if result.sprt_result is not None:
        print(f"SPRT:  {result.sprt_result}")


def _make_on_game(
    label: str, total_games: int
) -> callable:
    def on_game(played: int, wins: int, draws: int, losses: int) -> None:
        from denoisr.engine.elo import compute_elo

        elo, err = compute_elo(wins, draws, losses)
        elo_str = f"{elo:+.1f} +/- {err:.1f}" if not math.isinf(elo) else "N/A"
        print(
            f"  [{label}] Game {played}/{total_games}: "
            f"+{wins} ={draws} -{losses} | Elo: {elo_str}"
        )

    return on_game


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Denoisr against a reference engine"
    )
    parser.add_argument(
        "--engine-cmd",
        required=True,
        help="Command to run the Denoisr UCI engine",
    )
    parser.add_argument(
        "--engine-args",
        default="",
        help="Additional args for Denoisr engine (space-separated)",
    )
    parser.add_argument(
        "--baseline-cmd",
        default=None,
        help="Command for baseline engine (e.g. random model) for comparison",
    )
    parser.add_argument(
        "--baseline-args",
        default="",
        help="Additional args for baseline engine (space-separated)",
    )
    parser.add_argument(
        "--opponent-cmd",
        default=None,
        help="Command to run the opponent engine (default: auto-detect stockfish)",
    )
    parser.add_argument(
        "--opponent-args",
        default="",
        help="Additional args for opponent engine (space-separated)",
    )
    parser.add_argument(
        "--opponent-elo",
        type=int,
        default=None,
        help="Limit opponent strength via UCI_Elo (min 1320 for Stockfish)",
    )
    parser.add_argument(
        "--opponent-skill",
        type=int,
        default=None,
        help="Stockfish Skill Level 0-20 (0 = weakest, much weaker than UCI_Elo)",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--time-control",
        default="10+0.1",
        help="Time control as 'base+increment' in seconds (default: 10+0.1)",
    )
    parser.add_argument(
        "--openings",
        type=str,
        default=None,
        help="Path to EPD opening book (default: bundled 50-position book)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_default_concurrency(),
        help=f"Parallel games (default: cpu_count*2+1 = {_default_concurrency()})",
    )
    parser.add_argument("--sprt-elo0", type=float, default=None)
    parser.add_argument("--sprt-elo1", type=float, default=None)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for engine startup/UCI handshake (default: 120)",
    )
    args = parser.parse_args()

    # --- Resolve opponent ---
    opponent_cmd = args.opponent_cmd or shutil.which("stockfish")
    if opponent_cmd is None:
        print(
            "Error: Stockfish not found. Install it or pass --opponent-cmd",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.opponent_elo is not None and args.opponent_elo < 1320:
        print(
            f"Warning: --opponent-elo {args.opponent_elo} is below Stockfish's "
            "minimum UCI_Elo (1320). Clamping to 1320.",
            file=sys.stderr,
        )
        args.opponent_elo = 1320

    # --- Build opponent UCI options ---
    opponent_options: list[tuple[str, str]] = []
    if args.opponent_skill is not None:
        opponent_options.append(("Skill Level", str(args.opponent_skill)))

    # --- Openings ---
    openings_path: Path | None
    if args.openings is not None:
        openings_path = Path(args.openings)
    else:
        openings_path = _default_openings_path()

    tc = _parse_time_control(args.time_control)

    # --- Parse command strings ---
    engine_cmd, engine_args = _split_cmd(args.engine_cmd)
    if args.engine_args:
        engine_args = engine_args + tuple(shlex.split(args.engine_args))

    opponent_cmd_exe, opponent_args = _split_cmd(opponent_cmd)
    if args.opponent_args:
        opponent_args = opponent_args + tuple(shlex.split(args.opponent_args))

    # --- Shared config kwargs ---
    shared = dict(
        opponent_cmd=opponent_cmd_exe,
        opponent_args=opponent_args,
        opponent_elo=args.opponent_elo,
        opponent_options=tuple(opponent_options),
        games=args.games,
        time_control=tc,
        openings_path=openings_path,
        sprt_elo0=args.sprt_elo0,
        sprt_elo1=args.sprt_elo1,
        concurrency=args.concurrency,
        startup_timeout=args.startup_timeout,
    )

    config = BenchmarkConfig(
        engine_cmd=engine_cmd,
        engine_args=engine_args,
        **shared,
    )

    # --- Header ---
    sprt_msg = ""
    if config.sprt_elo0 is not None and config.sprt_elo1 is not None:
        sprt_msg = f", SPRT[{config.sprt_elo0:.0f},{config.sprt_elo1:.0f}]"
    elo_msg = f" vs Elo {config.opponent_elo}" if config.opponent_elo else ""
    skill_msg = (
        f" Skill {args.opponent_skill}" if args.opponent_skill is not None else ""
    )
    baseline_msg = (
        " + baseline" if args.baseline_cmd else ""
    )
    print(
        f"Benchmark: {config.games} games, {config.concurrency} workers, "
        f"TC {args.time_control}{elo_msg}{skill_msg}{sprt_msg}{baseline_msg}"
    )

    # --- Run primary benchmark ---
    print(f"\n{'=' * 60}")
    print("  Engine: " + args.engine_cmd)
    print(f"{'=' * 60}")
    result = run_benchmark(config, on_game=_make_on_game("engine", config.games))

    # --- Run baseline benchmark ---
    baseline_result: BenchmarkResult | None = None
    if args.baseline_cmd:
        baseline_cmd, baseline_args = _split_cmd(args.baseline_cmd)
        if args.baseline_args:
            baseline_args = baseline_args + tuple(shlex.split(args.baseline_args))

        baseline_config = BenchmarkConfig(
            engine_cmd=baseline_cmd,
            engine_args=baseline_args,
            **shared,
        )

        print(f"\n{'=' * 60}")
        print("  Baseline: " + args.baseline_cmd)
        print(f"{'=' * 60}")
        baseline_result = run_benchmark(
            baseline_config,
            on_game=_make_on_game("baseline", baseline_config.games),
        )

    # --- Results ---
    _print_result("Engine", result)

    if baseline_result is not None:
        _print_result("Baseline", baseline_result)

        # Comparison table
        print(f"\n{'=' * 60}")
        print("  Comparison")
        print(f"{'=' * 60}")
        print(f"  {'':18s} {'Engine':>16s}   {'Baseline':>16s}")
        print(f"  {'Score':18s} {_format_score(result):>16s}   "
              f"{_format_score(baseline_result):>16s}")
        print(f"  {'Elo vs opponent':18s} {_format_elo(result):>16s}   "
              f"{_format_elo(baseline_result):>16s}")
        print(f"  {'LOS':18s} {result.los:>15.1f}%   "
              f"{baseline_result.los:>15.1f}%")

        # Win rate comparison
        engine_score = result.wins + result.draws * 0.5
        baseline_score = baseline_result.wins + baseline_result.draws * 0.5
        engine_pct = engine_score / max(result.games_played, 1) * 100
        baseline_pct = baseline_score / max(baseline_result.games_played, 1) * 100
        print(f"  {'Score %':18s} {engine_pct:>15.1f}%   {baseline_pct:>15.1f}%")

        if engine_pct > baseline_pct:
            print(f"\n  Engine scores {engine_pct - baseline_pct:.1f}pp better "
                  "than baseline.")
        elif baseline_pct > engine_pct:
            print(f"\n  Baseline scores {baseline_pct - engine_pct:.1f}pp better "
                  "than engine.")
        else:
            print("\n  Engine and baseline score identically.")


if __name__ == "__main__":
    main()
