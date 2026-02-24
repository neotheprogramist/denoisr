"""Elo benchmarking — self-contained parallel match against a reference engine.

Runs the trained Denoisr engine against an opponent (e.g. Stockfish) using
parallel game execution with optional SPRT for statistical confidence.

When --baseline-cmd is provided, both the primary engine and baseline are
benchmarked against the same opponent, and a comparison table is printed.
"""

import argparse
import logging
import math
import shlex
import shutil
import sys
from importlib import resources
from pathlib import Path

from denoisr.engine.types import TimeControl
from denoisr.evaluation.analyzer import AnalysisResult, run_analysis
from denoisr.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    _default_concurrency,
    run_benchmark,
)
from denoisr.evaluation.pgn_writer import write_combined_pgn, write_pgn
from denoisr.scripts.interrupts import graceful_main

logger = logging.getLogger(__name__)


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


def _log_result(label: str, result: BenchmarkResult) -> None:
    logger.info("--- %s ---", label)
    logger.info("Score: %s (%d games)", _format_score(result), result.games_played)
    logger.info("Elo:   %s", _format_elo(result))
    logger.info("LOS:   %.1f%%", result.los)
    if result.sprt_result is not None:
        logger.info("SPRT:  %s", result.sprt_result)


def _make_on_game(label: str, total_games: int) -> callable:
    def on_game(played: int, wins: int, draws: int, losses: int) -> None:
        from denoisr.engine.elo import compute_elo

        elo, err = compute_elo(wins, draws, losses)
        elo_str = f"{elo:+.1f} +/- {err:.1f}" if not math.isinf(elo) else "N/A"
        logger.info(
            "[%s] Game %d/%d: +%d =%d -%d | Elo: %s",
            label,
            played,
            total_games,
            wins,
            draws,
            losses,
            elo_str,
        )

    return on_game


def _save_pgn(result: BenchmarkResult, pgn_dir: Path, label: str) -> None:
    """Write individual + combined PGN files for a benchmark result."""
    sub = pgn_dir / label
    for game in result.completed_games:
        write_pgn(game, sub)
    combined = pgn_dir / f"{label}_all.pgn"
    write_combined_pgn(result.completed_games, combined)
    logger.info(
        "PGN saved: %s/ (%d games) + %s", sub, len(result.completed_games), combined
    )


def _run_and_log_analysis(
    result: BenchmarkResult,
    label: str,
    stockfish_cmd: str,
    depth: int,
    concurrency: int,
) -> AnalysisResult:
    """Run ACPL analysis on a benchmark result and log summary."""
    logger.info(
        "Analyzing %s (%d games, depth %d)...",
        label,
        len(result.completed_games),
        depth,
    )
    analysis = run_analysis(
        result.completed_games,
        stockfish_cmd=stockfish_cmd,
        depth=depth,
        concurrency=concurrency,
    )
    logger.info("%s ACPL: %.1f", label, analysis.overall_acpl)
    logger.info("%s Est. Elo: %.0f", label, analysis.estimated_elo)
    logger.info("%s Blunders: %d", label, analysis.total_blunders)
    return analysis


def _log_comparison(
    result: BenchmarkResult,
    baseline: BenchmarkResult,
    engine_analysis: AnalysisResult | None = None,
    baseline_analysis: AnalysisResult | None = None,
) -> None:
    sep = "=" * 60
    lines = [
        sep,
        "  Comparison",
        sep,
        f"  {'':18s} {'Engine':>16s}   {'Baseline':>16s}",
        f"  {'Score':18s} {_format_score(result):>16s}   "
        f"{_format_score(baseline):>16s}",
        f"  {'Elo vs opponent':18s} {_format_elo(result):>16s}   "
        f"{_format_elo(baseline):>16s}",
        f"  {'LOS':18s} {result.los:>15.1f}%   {baseline.los:>15.1f}%",
    ]

    engine_score = result.wins + result.draws * 0.5
    baseline_score = baseline.wins + baseline.draws * 0.5
    engine_pct = engine_score / max(result.games_played, 1) * 100
    baseline_pct = baseline_score / max(baseline.games_played, 1) * 100
    lines.append(f"  {'Score %':18s} {engine_pct:>15.1f}%   {baseline_pct:>15.1f}%")

    if engine_analysis is not None and baseline_analysis is not None:
        lines.extend(
            [
                f"  {'ACPL':18s} {engine_analysis.overall_acpl:>15.1f}   "
                f"{baseline_analysis.overall_acpl:>15.1f}",
                f"  {'Est. Elo':18s} {engine_analysis.estimated_elo:>15.0f}   "
                f"{baseline_analysis.estimated_elo:>15.0f}",
                f"  {'Blunders':18s} {engine_analysis.total_blunders:>15d}   "
                f"{baseline_analysis.total_blunders:>15d}",
            ]
        )

    if engine_pct > baseline_pct:
        lines.append(
            f"\n  Engine scores {engine_pct - baseline_pct:.1f}pp better than baseline."
        )
    elif baseline_pct > engine_pct:
        lines.append(
            f"\n  Baseline scores {baseline_pct - engine_pct:.1f}pp better than engine."
        )
    else:
        lines.append("\n  Engine and baseline score identically.")

    logger.info("\n%s", "\n".join(lines))


@graceful_main("denoisr-benchmark", logger=logger)
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
        "--head-to-head",
        action="store_true",
        help="Play engine vs baseline directly (requires --baseline-cmd)",
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
        help=f"Parallel games (default: {_default_concurrency()})",
    )
    parser.add_argument("--sprt-elo0", type=float, default=None)
    parser.add_argument("--sprt-elo1", type=float, default=None)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for engine startup/UCI handshake (default: 120)",
    )
    parser.add_argument(
        "--pgn-out",
        type=str,
        default=None,
        help="Directory to save PGN files (one per game + combined)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run Stockfish ACPL analysis after games complete",
    )
    parser.add_argument(
        "--analysis-depth",
        type=int,
        default=12,
        help="Stockfish analysis depth (default: 12)",
    )
    parser.add_argument(
        "--analysis-concurrency",
        type=int,
        default=None,
        help="Parallel Stockfish analysis workers (default: same as --concurrency)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Validate flags ---
    if args.head_to_head and not args.baseline_cmd:
        logger.error("--head-to-head requires --baseline-cmd")
        sys.exit(1)

    # --- Resolve opponent ---
    if args.head_to_head:
        # In head-to-head mode the baseline IS the opponent
        opponent_cmd = args.baseline_cmd
    else:
        opponent_cmd = args.opponent_cmd or shutil.which("stockfish")
        if opponent_cmd is None:
            logger.error("Stockfish not found. Install it or pass --opponent-cmd")
            sys.exit(1)

    if args.opponent_elo is not None and args.opponent_elo < 1320:
        logger.warning(
            "--opponent-elo %d is below Stockfish's minimum UCI_Elo (1320). "
            "Clamping to 1320.",
            args.opponent_elo,
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
    mode_msg = " (head-to-head)" if args.head_to_head else ""
    baseline_msg = " + baseline" if args.baseline_cmd and not args.head_to_head else ""
    logger.info(
        "Benchmark: %d games, %d workers, TC %s%s%s%s%s%s",
        config.games,
        config.concurrency,
        args.time_control,
        elo_msg,
        skill_msg,
        sprt_msg,
        mode_msg,
        baseline_msg,
    )

    # --- Resolve analysis settings ---
    pgn_dir = Path(args.pgn_out) if args.pgn_out else None
    analysis_concurrency = args.analysis_concurrency or args.concurrency
    stockfish_for_analysis = (
        args.opponent_cmd or shutil.which("stockfish") or "stockfish"
    )

    sep = "=" * 60
    if args.head_to_head:
        # --- Head-to-head: engine vs baseline directly ---
        logger.info(
            "\n%s\n  Engine:   %s\n  Baseline: %s\n%s",
            sep,
            args.engine_cmd,
            args.baseline_cmd,
            sep,
        )
        result = run_benchmark(config, on_game=_make_on_game("h2h", config.games))
        _log_result("Engine vs Baseline", result)

        if pgn_dir is not None:
            _save_pgn(result, pgn_dir, "head_to_head")

        if args.analyze:
            _run_and_log_analysis(
                result,
                "Engine (h2h)",
                stockfish_for_analysis,
                args.analysis_depth,
                analysis_concurrency,
            )
    else:
        # --- Run primary benchmark ---
        logger.info("\n%s\n  Engine: %s\n%s", sep, args.engine_cmd, sep)
        result = run_benchmark(config, on_game=_make_on_game("engine", config.games))

        # --- Run baseline benchmark ---
        baseline_result: BenchmarkResult | None = None
        if args.baseline_cmd:
            baseline_cmd_exe, baseline_args = _split_cmd(args.baseline_cmd)
            if args.baseline_args:
                baseline_args = baseline_args + tuple(shlex.split(args.baseline_args))

            baseline_config = BenchmarkConfig(
                engine_cmd=baseline_cmd_exe,
                engine_args=baseline_args,
                **shared,
            )

            logger.info("\n%s\n  Baseline: %s\n%s", sep, args.baseline_cmd, sep)
            baseline_result = run_benchmark(
                baseline_config,
                on_game=_make_on_game("baseline", baseline_config.games),
            )

        # --- PGN export ---
        if pgn_dir is not None:
            _save_pgn(result, pgn_dir, "engine")
            if baseline_result is not None:
                _save_pgn(baseline_result, pgn_dir, "baseline")

        # --- ACPL analysis ---
        engine_analysis: AnalysisResult | None = None
        baseline_analysis: AnalysisResult | None = None
        if args.analyze:
            engine_analysis = _run_and_log_analysis(
                result,
                "Engine",
                stockfish_for_analysis,
                args.analysis_depth,
                analysis_concurrency,
            )
            if baseline_result is not None:
                baseline_analysis = _run_and_log_analysis(
                    baseline_result,
                    "Baseline",
                    stockfish_for_analysis,
                    args.analysis_depth,
                    analysis_concurrency,
                )

        # --- Results ---
        _log_result("Engine", result)

        if baseline_result is not None:
            _log_result("Baseline", baseline_result)
            _log_comparison(result, baseline_result, engine_analysis, baseline_analysis)


if __name__ == "__main__":
    main()
