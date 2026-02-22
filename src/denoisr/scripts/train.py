"""Unified training pipeline: PGN -> Phase 3 in one command."""

import argparse
import logging
from pathlib import Path

from denoisr.pipeline.config import load_config
from denoisr.pipeline.runner import PipelineRunner


def main() -> None:
    """Parse CLI arguments and run the full Denoisr training pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full Denoisr training pipeline from a TOML config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline.toml",
        help="Path to pipeline TOML config (default: pipeline.toml)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore saved state and start fresh",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma-separated list of steps to run "
            "(fetch,sort,init,phase1,phase2,phase3)"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(Path(args.config))
    only = frozenset(args.only.split(",")) if args.only else None
    runner = PipelineRunner(cfg, restart=args.restart, only=only)
    runner.run()


if __name__ == "__main__":
    main()
