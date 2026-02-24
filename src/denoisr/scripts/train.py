"""Unified training pipeline: PGN -> Phase 3 in one command."""

import logging
from pathlib import Path

from denoisr.pipeline.config import load_config
from denoisr.pipeline.runner import PipelineRunner
from denoisr.scripts.interrupts import graceful_main
from denoisr.scripts.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)

log = logging.getLogger(__name__)


@graceful_main("denoisr-train", logger=log)
def main() -> None:
    """Parse CLI arguments and run the full Denoisr training pipeline."""
    load_env_file()
    parser = build_parser("Run the full Denoisr training pipeline from env + optional TOML")
    add_env_argument(
        parser,
        "--config",
        env_var="DENOISR_CONFIG",
        type=str,
        required=False,
        help="Optional path to pipeline TOML config (env overrides always win)",
    )
    add_env_argument(
        parser,
        "--restart",
        action="store_true",
        env_var="DENOISR_RESTART",
        help="Ignore saved state and start fresh",
    )
    add_env_argument(
        parser,
        "--only",
        type=str,
        env_var="DENOISR_ONLY",
        required=False,
        help=("Comma-separated list of steps to run (fetch,init,phase1,phase2,phase3)"),
    )
    args = parser.parse_args()
    log_path = configure_logging()
    log.info("logging to %s", log_path)

    cfg_path = Path(args.config) if args.config else None
    if cfg_path is not None and not cfg_path.exists():
        log.warning(
            "Config file %s not found; continuing with env/default settings",
            cfg_path,
        )
        cfg_path = None
    cfg = load_config(cfg_path)
    if cfg_path is None:
        log.info("No config file provided; using env/default pipeline settings")
    else:
        log.info("Loaded base pipeline settings from %s with env overrides", cfg_path)
    only = frozenset(args.only.split(",")) if args.only else None
    runner = PipelineRunner(cfg, restart=args.restart, only=only)
    try:
        runner.run()
    except KeyboardInterrupt:
        runner._save_state()
        log.warning(
            "Pipeline interrupted. State saved to %s",
            runner.state_path,
        )
        raise


if __name__ == "__main__":
    main()
