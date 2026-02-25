"""Unified training pipeline: PGN -> Phase 3 in one command."""

import logging

from denoisr.pipeline.config import load_config
from denoisr.pipeline.runner import PipelineRunner
from denoisr.scripts.config import validate_required_env as validate_training_env
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
    parser = build_parser("Run the full Denoisr training pipeline from env")
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
    validate_training_env(include_phase3=True)

    log_path = configure_logging()
    log.info("logging to %s", log_path)

    cfg = load_config()
    log.info("Loaded pipeline settings from strict env validation")
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
