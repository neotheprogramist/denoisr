"""Launch the Denoisr chess GUI."""

from __future__ import annotations

import logging

from denoisr.scripts.interrupts import graceful_main
from denoisr.scripts.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)

log = logging.getLogger(__name__)


@graceful_main("denoisr-gui")
def main() -> None:
    load_env_file()
    parser = build_parser("Denoisr Chess GUI")
    add_env_argument(
        parser,
        "--checkpoint",
        env_var="DENOISR_GUI_CHECKPOINT",
        type=str,
        required=False,
        default="",
        help="Path to model checkpoint (pre-fills the GUI field)",
    )
    add_env_argument(
        parser,
        "--mode",
        env_var="DENOISR_GUI_MODE",
        type=str,
        choices=["single", "diffusion"],
        required=False,
        default="single",
        help="Engine inference mode",
    )
    args = parser.parse_args()
    log_path = configure_logging()
    log.info("logging to %s", log_path)

    from denoisr.gui.app import DenoisrApp

    app = DenoisrApp()
    app._engine_mode_var.set(args.mode)
    if args.checkpoint:
        app._ckpt_var.set(args.checkpoint)
        app.auto_start()
    app.run()
