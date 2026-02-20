"""Launch the Denoisr chess GUI."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoisr Chess GUI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint (pre-fills the GUI field)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "diffusion"],
        default="single",
        help="Engine inference mode",
    )
    args = parser.parse_args()

    from denoisr.gui.app import DenoisrApp

    app = DenoisrApp()
    if args.checkpoint:
        app._ckpt_var.set(args.checkpoint)
    app._engine_mode_var.set(args.mode)
    app.run()
