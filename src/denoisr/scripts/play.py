"""UCI chess engine interface.

Loads a trained checkpoint and runs the UCI protocol loop,
allowing connection to any UCI-compatible GUI (CuteChess, Arena, etc.).

Model loading is deferred until the first 'isready' command so that the
'uci' → 'uciok' handshake completes instantly.

Supports two inference modes:
  --mode single   Single-pass (fastest, weakest)
  --mode diffusion   Diffusion-enhanced with anytime search (stronger)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from denoisr.inference.uci import run_uci_loop
from denoisr.scripts.interrupts import graceful_main
from denoisr.scripts.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import chess


log = logging.getLogger(__name__)


@graceful_main("denoisr-play")
def main() -> None:
    load_env_file()
    parser = build_parser("Denoisr UCI chess engine")
    add_env_argument(
        parser,
        "--checkpoint",
        env_var="DENOISR_PLAY_CHECKPOINT",
        help="Path to model checkpoint",
    )
    add_env_argument(
        parser,
        "--mode",
        env_var="DENOISR_PLAY_MODE",
        choices=["single", "diffusion"],
        help="Inference mode",
    )
    add_env_argument(
        parser,
        "--denoising-steps",
        env_var="DENOISR_PLAY_DENOISING_STEPS",
        type=int,
        help="Denoising steps for diffusion mode (more = stronger, slower)",
    )
    args = parser.parse_args()
    log_path = configure_logging()
    log.info("logging to %s", log_path)

    # Mutable container — populated by load_model() on first 'isready'.
    _engine_fn: list[Callable[[chess.Board], chess.Move]] = []

    def load_model() -> None:
        """Heavy init: import torch, load checkpoint, build model."""
        from pathlib import Path

        from denoisr.scripts.config import (
            build_backbone,
            build_board_encoder,
            build_diffusion,
            build_encoder,
            build_policy_head,
            build_schedule,
            build_value_head,
            detect_device,
            load_checkpoint,
        )

        device = detect_device()
        cfg, state = load_checkpoint(Path(args.checkpoint), device)
        log.info("loaded checkpoint=%s mode=%s device=%s", args.checkpoint, args.mode, device)

        encoder = build_encoder(cfg).to(device)
        backbone = build_backbone(cfg).to(device)
        policy_head = build_policy_head(cfg).to(device)
        value_head = build_value_head(cfg).to(device)

        encoder.load_state_dict(state["encoder"])
        backbone.load_state_dict(state["backbone"])
        policy_head.load_state_dict(state["policy_head"])
        value_head.load_state_dict(state["value_head"])

        board_encoder = build_board_encoder(cfg)

        if args.mode == "diffusion":
            from denoisr.inference.diffusion_engine import DiffusionChessEngine

            diffusion = build_diffusion(cfg).to(device)
            diffusion.load_state_dict(state["diffusion"])
            schedule = build_schedule(cfg).to(device)

            _engine_fn.append(
                DiffusionChessEngine(
                    encoder=encoder,
                    backbone=backbone,
                    policy_head=policy_head,
                    value_head=value_head,
                    diffusion=diffusion,
                    schedule=schedule,
                    board_encoder=board_encoder,
                    device=device,
                    num_denoising_steps=args.denoising_steps,
                ).select_move
            )
        else:
            from denoisr.inference.engine import ChessEngine

            _engine_fn.append(
                ChessEngine(
                    encoder=encoder,
                    backbone=backbone,
                    policy_head=policy_head,
                    value_head=value_head,
                    board_encoder=board_encoder,
                    device=device,
                ).select_move
            )

    def select_move(board: chess.Board) -> chess.Move:
        return _engine_fn[0](board)

    run_uci_loop(engine_select_move_fn=select_move, on_isready=load_model)


if __name__ == "__main__":
    main()
