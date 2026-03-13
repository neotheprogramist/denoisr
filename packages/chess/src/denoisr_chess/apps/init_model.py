"""Initialize a random (untrained) model checkpoint.

Creates a checkpoint with random weights so you can immediately
play against the engine before running any training. The untrained
model plays random-looking moves — compare with a trained model
to see how much training improves play.
"""

import logging
from pathlib import Path

from denoisr_chess.config import (
    add_model_args,
    build_backbone,
    build_consistency,
    build_diffusion,
    build_encoder,
    build_policy_head,
    build_value_head,
    build_world_model,
    config_from_args,
    save_checkpoint,
)
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)

log = logging.getLogger(__name__)


@graceful_main("denoisr-chess-init", logger=log)
def main() -> None:
    load_env_file()
    log_path = configure_logging()
    parser = build_parser("Initialize a random model checkpoint")
    add_env_argument(
        parser,
        "--output",
        env_var="DENOISR_INIT_OUTPUT",
        type=str,
        help="Output checkpoint path",
    )
    add_model_args(parser)
    args = parser.parse_args()
    log.info("logging to %s", log_path)

    cfg = config_from_args(args)
    log.info("Initializing random model: d_s=%d, layers=%d", cfg.d_s, cfg.num_layers)

    encoder = build_encoder(cfg)
    backbone = build_backbone(cfg)
    policy_head = build_policy_head(cfg)
    value_head = build_value_head(cfg)
    world_model = build_world_model(cfg)
    diffusion = build_diffusion(cfg)
    consistency = build_consistency(cfg)

    save_checkpoint(
        Path(args.output),
        cfg,
        encoder=encoder.state_dict(),
        backbone=backbone.state_dict(),
        policy_head=policy_head.state_dict(),
        value_head=value_head.state_dict(),
        world_model=world_model.state_dict(),
        diffusion=diffusion.state_dict(),
        consistency=consistency.state_dict(),
    )
    log.info("Random model saved to %s", args.output)
    log.info("Play against it: uv run denoisr-chess-play --checkpoint %s", args.output)


if __name__ == "__main__":
    main()
