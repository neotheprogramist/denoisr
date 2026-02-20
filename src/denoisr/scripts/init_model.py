"""Initialize a random (untrained) model checkpoint.

Creates a checkpoint with random weights so you can immediately
play against the engine before running any training. The untrained
model plays random-looking moves — compare with a trained model
to see how much training improves play.
"""

import argparse
from pathlib import Path

from denoisr.scripts.config import (
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a random model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/random_model.pt",
        help="Output checkpoint path",
    )
    add_model_args(parser)
    args = parser.parse_args()

    cfg = config_from_args(args)
    print(f"Initializing random model: d_s={cfg.d_s}, layers={cfg.num_layers}")

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
    print(f"Random model saved to {args.output}")
    print("Play against it: uv run denoisr-play --checkpoint " + args.output)


if __name__ == "__main__":
    main()
