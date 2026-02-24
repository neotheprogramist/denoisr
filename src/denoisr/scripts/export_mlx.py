"""Export PyTorch checkpoint to safetensors for MLX inference."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from denoisr.scripts.config import load_checkpoint
from denoisr.scripts.interrupts import graceful_main

log = logging.getLogger(__name__)


def export_weights(checkpoint_path: Path, output_path: Path) -> None:
    """Convert a PyTorch checkpoint to safetensors format.

    Merges all sub-model state dicts (encoder, backbone, policy_head,
    value_head) into a single flat dict with dotted prefixes, then writes
    to safetensors. Also writes a companion JSON config file.
    """
    cfg, state = load_checkpoint(checkpoint_path, torch.device("cpu"))

    weights: dict[str, torch.Tensor] = {}
    for prefix in ("encoder", "backbone", "policy_head", "value_head"):
        if prefix not in state:
            continue
        for key, tensor in state[prefix].items():
            weights[f"{prefix}.{key}"] = tensor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output_path))

    config_path = output_path.with_suffix(".json")
    config_path.write_text(json.dumps(cfg.__dict__, indent=2))
    log.info("Exported %d tensors to %s", len(weights), output_path)
    log.info("Config saved to %s", config_path)


@graceful_main("denoisr-export-mlx", logger=log)
def main() -> None:
    """CLI entry point for denoisr-export-mlx."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Export PyTorch checkpoint to safetensors for MLX inference",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="PyTorch checkpoint path",
    )
    parser.add_argument(
        "--output",
        default="outputs/model.safetensors",
        help="Output safetensors path",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error("%s not found", checkpoint_path)
        sys.exit(1)

    export_weights(checkpoint_path, Path(args.output))


if __name__ == "__main__":
    main()
