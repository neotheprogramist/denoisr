"""Export PyTorch checkpoint to safetensors for MLX inference."""

import json
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from denoisr_chess.config import load_checkpoint
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)

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


@graceful_main("denoisr-chess-export-mlx", logger=log)
def main() -> None:
    """CLI entry point for denoisr-chess-export-mlx."""
    load_env_file()
    log_path = configure_logging()
    parser = build_parser("Export PyTorch checkpoint to safetensors for MLX inference")
    add_env_argument(
        parser,
        "--checkpoint",
        env_var="DENOISR_EXPORT_CHECKPOINT",
        help="PyTorch checkpoint path",
    )
    add_env_argument(
        parser,
        "--output",
        env_var="DENOISR_EXPORT_OUTPUT",
        help="Output safetensors path",
    )
    args = parser.parse_args()
    log.info("logging to %s", log_path)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error("%s not found", checkpoint_path)
        sys.exit(1)

    export_weights(checkpoint_path, Path(args.output))


if __name__ == "__main__":
    main()
