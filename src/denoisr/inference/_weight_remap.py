"""Weight key remapping from PyTorch nn.Sequential to MLX flat attributes.

Pure dict manipulation — no MLX or PyTorch dependency, testable everywhere.
"""

from __future__ import annotations


def _remap_sequential_keys(
    weights: dict[str, object],
    prefix: str,
    seq_name: str,
    mapping: list[tuple[int, str]],
) -> None:
    """Remap PyTorch nn.Sequential indexed keys to named MLX linear layers.

    PyTorch Sequential indices include param-less layers (e.g., Mish at index 1),
    so we need explicit (old_index, new_name) pairs rather than enumerate.

    E.g., "encoder.global_embed.0.weight" -> "encoder.global_embed_0.weight"
         "encoder.global_embed.2.weight" -> "encoder.global_embed_2.weight"
    """
    for old_idx, new_name in mapping:
        for param in ("weight", "bias"):
            old_key = f"{prefix}.{seq_name}.{old_idx}.{param}"
            new_key = f"{prefix}.{new_name}.{param}"
            if old_key in weights:
                weights[new_key] = weights.pop(old_key)


def _remap_all_keys(weights: dict[str, object], num_layers: int) -> None:
    """Remap all PyTorch Sequential keys to MLX flat attribute names."""
    _remap_sequential_keys(
        weights, "encoder", "global_embed",
        [(0, "global_embed_0"), (2, "global_embed_2")],
    )
    _remap_sequential_keys(
        weights, "backbone.smolgen", "compress",
        [(0, "compress_0")],
    )
    for i in range(num_layers):
        _remap_sequential_keys(
            weights, f"backbone.layers.{i}", "ffn",
            [(0, "ffn_0"), (2, "ffn_2")],
        )
