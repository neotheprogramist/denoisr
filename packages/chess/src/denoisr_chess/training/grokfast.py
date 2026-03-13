"""Grokfast: EMA gradient filtering for accelerated grokking.

Amplifies slow-varying gradient components (the generalizing circuit's signal)
while leaving fast-varying components (memorization) alone.

Reference: Lee et al. (2024) "Grokfast: Accelerated Grokking by Amplifying
Slow Gradients"
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GrokfastFilter:
    """EMA-based gradient filter that amplifies slow-varying components.

    Usage::

        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        # After loss.backward() and scaler.unscale_():
        gf.apply(model)
        # Then clip_grad_norm_() and optimizer.step()
    """

    def __init__(self, alpha: float = 0.98, lamb: float = 2.0) -> None:
        self.alpha = alpha
        self.lamb = lamb
        self.grads: dict[str, Tensor] = {}

    def reset(self) -> None:
        """Drop all EMA buffers, e.g. after a numerical instability event."""
        self.grads.clear()

    def apply(self, model: nn.Module, *, key_prefix: str = "") -> None:
        prefix = f"{key_prefix}." if key_prefix else ""
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            full_name = f"{prefix}{name}" if prefix else name
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                # Non-finite gradient: skip this parameter, don't corrupt EMA
                self.grads.pop(full_name, None)
                continue
            if full_name not in self.grads:
                self.grads[full_name] = grad.clone()
            else:
                ema = self.grads[full_name]
                if ema.shape != grad.shape:
                    raise ValueError(
                        f"Grokfast EMA shape mismatch for {full_name}: "
                        f"expected {ema.shape}, got {grad.shape}. "
                        "This indicates a model architecture change mid-training."
                    )
                if ema.device != grad.device:
                    raise ValueError(
                        f"Grokfast EMA device mismatch for {full_name}: "
                        f"expected {ema.device}, got {grad.device}."
                    )
                if not torch.isfinite(ema).all():
                    raise RuntimeError(
                        f"Grokfast EMA buffer for {full_name} contains non-finite values. "
                        "Training has diverged."
                    )
                ema.mul_(self.alpha).add_(grad, alpha=1 - self.alpha)
            param.grad.add_(self.grads[full_name], alpha=self.lamb)
