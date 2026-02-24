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

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                self.grads.pop(name, None)
                continue
            if name not in self.grads:
                self.grads[name] = grad.clone()
            else:
                ema = self.grads[name]
                if (
                    ema.shape != grad.shape
                    or ema.device != grad.device
                    or not torch.isfinite(ema).all()
                ):
                    self.grads[name] = grad.clone()
                else:
                    ema.mul_(self.alpha).add_(grad, alpha=1 - self.alpha)
            param.grad.add_(self.grads[name], alpha=self.lamb)
