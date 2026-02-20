"""Grokfast: EMA gradient filtering for accelerated grokking.

Amplifies slow-varying gradient components (the generalizing circuit's signal)
while leaving fast-varying components (memorization) alone.

Reference: Lee et al. (2024) "Grokfast: Accelerated Grokking by Amplifying
Slow Gradients"
"""

from __future__ import annotations

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

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if name not in self.grads:
                self.grads[name] = param.grad.data.detach().clone()
            else:
                self.grads[name].mul_(self.alpha).add_(
                    param.grad.data.detach(), alpha=1 - self.alpha
                )
                param.grad.data.add_(self.grads[name], alpha=self.lamb)
