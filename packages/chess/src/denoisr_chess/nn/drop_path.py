"""DropPath (stochastic depth) for transformer residual connections.

Randomly drops entire residual branches during training with probability
drop_prob. Surviving branches are scaled by 1/(1-drop_prob) to maintain
expected magnitude. In non-training mode, acts as identity.

Reference: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
"""

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        if self.drop_prob == 1.0:
            return torch.zeros_like(x)
        keep_prob = 1.0 - self.drop_prob
        # Shape [B, 1, 1, ...] -- drop entire sample's residual branch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep_prob, device=x.device, dtype=x.dtype)
        )
        return x * mask / keep_prob
