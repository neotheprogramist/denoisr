"""Threat prediction auxiliary loss head.

Predicts a 64-dim binary vector indicating which squares are under threat
(attacked by the opponent). Forces intermediate representations to encode
threat information, improving downstream defensive move selection.
"""

from torch import Tensor, nn
from torch.nn import functional as F


class ThreatHead(nn.Module):
    """Predicts per-square threat probabilities from backbone features."""

    def __init__(self, d_s: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_s, 1)

    def forward(self, features: Tensor) -> Tensor:
        """features: [B, 64, D] -> logits: [B, 64]."""
        result: Tensor = self.linear(features).squeeze(-1)
        return result


def threat_loss(pred_logits: Tensor, target: Tensor) -> Tensor:
    """Binary cross-entropy between predicted threat logits and target mask.

    pred_logits: [B, 64] raw logits
    target: [B, 64] binary (1 = square under threat)
    """
    result: Tensor = F.binary_cross_entropy_with_logits(pred_logits, target)
    return result
