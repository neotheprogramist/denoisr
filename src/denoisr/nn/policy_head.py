import torch
from torch import Tensor, nn


class ChessPolicyHead(nn.Module):
    """Source-destination attention policy head.

    Computes bilinear attention between source-square queries and
    destination-square keys, producing a [B, 64, 64] logit matrix
    where entry (i, j) is the unnormalized log-probability of moving
    from square i to square j.
    """

    def __init__(self, d_s: int, d_head: int = 128) -> None:
        super().__init__()
        self.query = nn.Linear(d_s, d_head)
        self.key = nn.Linear(d_s, d_head)
        self.scale = d_head**-0.5

    def forward(self, x: Tensor) -> Tensor:
        q: Tensor = self.query(x)
        k: Tensor = self.key(x)
        logits: Tensor = torch.bmm(q, k.transpose(1, 2)) * self.scale
        return logits
