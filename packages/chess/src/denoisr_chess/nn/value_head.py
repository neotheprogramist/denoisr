import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ChessValueHead(nn.Module):
    """WDLP value head: Win/Draw/Loss probabilities + ply prediction.

    Pools 64 square tokens into a single board representation via learned
    attention pooling (a learnable query attends over the 64 positions),
    then produces WDL logits and a non-negative ply estimate.
    """

    def __init__(self, d_s: int) -> None:
        super().__init__()
        self.d_s = d_s
        self.norm = nn.LayerNorm(d_s)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_s))
        self.wdl_linear = nn.Linear(d_s, 3)
        self.ply_linear = nn.Linear(d_s, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        normed = self.norm(x)
        # Attention pooling: query [B, 1, D] @ keys [B, D, 64] -> [B, 1, 64]
        query = self.pool_query.expand(x.shape[0], -1, -1)
        attn_weights = F.softmax(
            torch.bmm(query, normed.transpose(1, 2)) / math.sqrt(self.d_s),
            dim=-1,
        )
        pooled = torch.bmm(attn_weights, normed).squeeze(1)  # [B, D]
        with torch.amp.autocast("cuda", enabled=False):  # type: ignore[attr-defined]
            wdl_logits = self.wdl_linear(pooled.float())
        ply = torch.nn.functional.softplus(self.ply_linear(pooled))
        return wdl_logits, ply

    def infer(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return WDL probabilities + ply for inference callers."""
        wdl_logits, ply = self.forward(x)
        return torch.softmax(wdl_logits, dim=-1), ply
