import torch
from torch import Tensor, nn


class ChessValueHead(nn.Module):
    """WDLP value head: Win/Draw/Loss probabilities + ply prediction.

    Pools 64 square tokens into a single board representation, then
    produces WDL probabilities (via softmax) and a non-negative ply
    estimate (via softplus).
    """

    def __init__(self, d_s: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_s)
        self.wdl_linear = nn.Linear(d_s, 3)
        self.ply_linear = nn.Linear(d_s, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self.norm(x.mean(dim=1))
        wdl = torch.softmax(self.wdl_linear(pooled), dim=-1)
        ply = torch.nn.functional.softplus(self.ply_linear(pooled))
        return wdl, ply
