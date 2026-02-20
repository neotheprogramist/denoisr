import torch
from torch import Tensor, nn


class ChessValueHead(nn.Module):
    """WDLP value head: Win/Draw/Loss probabilities + ply prediction.

    Pools 64 square tokens into a single board representation, then
    produces WDL logits (softmax applied only via `infer()` for inference)
    and a non-negative ply estimate (via softplus).
    """

    def __init__(self, d_s: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_s)
        self.wdl_linear = nn.Linear(d_s, 3)
        self.ply_linear = nn.Linear(d_s, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self.norm(x.mean(dim=1))
        with torch.amp.autocast("cuda", enabled=False):  # type: ignore[attr-defined]
            wdl_logits = self.wdl_linear(pooled.float())
        ply = torch.nn.functional.softplus(self.ply_linear(pooled))
        return wdl_logits, ply

    def infer(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return WDL probabilities + ply for inference callers."""
        wdl_logits, ply = self.forward(x)
        return torch.softmax(wdl_logits, dim=-1), ply
