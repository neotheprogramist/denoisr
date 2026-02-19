import torch
from torch import Tensor, nn


class ChessEncoder(nn.Module):
    """Encodes board tensor [B, C, 8, 8] into latent tokens [B, 64, d_s].

    Uses per-square linear projection plus a global board embedding
    (following BT3/BT4's approach for encoding whole-board context from layer 0).
    """

    def __init__(self, num_planes: int, d_s: int) -> None:
        super().__init__()
        self.square_embed = nn.Linear(num_planes, d_s)
        self.global_embed = nn.Sequential(
            nn.Linear(num_planes * 64, d_s),
            nn.Mish(),
            nn.Linear(d_s, d_s),
        )
        self.norm = nn.LayerNorm(d_s)

    def forward(self, x: Tensor) -> Tensor:
        B, C, _H, _W = x.shape
        # Per-square features: [B, 64, C]
        squares = x.reshape(B, C, 64).permute(0, 2, 1)
        local = self.square_embed(squares)

        # Global context: flatten entire board, project, broadcast
        flat = x.reshape(B, C * 64)
        glob = self.global_embed(flat).unsqueeze(1).expand(-1, 64, -1)

        return self.norm(local + glob)
