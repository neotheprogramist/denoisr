from torch import Tensor, nn


class SmolgenBias(nn.Module):
    """Generates dynamic per-head attention biases from the full board state.

    Compresses 64 token embeddings into a small vector, then projects
    to H x 64 x 64 attention bias matrices. This lets the attention
    pattern adapt to the specific position (e.g., suppress long-range
    connections in closed positions).
    """

    def __init__(
        self, d_s: int, num_heads: int, compress_dim: int = 256
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.compress = nn.Sequential(
            nn.Linear(64 * d_s, compress_dim),
            nn.Mish(),
        )
        self.project = nn.Linear(compress_dim, num_heads * 64 * 64)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        flat = x.reshape(B, -1)
        compressed: Tensor = self.compress(flat)
        biases: Tensor = self.project(compressed)
        return biases.reshape(B, self.num_heads, 64, 64)
