import chess
import torch
from torch import Tensor, nn


class ShawRelativePositionBias(nn.Module):
    """Shaw relative position encoding for chess board topology.

    Learns a bias table indexed by (delta_rank, delta_file) between
    every pair of squares. Superior to RoPE for chess because it
    directly captures the spatial relationships of the 8x8 board
    (adjacent squares, diagonals, knight jumps) rather than treating
    positions as a 1D sequence.

    The bias table has shape [num_heads, 15, 15] for rank/file
    differences in range [-7, +7]. The output is [num_heads, 64, 64].
    """

    rank_idx: Tensor
    file_idx: Tensor

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Rank and file differences range from -7 to +7 = 15 values
        self.bias_table = nn.Parameter(torch.zeros(num_heads, 15, 15))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # Precompute (rank_diff, file_diff) for all 64x64 square pairs
        coords = torch.tensor(
            [(chess.square_rank(sq), chess.square_file(sq)) for sq in range(64)]
        )
        rank_diff = coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0) + 7
        file_diff = coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0) + 7
        self.register_buffer("rank_idx", rank_diff.long())
        self.register_buffer("file_idx", file_diff.long())

    def forward(self) -> Tensor:
        return self.bias_table[:, self.rank_idx, self.file_idx]
