from dataclasses import dataclass

import torch

BOARD_SIZE = 8
NUM_SQUARES = 64
NUM_PIECE_TYPES = 6
NUM_COLORS = 2
NUM_PLANES = NUM_PIECE_TYPES * NUM_COLORS


@dataclass(frozen=True)
class BoardTensor:
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {self.data.ndim}D")
        if self.data.shape[1:] != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"Expected shape [C, 8, 8], got {list(self.data.shape)}"
            )
        if self.data.dtype != torch.float32:
            raise ValueError(f"Expected float32, got {self.data.dtype}")
