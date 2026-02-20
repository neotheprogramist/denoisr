"""Chess training dataset for DataLoader integration."""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from denoisr.training.augmentation import flip_board, flip_policy


class ChessDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Wraps pre-stacked training tensors for use with DataLoader.

    Augmentation (50% random board flip) runs in __getitem__,
    which means it executes in DataLoader worker processes --
    overlapped with GPU training.
    """

    def __init__(
        self,
        boards: Tensor,
        policies: Tensor,
        values: Tensor,
        num_planes: int,
        augment: bool = True,
    ) -> None:
        self.boards = boards
        self.policies = policies
        self.values = values
        self.num_planes = num_planes
        self.augment = augment

    def __len__(self) -> int:
        return self.boards.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        board = self.boards[idx]
        policy = self.policies[idx]
        value = self.values[idx]
        if self.augment and torch.rand(1).item() < 0.5:
            board = flip_board(board, self.num_planes)
            policy = flip_policy(policy)
            value = value.flip(0)  # [w,d,l] -> [l,d,w]
        return board, policy, value
