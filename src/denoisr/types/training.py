from dataclasses import dataclass

import torch

from denoisr.types.action import Action
from denoisr.types.board import BoardTensor


@dataclass(frozen=True)
class PolicyTarget:
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.shape != (64, 64):
            raise ValueError(
                f"Expected shape [64, 64], got {list(self.data.shape)}"
            )


@dataclass(frozen=True)
class ValueTarget:
    win: float
    draw: float
    loss: float

    def __post_init__(self) -> None:
        total = self.win + self.draw + self.loss
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"WDL must sum to 1.0, got {total}")


@dataclass(frozen=True)
class TrainingExample:
    board: BoardTensor
    policy: PolicyTarget
    value: ValueTarget
    game_id: int | None = None
    eco_code: str | None = None
    piece_count: int | None = None


@dataclass(frozen=True)
class GameRecord:
    actions: tuple[Action, ...]
    result: float
    eco_code: str | None = None
    white_elo: int | None = None
    black_elo: int | None = None
