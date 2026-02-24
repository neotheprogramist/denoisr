from dataclasses import dataclass

import torch

_VALID_PROMOTIONS = frozenset({2, 3, 4, 5})


@dataclass(frozen=True)
class Action:
    from_square: int
    to_square: int
    promotion: int | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.from_square < 64):
            raise ValueError(f"from_square must be 0-63, got {self.from_square}")
        if not (0 <= self.to_square < 64):
            raise ValueError(f"to_square must be 0-63, got {self.to_square}")
        if self.promotion is not None and self.promotion not in _VALID_PROMOTIONS:
            raise ValueError(f"promotion must be 2-5 or None, got {self.promotion}")


@dataclass(frozen=True)
class LegalMask:
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.shape != (64, 64):
            raise ValueError(f"Expected shape [64, 64], got {list(self.data.shape)}")
        if self.data.dtype != torch.bool:
            raise ValueError(f"Expected bool, got {self.data.dtype}")
