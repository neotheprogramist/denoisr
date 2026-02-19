from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentState:
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {self.data.ndim}D")
        if self.data.shape[0] != 64:
            raise ValueError(
                f"Expected 64 tokens, got {self.data.shape[0]}"
            )


@dataclass(frozen=True)
class LatentTrajectory:
    states: tuple[LatentState, ...]

    def __post_init__(self) -> None:
        if len(self.states) == 0:
            raise ValueError("Trajectory must have at least one state")
        d_s = self.states[0].data.shape[1]
        for i, s in enumerate(self.states):
            if s.data.shape[1] != d_s:
                raise ValueError(
                    f"State {i} has d_s={s.data.shape[1]}, expected {d_s}"
                )
