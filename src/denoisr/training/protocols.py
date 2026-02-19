from typing import Protocol

import torch

from denoisr.types import GameRecord


class LossComputer(Protocol):
    def compute(
        self,
        pred_policy: torch.Tensor,
        pred_value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        **auxiliary_losses: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (total_loss, {loss_name: scalar_value}).

        auxiliary_losses may include: consistency_loss, diffusion_loss,
        reward_loss, ply_loss (added in Phase 2/3).
        """
        ...


class ReplayBuffer(Protocol):
    def add(self, record: GameRecord, priority: float = 1.0) -> None: ...
    def sample(self, batch_size: int) -> list[GameRecord]: ...
    def update_priorities(self, indices: list[int], priorities: list[float]) -> None: ...
    def __len__(self) -> int: ...


class MCTSPolicy(Protocol):
    def search(
        self,
        root_state: torch.Tensor,
        num_simulations: int,
    ) -> torch.Tensor:
        """Returns visit count distribution [64, 64]."""
        ...
