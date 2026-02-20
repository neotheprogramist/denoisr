import pytest
import torch

from denoisr.training.phase2_trainer import TrajectoryBatch


class TestTrajectoryBatch:
    def test_valid_shapes(self) -> None:
        B, T, C = 4, 5, 12
        batch = TrajectoryBatch(
            boards=torch.randn(B, T, C, 8, 8),
            actions_from=torch.randint(0, 64, (B, T - 1)),
            actions_to=torch.randint(0, 64, (B, T - 1)),
            policies=torch.zeros(B, T - 1, 64, 64),
            values=torch.tensor([[1.0, 0.0, 0.0]] * B),
            rewards=torch.zeros(B, T - 1),
        )
        assert batch.boards.shape == (B, T, C, 8, 8)
        assert batch.actions_from.shape == (B, T - 1)
        assert batch.policies.shape == (B, T - 1, 64, 64)
        assert batch.values.shape == (B, 3)
        assert batch.rewards.shape == (B, T - 1)

    def test_frozen(self) -> None:
        batch = TrajectoryBatch(
            boards=torch.randn(2, 3, 12, 8, 8),
            actions_from=torch.zeros(2, 2, dtype=torch.long),
            actions_to=torch.zeros(2, 2, dtype=torch.long),
            policies=torch.zeros(2, 2, 64, 64),
            values=torch.tensor([[1.0, 0.0, 0.0]] * 2),
            rewards=torch.zeros(2, 2),
        )
        with pytest.raises(AttributeError):
            batch.boards = torch.randn(2, 3, 12, 8, 8)  # type: ignore[misc]

    def test_rejects_mismatched_time_dims(self) -> None:
        with pytest.raises(ValueError, match="time"):
            TrajectoryBatch(
                boards=torch.randn(2, 5, 12, 8, 8),
                actions_from=torch.zeros(2, 3, dtype=torch.long),  # should be 4
                actions_to=torch.zeros(2, 4, dtype=torch.long),
                policies=torch.zeros(2, 4, 64, 64),
                values=torch.tensor([[1.0, 0.0, 0.0]] * 2),
                rewards=torch.zeros(2, 4),
            )

    def test_rejects_mismatched_batch_dims(self) -> None:
        with pytest.raises(ValueError, match="batch"):
            TrajectoryBatch(
                boards=torch.randn(4, 5, 12, 8, 8),
                actions_from=torch.zeros(2, 4, dtype=torch.long),  # B=2 vs 4
                actions_to=torch.zeros(4, 4, dtype=torch.long),
                policies=torch.zeros(4, 4, 64, 64),
                values=torch.tensor([[1.0, 0.0, 0.0]] * 4),
                rewards=torch.zeros(4, 4),
            )
