import pytest
import torch

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)
from denoisr.nn.consistency import ChessConsistencyProjector
from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.nn.world_model import ChessWorldModel
from denoisr.training.loss import ChessLossComputer
from denoisr.training.phase2_trainer import Phase2Trainer, TrajectoryBatch


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


def _make_trajectory_batch(
    B: int = 2,
    T: int = 5,
    C: int = 12,
    device: torch.device = torch.device("cpu"),
) -> TrajectoryBatch:
    policies = torch.zeros(B, T - 1, 64, 64, device=device)
    for b in range(B):
        for t in range(T - 1):
            policies[b, t, 12, 28] = 1.0
    return TrajectoryBatch(
        boards=torch.randn(B, T, C, 8, 8, device=device),
        actions_from=torch.randint(0, 64, (B, T - 1), device=device),
        actions_to=torch.randint(0, 64, (B, T - 1), device=device),
        policies=policies,
        values=torch.tensor([[1.0, 0.0, 0.0]] * B, device=device),
        rewards=torch.ones(B, T - 1, device=device),
    )


class TestPhase2Trainer:
    @pytest.fixture
    def trainer(self, device: torch.device) -> Phase2Trainer:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        world_model = ChessWorldModel(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        diffusion = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        consistency = ChessConsistencyProjector(d_s=SMALL_D_S).to(device)
        schedule = CosineNoiseSchedule(
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        loss_fn = ChessLossComputer()
        return Phase2Trainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            world_model=world_model,
            diffusion=diffusion,
            consistency=consistency,
            schedule=schedule,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
        )

    def test_train_step_returns_loss_and_breakdown(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_breakdown_contains_all_6_terms(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        _, breakdown = trainer.train_step(batch)
        for key in ("policy", "value", "diffusion", "state", "reward", "consistency"):
            assert key in breakdown, f"Missing '{key}' in breakdown"
        assert "grad_norm" in breakdown

    def test_encoder_is_frozen(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        trainer.train_step(batch)
        for p in trainer.encoder.parameters():
            assert not p.requires_grad

    def test_backbone_lr_lower_than_head_lr(
        self,
        trainer: Phase2Trainer,
    ) -> None:
        groups = trainer.optimizer.param_groups
        backbone_lr = groups[0]["lr"]
        head_lr = groups[1]["lr"]
        assert backbone_lr < head_lr

    def test_loss_is_finite(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        loss, _ = trainer.train_step(batch)
        assert loss == loss  # NaN check
        assert loss < float("inf")

    def test_loss_decreases_over_steps(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        losses = [trainer.train_step(batch)[0] for _ in range(30)]
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg

    def test_curriculum_advances(self, trainer: Phase2Trainer) -> None:
        initial = trainer.current_max_steps
        for _ in range(100):
            trainer.advance_curriculum()
        assert trainer.current_max_steps > initial
