import pathlib

import pytest
import torch

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


def _make_batch(n: int = 8) -> list[TrainingExample]:
    examples = []
    for _ in range(n):
        board = BoardTensor(torch.randn(12, 8, 8))
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0
        policy = PolicyTarget(policy_data)
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        examples.append(
            TrainingExample(board=board, policy=policy, value=value)
        )
    return examples


class TestSupervisedTrainer:
    @pytest.fixture
    def trainer(self, device: torch.device) -> SupervisedTrainer:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        loss_fn = ChessLossComputer()
        return SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
        )

    def test_train_step_returns_loss(
        self, trainer: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        assert "policy" in breakdown

    def test_loss_decreases_over_steps(
        self, trainer: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        losses = []
        for _ in range(20):
            loss, _ = trainer.train_step(batch)
            losses.append(loss)
        assert losses[-1] < losses[0]

    def test_save_and_load_checkpoint(
        self, trainer: SupervisedTrainer, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(path)
        assert path.exists()
        trainer.load_checkpoint(path)

    def test_encoder_lr_lower_than_head_lr(
        self, trainer: SupervisedTrainer
    ) -> None:
        groups = trainer.optimizer.param_groups
        encoder_lr = groups[0]["lr"]
        head_lr = groups[2]["lr"]
        assert encoder_lr < head_lr

    def test_gradients_are_clipped(
        self, trainer: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        trainer.train_step(batch)
        all_params = [
            p
            for group in trainer.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        total_norm = torch.nn.utils.clip_grad_norm_(
            all_params, float("inf")
        )
        assert total_norm.item() < 100.0

    def test_scheduler_reduces_lr(self, trainer: SupervisedTrainer) -> None:
        """After stepping the scheduler, learning rates should decrease."""
        initial_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
        batch = _make_batch(8)
        # Simulate several epochs
        for _ in range(5):
            trainer.train_step(batch)
            trainer.scheduler_step()
        current_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
        # At least one group should have a lower LR
        assert any(c < i for c, i in zip(current_lrs, initial_lrs))
