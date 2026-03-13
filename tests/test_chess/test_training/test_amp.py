import pytest
import torch

from denoisr_chess.nn.encoder import ChessEncoder
from denoisr_chess.nn.policy_backbone import ChessPolicyBackbone
from denoisr_chess.nn.policy_head import ChessPolicyHead
from denoisr_chess.nn.value_head import ChessValueHead
from denoisr_chess.training.loss import ChessLossComputer
from denoisr_chess.training.supervised_trainer import SupervisedTrainer
from denoisr_chess.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


def _make_batch(n: int = 4) -> list[TrainingExample]:
    examples = []
    for _ in range(n):
        board = BoardTensor(torch.randn(12, 8, 8))
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0
        policy = PolicyTarget(policy_data)
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        examples.append(TrainingExample(board=board, policy=policy, value=value))
    return examples


class TestSupervisedTrainerAMP:
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

    def test_trainer_has_scaler(self, trainer: SupervisedTrainer) -> None:
        assert hasattr(trainer, "scaler")
        assert isinstance(trainer.scaler, torch.amp.GradScaler)

    def test_train_step_works_with_amp(self, trainer: SupervisedTrainer) -> None:
        batch = _make_batch(4)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        assert "policy" in breakdown

    def test_breakdown_includes_grad_norm(self, trainer: SupervisedTrainer) -> None:
        """Training breakdown should include gradient norm."""
        batch = _make_batch(4)
        _, breakdown = trainer.train_step(batch)
        assert "grad_norm" in breakdown
        assert breakdown["grad_norm"] >= 0

    def test_loss_decreases_with_amp(self, trainer: SupervisedTrainer) -> None:
        batch = _make_batch(4)
        losses = [trainer.train_step(batch)[0] for _ in range(20)]
        assert losses[-1] < losses[0]
