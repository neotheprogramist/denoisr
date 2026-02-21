# tests/test_scripts/test_measure_accuracy.py
import torch
import pytest

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

from conftest import SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM


def _make_trainer(device: torch.device) -> SupervisedTrainer:
    encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
    backbone = ChessPolicyBackbone(
        d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
        num_layers=SMALL_NUM_LAYERS, ffn_dim=SMALL_FFN_DIM,
    ).to(device)
    policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
    value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
    loss_fn = ChessLossComputer()
    return SupervisedTrainer(
        encoder=encoder, backbone=backbone,
        policy_head=policy_head, value_head=value_head,
        loss_fn=loss_fn, lr=1e-3, device=device,
    )


class TestMeasureAccuracy:
    def test_accuracy_masks_illegal_moves(self, device: torch.device) -> None:
        """With only 1 legal move, top-5 must always be 100% (the sole legal
        move is guaranteed to appear in top-5 of masked logits)."""
        from denoisr.scripts.train_phase1 import measure_accuracy

        trainer = _make_trainer(device)
        # Create example where target has ONE legal move at (12, 28)
        board = BoardTensor(torch.randn(12, 8, 8))
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0  # only legal move
        policy = PolicyTarget(policy_data)
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        examples = [TrainingExample(board=board, policy=policy, value=value)]

        top1, top5 = measure_accuracy(trainer, examples, device)
        # With 1 legal move, masking guarantees it's the top-1 AND top-5 pick
        assert top1 == 1.0
        assert top5 == 1.0

    def test_perfect_model_gets_100_percent(self, device: torch.device) -> None:
        """A model trained to predict the exact target should reach high accuracy."""
        from denoisr.scripts.train_phase1 import measure_accuracy

        trainer = _make_trainer(device)

        board_data = torch.randn(12, 8, 8)
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0
        examples = [
            TrainingExample(
                board=BoardTensor(board_data),
                policy=PolicyTarget(policy_data),
                value=ValueTarget(win=1.0, draw=0.0, loss=0.0),
            )
        ]

        # Overtrain on a single example
        for _ in range(200):
            trainer.train_step(examples)

        top1, top5 = measure_accuracy(trainer, examples, device)
        # After overtraining, should get this right with masking
        assert top1 > 0.5
