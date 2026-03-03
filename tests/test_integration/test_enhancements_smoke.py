"""Smoke test: enhanced training runs for 2 steps without crashing."""

import torch
from conftest import SMALL_D_S, SMALL_FFN_DIM, SMALL_NUM_HEADS, SMALL_NUM_LAYERS

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer


class TestEnhancementsSmokeTest:
    def test_full_enhanced_training_step(self, device: torch.device) -> None:
        """All enhancements active: dropout, drop_path, accumulation, smoothing."""
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            SMALL_D_S,
            SMALL_NUM_HEADS,
            SMALL_NUM_LAYERS,
            SMALL_FFN_DIM,
            dropout=0.1,
            drop_path_rate=0.1,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        loss_fn = ChessLossComputer(label_smoothing=0.1)

        trainer = SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
            total_epochs=2,
            accumulation_steps=2,
        )

        boards = torch.randn(4, 12, 8, 8, device=device)
        target_policy = torch.zeros(4, 64, 64, device=device)
        target_policy[:, 0, 1] = 0.7
        target_policy[:, 0, 3] = 0.3
        target_value = torch.tensor([[0.5, 0.3, 0.2]] * 4, device=device)

        # Run 2 accumulation steps (should trigger 1 optimizer step)
        loss1, bd1 = trainer.train_step_tensors(boards, target_policy, target_value)
        loss2, bd2 = trainer.train_step_tensors(boards, target_policy, target_value)

        assert loss1 > 0
        assert loss2 > 0
        assert bd2.get("grad_norm", 0) > 0  # second step should have grad_norm
