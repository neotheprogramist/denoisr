"""Integration tests verifying all CUDA optimizations work together."""

import math

import torch

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.nn.diffusion import CosineNoiseSchedule
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.training.dataset import ChessDataset
from denoisr.scripts.config import maybe_compile

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


class TestFullPipeline:
    """Verify all optimizations work together end-to-end."""

    def test_compiled_amp_checkpointed_training(
        self, device: torch.device
    ) -> None:
        """Full pipeline: compile + AMP + gradient checkpointing + DataLoader."""
        encoder = maybe_compile(
            ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device), device
        )
        backbone = maybe_compile(
            ChessPolicyBackbone(
                d_s=SMALL_D_S,
                num_heads=SMALL_NUM_HEADS,
                num_layers=SMALL_NUM_LAYERS,
                ffn_dim=SMALL_FFN_DIM,
                gradient_checkpointing=True,
            ).to(device),
            device,
        )
        policy_head = maybe_compile(
            ChessPolicyHead(d_s=SMALL_D_S).to(device), device
        )
        value_head = maybe_compile(
            ChessValueHead(d_s=SMALL_D_S).to(device), device
        )

        loss_fn = ChessLossComputer()
        trainer = SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
        )

        # Create dataset + DataLoader
        boards = torch.randn(32, 12, 8, 8)
        policies = torch.zeros(32, 64, 64)
        policies[:, 12, 28] = 1.0
        values = torch.tensor([[1.0, 0.0, 0.0]] * 32)
        dataset = ChessDataset(boards, policies, values, num_planes=12, augment=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        # Train a few steps
        losses: list[float] = []
        for batch_boards, batch_policies, batch_values in loader:
            loss, _ = trainer.train_step_tensors(
                batch_boards, batch_policies, batch_values
            )
            assert math.isfinite(loss), f"Non-finite loss: {loss}"
            losses.append(loss)

        assert len(losses) == 4  # 32 examples / batch_size 8
        assert all(loss_val > 0 for loss_val in losses)

    def test_schedule_buffer_on_device(self, device: torch.device) -> None:
        """CosineNoiseSchedule.alpha_bar should live on the correct device."""
        schedule = CosineNoiseSchedule(num_timesteps=SMALL_NUM_TIMESTEPS)
        schedule.to(device)
        assert schedule.alpha_bar.device.type == device.type

        # q_sample should work without explicit .to()
        x_0 = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.tensor([0, 1], device=device)
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise)
        assert x_t.device.type == device.type
