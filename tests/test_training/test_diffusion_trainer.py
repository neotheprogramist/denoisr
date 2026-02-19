import pytest
import torch

from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.training.diffusion_trainer import DiffusionTrainer

from conftest import (
    SMALL_D_S,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


class TestDiffusionTrainer:
    @pytest.fixture
    def trainer(self, device: torch.device) -> DiffusionTrainer:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        diffusion = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        schedule = CosineNoiseSchedule(num_timesteps=SMALL_NUM_TIMESTEPS)
        return DiffusionTrainer(
            encoder=encoder,
            diffusion=diffusion,
            schedule=schedule,
            lr=1e-3,
            device=device,
        )

    def test_train_step_returns_loss(
        self, trainer: DiffusionTrainer, device: torch.device
    ) -> None:
        trajectory = torch.randn(2, 5, 12, 8, 8, device=device)
        loss = trainer.train_step(trajectory)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_is_finite(
        self, trainer: DiffusionTrainer, device: torch.device
    ) -> None:
        trajectory = torch.randn(2, 3, 12, 8, 8, device=device)
        loss = trainer.train_step(trajectory)
        assert not (loss != loss)  # NaN check

    def test_loss_decreases(
        self, trainer: DiffusionTrainer, device: torch.device
    ) -> None:
        trajectory = torch.randn(2, 4, 12, 8, 8, device=device)
        losses = [trainer.train_step(trajectory) for _ in range(30)]
        assert losses[-1] < losses[0]
