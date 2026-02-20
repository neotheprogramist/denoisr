import pytest
import torch

from denoisr.nn.diffusion import (
    ChessDiffusionModule,
    CosineNoiseSchedule,
)

from conftest import (
    SMALL_D_S,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


class TestCosineNoiseSchedule:
    @pytest.fixture
    def schedule(self) -> CosineNoiseSchedule:
        return CosineNoiseSchedule(num_timesteps=SMALL_NUM_TIMESTEPS)

    def test_alpha_bar_monotonic_decreasing(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        ab = schedule.alpha_bar
        assert ab.shape == (SMALL_NUM_TIMESTEPS,)
        for i in range(len(ab) - 1):
            assert ab[i] > ab[i + 1]

    def test_alpha_bar_bounds(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        assert schedule.alpha_bar[0] > 0.9
        assert schedule.alpha_bar[-1] < 0.1

    def test_q_sample_shape(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        x_0 = torch.randn(2, 64, SMALL_D_S)
        t = torch.tensor([0, SMALL_NUM_TIMESTEPS - 1])
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise)
        assert x_t.shape == x_0.shape

    def test_q_sample_t0_close_to_clean(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        x_0 = torch.randn(1, 64, SMALL_D_S)
        t = torch.tensor([0])
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise)
        assert torch.allclose(x_t, x_0, atol=0.2)

    def test_schedule_is_nn_module(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        import torch.nn as nn

        assert isinstance(schedule, nn.Module)

    def test_alpha_bar_moves_with_to(
        self, schedule: CosineNoiseSchedule, device: torch.device
    ) -> None:
        schedule.to(device)
        assert schedule.alpha_bar.device.type == device.type


class TestChessDiffusionModule:
    @pytest.fixture
    def diffusion(self, device: torch.device) -> ChessDiffusionModule:
        return ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)

    def test_output_shape(
        self, diffusion: ChessDiffusionModule, device: torch.device
    ) -> None:
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.randint(0, SMALL_NUM_TIMESTEPS, (2,), device=device)
        cond = torch.randn(2, 64, SMALL_D_S, device=device)
        out = diffusion(x, t, cond)
        assert out.shape == (2, 64, SMALL_D_S)

    def test_different_timesteps_different_outputs(
        self, diffusion: ChessDiffusionModule, device: torch.device
    ) -> None:
        # Reinit final_proj so output is non-zero (it starts zero-initialized
        # for training stability, verified by test_adaln_zero_init)
        torch.nn.init.xavier_uniform_(diffusion.final_proj.weight)
        diffusion.eval()
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        cond = torch.randn(1, 64, SMALL_D_S, device=device)
        t0 = torch.tensor([0], device=device)
        t1 = torch.tensor([SMALL_NUM_TIMESTEPS - 1], device=device)
        out0 = diffusion(x, t0, cond)
        out1 = diffusion(x, t1, cond)
        assert not torch.allclose(out0, out1)

    def test_gradient_flows(
        self, diffusion: ChessDiffusionModule, device: torch.device
    ) -> None:
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.randint(0, SMALL_NUM_TIMESTEPS, (2,), device=device)
        cond = torch.randn(2, 64, SMALL_D_S, device=device)
        out = diffusion(x, t, cond)
        out.sum().backward()
        for name, p in diffusion.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_adaln_zero_init(
        self, diffusion: ChessDiffusionModule
    ) -> None:
        w = diffusion.final_proj.weight
        b = diffusion.final_proj.bias
        assert torch.allclose(w, torch.zeros_like(w))
        assert torch.allclose(b, torch.zeros_like(b))

    def test_no_nan(
        self, diffusion: ChessDiffusionModule, device: torch.device
    ) -> None:
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.randint(0, SMALL_NUM_TIMESTEPS, (2,), device=device)
        cond = torch.randn(2, 64, SMALL_D_S, device=device)
        out = diffusion(x, t, cond)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_checkpointing_produces_gradients(
        self, device: torch.device
    ) -> None:
        """Diffusion module with gradient checkpointing should produce valid gradients."""
        diff = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
            gradient_checkpointing=True,
        ).to(device)
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.randint(0, SMALL_NUM_TIMESTEPS, (2,), device=device)
        cond = torch.randn(2, 64, SMALL_D_S, device=device)
        out = diff(x, t, cond)
        out.sum().backward()
        for name, p in diff.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_gradient_checkpointing_matches_output(
        self, device: torch.device
    ) -> None:
        """Checkpointed and non-checkpointed diffusion produce identical output."""
        torch.manual_seed(42)
        diff_normal = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
            gradient_checkpointing=False,
        ).to(device)
        torch.manual_seed(42)
        diff_ckpt = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
            gradient_checkpointing=True,
        ).to(device)
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        t = torch.tensor([0], device=device)
        cond = torch.randn(1, 64, SMALL_D_S, device=device)
        assert torch.allclose(
            diff_normal(x, t, cond), diff_ckpt(x, t, cond), atol=1e-5
        )
