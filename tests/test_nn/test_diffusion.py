import pytest
import torch

from denoisr.nn.diffusion import (
    ChessDiffusionModule,
    CosineNoiseSchedule,
    DPMSolverPP,
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

    def test_compute_v_target_shape(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        x_0 = torch.randn(2, 64, SMALL_D_S)
        noise = torch.randn_like(x_0)
        t = torch.tensor([0, SMALL_NUM_TIMESTEPS - 1])
        v = schedule.compute_v_target(x_0, noise, t)
        assert v.shape == x_0.shape

    def test_v_prediction_roundtrip_recovers_x0(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        """v-prediction must allow exact recovery of x_0."""
        x_0 = torch.randn(2, 64, SMALL_D_S)
        noise = torch.randn_like(x_0)
        t = torch.tensor([5, 10])
        x_t = schedule.q_sample(x_0, t, noise)
        v = schedule.compute_v_target(x_0, noise, t)
        x_0_recovered = schedule.predict_x0_from_v(x_t, v, t)
        assert torch.allclose(x_0_recovered, x_0, atol=1e-5)

    def test_v_prediction_roundtrip_recovers_eps(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        """v-prediction must allow exact recovery of noise."""
        x_0 = torch.randn(2, 64, SMALL_D_S)
        noise = torch.randn_like(x_0)
        t = torch.tensor([5, 10])
        x_t = schedule.q_sample(x_0, t, noise)
        v = schedule.compute_v_target(x_0, noise, t)
        eps_recovered = schedule.predict_eps_from_v(x_t, v, t)
        assert torch.allclose(eps_recovered, noise, atol=1e-5)

    def test_v_target_at_t0_approximates_noise(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        """At t=0, alpha_bar is near 1, so v approximates epsilon."""
        x_0 = torch.randn(1, 64, SMALL_D_S)
        noise = torch.randn_like(x_0)
        t = torch.tensor([0])
        v = schedule.compute_v_target(x_0, noise, t)
        assert torch.allclose(v, noise, atol=0.2)


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


class TestDPMSolverPP:
    @pytest.fixture
    def schedule(self) -> CosineNoiseSchedule:
        return CosineNoiseSchedule(num_timesteps=SMALL_NUM_TIMESTEPS)

    def test_sample_returns_correct_shape(
        self, schedule: CosineNoiseSchedule, device: torch.device
    ) -> None:
        schedule = schedule.to(device)
        solver = DPMSolverPP(schedule, num_steps=5)

        def dummy_model(x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        cond = torch.randn(2, 64, SMALL_D_S, device=device)
        result = solver.sample(
            dummy_model,
            shape=(2, 64, SMALL_D_S),
            cond=cond,
            device=device,
        )
        assert result.shape == (2, 64, SMALL_D_S)

    def test_sample_is_finite(
        self, schedule: CosineNoiseSchedule, device: torch.device
    ) -> None:
        schedule = schedule.to(device)
        solver = DPMSolverPP(schedule, num_steps=5)
        diffusion = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        diffusion.eval()
        cond = torch.randn(2, 64, SMALL_D_S, device=device)

        with torch.no_grad():
            result = solver.sample(diffusion, (2, 64, SMALL_D_S), cond, device)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_sample_deterministic_with_seed(
        self, schedule: CosineNoiseSchedule, device: torch.device
    ) -> None:
        schedule = schedule.to(device)
        solver = DPMSolverPP(schedule, num_steps=5)

        def dummy_model(x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            return x * 0.1

        cond = torch.randn(1, 64, SMALL_D_S, device=device)
        torch.manual_seed(42)
        r1 = solver.sample(dummy_model, (1, 64, SMALL_D_S), cond, device)
        torch.manual_seed(42)
        r2 = solver.sample(dummy_model, (1, 64, SMALL_D_S), cond, device)
        assert torch.allclose(r1, r2)

    def test_fewer_steps_still_works(
        self, schedule: CosineNoiseSchedule, device: torch.device
    ) -> None:
        """DPMSolverPP should work with as few as 2 steps."""
        schedule = schedule.to(device)
        solver = DPMSolverPP(schedule, num_steps=2)

        def dummy_model(x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        cond = torch.randn(1, 64, SMALL_D_S, device=device)
        result = solver.sample(dummy_model, (1, 64, SMALL_D_S), cond, device)
        assert result.shape == (1, 64, SMALL_D_S)
        assert not torch.isnan(result).any()
