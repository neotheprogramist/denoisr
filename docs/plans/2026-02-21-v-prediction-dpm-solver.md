# v-Prediction + DPM-Solver++ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace epsilon-prediction with v-prediction training and DDIM inference with DPM-Solver++ for faster convergence and 2-3x faster inference, with zero new dependencies.

**Architecture:** Add `compute_v_target`, `predict_x0_from_v`, `predict_eps_from_v` helpers to `CosineNoiseSchedule`. Add a self-contained `DPMSolverPP` sampler class. Swap 5-line blocks in both trainers and both inference paths. DiT architecture unchanged.

**Tech Stack:** PyTorch (existing), no new dependencies

**Design doc:** `docs/plans/2026-02-21-v-prediction-dpm-solver-design.md`

---

## Task 1: Add v-prediction helpers to CosineNoiseSchedule

Three new methods on the existing schedule class. These are pure math with no side effects -- the foundation that all later tasks depend on.

**Files:**
- Modify: `src/denoisr/nn/diffusion.py:9-37` (CosineNoiseSchedule class)
- Test: `tests/test_nn/test_diffusion.py`

**Step 1: Write the failing tests**

Add to `tests/test_nn/test_diffusion.py` inside `TestCosineNoiseSchedule`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nn/test_diffusion.py::TestCosineNoiseSchedule::test_compute_v_target_shape tests/test_nn/test_diffusion.py::TestCosineNoiseSchedule::test_v_prediction_roundtrip_recovers_x0 tests/test_nn/test_diffusion.py::TestCosineNoiseSchedule::test_v_prediction_roundtrip_recovers_eps tests/test_nn/test_diffusion.py::TestCosineNoiseSchedule::test_v_target_at_t0_approximates_noise -v`

Expected: FAIL (AttributeError: 'CosineNoiseSchedule' object has no attribute 'compute_v_target')

**Step 3: Write minimal implementation**

Add three methods to `CosineNoiseSchedule` in `src/denoisr/nn/diffusion.py`, after the existing `q_sample` method:

```python
def _broadcast_ab(self, t: Tensor, target: Tensor) -> Tensor:
    ab = self.alpha_bar[t]
    while ab.ndim < target.ndim:
        ab = ab.unsqueeze(-1)
    return ab

def compute_v_target(self, x_0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
    """Compute v-prediction target: v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x_0."""
    ab = self._broadcast_ab(t, x_0)
    return ab.sqrt() * noise - (1 - ab).sqrt() * x_0

def predict_x0_from_v(self, x_t: Tensor, v: Tensor, t: Tensor) -> Tensor:
    """Recover x_0 from v-prediction: x_0 = sqrt(alpha_bar)*x_t - sqrt(1-alpha_bar)*v."""
    ab = self._broadcast_ab(t, x_t)
    return ab.sqrt() * x_t - (1 - ab).sqrt() * v

def predict_eps_from_v(self, x_t: Tensor, v: Tensor, t: Tensor) -> Tensor:
    """Recover eps from v-prediction: eps = sqrt(1-alpha_bar)*x_t + sqrt(alpha_bar)*v."""
    ab = self._broadcast_ab(t, x_t)
    return (1 - ab).sqrt() * x_t + ab.sqrt() * v
```

Also refactor `q_sample` to use `_broadcast_ab`:

```python
def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
    """Forward diffusion: add noise at timestep t."""
    ab = self._broadcast_ab(t, x_0)
    return ab.sqrt() * x_0 + (1 - ab).sqrt() * noise
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_nn/test_diffusion.py::TestCosineNoiseSchedule -v`

Expected: All tests PASS (existing + 4 new)

**Step 5: Commit**

```bash
git add src/denoisr/nn/diffusion.py tests/test_nn/test_diffusion.py
git commit -m "feat: add v-prediction helpers to CosineNoiseSchedule"
```

---

## Task 2: Add DPMSolverPP sampler

A self-contained 2nd-order ODE sampler that replaces DDIM loops everywhere. Takes a schedule, step count, and model callable; returns the denoised sample.

**Files:**
- Modify: `src/denoisr/nn/diffusion.py` (add DPMSolverPP class after CosineNoiseSchedule)
- Test: `tests/test_nn/test_diffusion.py`

**Step 1: Write the failing tests**

Add new test class to `tests/test_nn/test_diffusion.py`:

```python
from denoisr.nn.diffusion import DPMSolverPP


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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nn/test_diffusion.py::TestDPMSolverPP -v`

Expected: FAIL (ImportError: cannot import name 'DPMSolverPP')

**Step 3: Write minimal implementation**

Add to `src/denoisr/nn/diffusion.py` after the `CosineNoiseSchedule` class, before `DiTBlock`:

```python
from collections.abc import Callable


class DPMSolverPP:
    """DPM-Solver++ 2nd-order multistep sampler (Lu et al. 2022).

    Operates in log-SNR space for numerical stability. Uses v-prediction
    internally: the model outputs v, which is converted to epsilon for the
    ODE update. Falls back to 1st-order for the first step (no history).
    """

    def __init__(
        self,
        schedule: CosineNoiseSchedule,
        num_steps: int = 5,
    ) -> None:
        self.schedule = schedule
        self.num_steps = num_steps

    def _get_timesteps(self) -> list[int]:
        """Evenly-spaced timesteps from T-1 down to 0."""
        T = self.schedule.num_timesteps
        if self.num_steps >= T:
            return list(range(T - 1, -1, -1))
        step_size = T / self.num_steps
        return [int(T - 1 - i * step_size) for i in range(self.num_steps)] + [0]

    def _log_snr(self, t: int) -> float:
        """lambda_t = log(sqrt(alpha_bar_t) / sqrt(1 - alpha_bar_t))."""
        ab = self.schedule.alpha_bar[t].item()
        ab = max(ab, 1e-8)
        return 0.5 * math.log(ab / max(1 - ab, 1e-8))

    def sample(
        self,
        model_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        cond: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Run DPM-Solver++ sampling loop.

        model_fn: (x_t, t, cond) -> v_prediction
        Returns denoised sample x_0.
        """
        timesteps = self._get_timesteps()
        x = torch.randn(shape, device=device)
        prev_eps: Tensor | None = None
        prev_h: float | None = None

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_tensor = torch.full((shape[0],), t_cur, device=device)
            v_pred = model_fn(x, t_tensor, cond)

            eps = self.schedule.predict_eps_from_v(x, v_pred, t_tensor)

            ab_cur = self.schedule.alpha_bar[t_cur]
            ab_next = self.schedule.alpha_bar[t_next]
            sigma_cur = (1 - ab_cur).sqrt()
            sigma_next = (1 - ab_next).sqrt()
            alpha_next = ab_next.sqrt()

            lambda_cur = self._log_snr(t_cur)
            lambda_next = self._log_snr(t_next)
            h = lambda_next - lambda_cur

            # 2nd-order correction when we have a previous model output
            if prev_eps is not None and prev_h is not None:
                r = prev_h / h
                D = (1.0 + 1.0 / (2.0 * r)) * eps - (1.0 / (2.0 * r)) * prev_eps
            else:
                D = eps

            # DPM-Solver++ update
            x = (sigma_next / sigma_cur) * x - alpha_next * (
                torch.exp(torch.tensor(-h, device=device)) - 1.0
            ) * D

            prev_eps = eps
            prev_h = h

        return x
```

Also ensure `import math` is at the top of the file (it already is) and add `from collections.abc import Callable` to the imports.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_nn/test_diffusion.py::TestDPMSolverPP -v`

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/denoisr/nn/diffusion.py tests/test_nn/test_diffusion.py
git commit -m "feat: add DPMSolverPP 2nd-order multistep sampler"
```

---

## Task 3: Switch DiffusionTrainer to v-prediction

The standalone diffusion trainer used in early development. Change the loss from MSE(eps_hat, eps) to MSE(v_hat, v).

**Files:**
- Modify: `src/denoisr/training/diffusion_trainer.py:70-78`
- Test: `tests/test_training/test_diffusion_trainer.py`

**Step 1: Write the failing test**

Add to `tests/test_training/test_diffusion_trainer.py` inside `TestDiffusionTrainer`:

```python
def test_uses_v_prediction_loss(
    self, trainer: DiffusionTrainer, device: torch.device
) -> None:
    """Verify the trainer computes v-prediction loss (positive and finite)."""
    trajectory = torch.randn(2, 5, 12, 8, 8, device=device)
    loss, breakdown = trainer.train_step(trajectory)
    assert isinstance(loss, float)
    assert loss > 0
    assert loss == loss  # NaN check
```

**Step 2: Run tests to verify they pass (baseline)**

Run: `uv run pytest tests/test_training/test_diffusion_trainer.py -v`

Expected: All existing tests PASS

**Step 3: Switch to v-prediction**

In `src/denoisr/training/diffusion_trainer.py`, replace the noise prediction and loss lines.

Before:
```python
            predicted_noise = self.diffusion(noisy_target, t, cond)

            loss = nn.functional.mse_loss(predicted_noise, noise)
```

After:
```python
            v_target = self.schedule.compute_v_target(target, noise, t)
            v_pred = self.diffusion(noisy_target, t, cond)

            loss = nn.functional.mse_loss(v_pred, v_target)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_diffusion_trainer.py -v`

Expected: All tests PASS (including test_loss_decreases -- v-prediction loss should also decrease over 50 steps on a fixed batch)

**Step 5: Commit**

```bash
git add src/denoisr/training/diffusion_trainer.py tests/test_training/test_diffusion_trainer.py
git commit -m "feat: switch DiffusionTrainer to v-prediction loss"
```

---

## Task 4: Switch Phase2Trainer to v-prediction

The unified 6-loss trainer. Same 2-line swap for the diffusion loss term.

**Files:**
- Modify: `src/denoisr/training/phase2_trainer.py:171-175`
- Test: `tests/test_training/test_phase2_trainer.py`

**Step 1: Run existing tests (baseline)**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestPhase2Trainer -v`

Expected: All tests PASS

**Step 2: Switch to v-prediction**

In `src/denoisr/training/phase2_trainer.py`, replace the noise prediction and loss lines.

Before:
```python
            predicted_noise = self.diffusion(noisy_target, t, cond)
            diffusion_loss = F.mse_loss(predicted_noise, noise)
```

After:
```python
            v_target = self.schedule.compute_v_target(diff_target, noise, t)
            v_pred = self.diffusion(noisy_target, t, cond)
            diffusion_loss = F.mse_loss(v_pred, v_target)
```

**Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestPhase2Trainer -v`

Expected: All 7 tests PASS (loss_decreases_over_steps should still work -- v-prediction loss also decreases on a fixed batch)

**Step 4: Commit**

```bash
git add src/denoisr/training/phase2_trainer.py
git commit -m "feat: switch Phase2Trainer diffusion loss to v-prediction"
```

---

## Task 5: Switch evaluate_phase2_gate to DPMSolverPP

Replace the hand-written DDIM loop in the gate function with the new DPMSolverPP sampler.

**Files:**
- Modify: `src/denoisr/training/phase2_trainer.py:226-301` (evaluate_phase2_gate function)
- Test: `tests/test_training/test_phase2_trainer.py`

**Step 1: Run existing gate test (baseline)**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestPhase2Gate -v`

Expected: PASS

**Step 2: Replace DDIM loop with DPMSolverPP**

In `src/denoisr/training/phase2_trainer.py`, add import at top:

```python
from denoisr.nn.diffusion import CosineNoiseSchedule, DPMSolverPP
```

Replace the diffusion-conditioned accuracy block (the DDIM loop and fusion) in `evaluate_phase2_gate`.

Before (the entire DDIM denoising block):
```python
        # Diffusion-conditioned accuracy (DDIM-like denoising)
        x = torch.randn_like(latent)
        num_ts = schedule.num_timesteps
        step_size = max(1, num_ts // num_diff_steps)

        for i in range(num_diff_steps):
            t_val = max(0, num_ts - 1 - i * step_size)
            t = torch.full(
                (boards.shape[0],), t_val, device=device,
            )
            noise_pred = diffusion(x, t, latent)

            ab_t = schedule.alpha_bar[t_val]
            x0_pred = (
                (x - (1 - ab_t).sqrt() * noise_pred) / ab_t.sqrt()
            )

            t_prev = max(0, t_val - step_size)
            if t_prev > 0:
                ab_prev = schedule.alpha_bar[t_prev]
                x = (
                    ab_prev.sqrt() * x0_pred
                    + (1 - ab_prev).sqrt() * noise_pred
                )
            else:
                x = x0_pred

        fused = (latent + x) / 2
```

After:
```python
        # Diffusion-conditioned accuracy (DPM-Solver++)
        solver = DPMSolverPP(schedule, num_steps=num_diff_steps)
        x = solver.sample(diffusion, latent.shape, latent, device)
        fused = (latent + x) / 2
```

**Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestPhase2Gate -v`

Expected: PASS (returns three floats, accuracies in [0, 1])

**Step 4: Commit**

```bash
git add src/denoisr/training/phase2_trainer.py
git commit -m "feat: replace DDIM loop with DPMSolverPP in Phase 2 gate"
```

---

## Task 6: Switch DiffusionChessEngine to DPMSolverPP

Replace the DDIM loop in the inference engine with DPMSolverPP for 2-3x faster move generation.

**Files:**
- Modify: `src/denoisr/inference/diffusion_engine.py:87-110` (_diffusion_imagine method)
- Test: `tests/test_inference/test_diffusion_engine.py`

**Step 1: Run existing tests (baseline)**

Run: `uv run pytest tests/test_inference/test_diffusion_engine.py -v`

Expected: All 3 tests PASS

**Step 2: Replace _diffusion_imagine with DPMSolverPP**

In `src/denoisr/inference/diffusion_engine.py`, add import at top:

```python
from denoisr.nn.diffusion import CosineNoiseSchedule, DPMSolverPP
```

Replace `_diffusion_imagine` method.

Before:
```python
    def _diffusion_imagine(self, latent: torch.Tensor) -> torch.Tensor:
        """Run DDIM-style iterative denoising to imagine future trajectories."""
        x = torch.randn_like(latent)
        T = self._schedule.num_timesteps
        step_size = max(1, T // self._num_steps)

        for i in range(self._num_steps):
            t_val = max(0, T - 1 - i * step_size)
            t = torch.tensor([t_val], device=self._device)
            noise_pred = self._diffusion(x, t, latent)

            ab_t = self._schedule.alpha_bar[t_val]
            x0_pred = (x - (1 - ab_t).sqrt() * noise_pred) / ab_t.sqrt()

            t_prev = max(0, t_val - step_size)
            if t_prev > 0:
                ab_prev = self._schedule.alpha_bar[t_prev]
                x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * noise_pred
            else:
                x = x0_pred

        return (latent + x) / 2
```

After:
```python
    def _diffusion_imagine(self, latent: torch.Tensor) -> torch.Tensor:
        """Run DPM-Solver++ denoising to imagine future trajectories."""
        solver = DPMSolverPP(self._schedule, num_steps=self._num_steps)
        x = solver.sample(self._diffusion, latent.shape, latent, self._device)
        return (latent + x) / 2
```

**Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_inference/test_diffusion_engine.py -v`

Expected: All 3 tests PASS (legal moves produced, valid WDL, anytime property)

**Step 4: Commit**

```bash
git add src/denoisr/inference/diffusion_engine.py
git commit -m "feat: replace DDIM with DPMSolverPP in DiffusionChessEngine"
```

---

## Task 7: Update train_phase2.py script (clean up DDIM remnants)

The training script may reference DDIM in log messages or comments. Verify and clean up.

**Files:**
- Modify: `src/denoisr/scripts/train_phase2.py` (if any DDIM references remain)

**Step 1: Check for DDIM references**

Run: `grep -rn "DDIM\|ddim\|noise_pred\|predicted_noise" src/denoisr/scripts/train_phase2.py`

If any references exist, update them. The training script delegates to `Phase2Trainer.train_step()` and `evaluate_phase2_gate()`, both already updated. The script itself should need no code changes -- just verify.

**Step 2: Run full Phase 2 test suite**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py -v`

Expected: All tests PASS

**Step 3: Commit (if changes needed)**

```bash
git add src/denoisr/scripts/train_phase2.py
git commit -m "chore: clean up DDIM references in train_phase2 script"
```

---

## Task 8: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -q`

Expected: All tests PASS (370+ existing + new tests)

**Step 2: Run linter on all modified files**

Run: `uvx ruff check src/denoisr/nn/diffusion.py src/denoisr/training/diffusion_trainer.py src/denoisr/training/phase2_trainer.py src/denoisr/inference/diffusion_engine.py`

Expected: No errors

**Step 3: Run type checker on modified files**

Run: `uv run --with mypy mypy --strict src/denoisr/nn/diffusion.py src/denoisr/training/phase2_trainer.py src/denoisr/inference/diffusion_engine.py`

Expected: Success (or only pre-existing issues)

**Step 4: Verify no epsilon-prediction remnants**

Run: `grep -rn "predicted_noise\|mse_loss.*noise)" src/denoisr/`

Expected: No matches in training or inference code. Only in comments/docs if any.

**Step 5: Commit any final cleanup**

```bash
git add -A
git commit -m "chore: final verification for v-prediction + DPM-Solver++"
```

---

## Reference: Key file locations

| File | Purpose |
|---|---|
| `src/denoisr/nn/diffusion.py` | CosineNoiseSchedule (v-pred helpers), DPMSolverPP (new), DiTBlock, ChessDiffusionModule (unchanged) |
| `src/denoisr/training/diffusion_trainer.py` | DiffusionTrainer (v-prediction loss) |
| `src/denoisr/training/phase2_trainer.py` | Phase2Trainer (v-prediction loss), evaluate_phase2_gate (DPMSolverPP) |
| `src/denoisr/inference/diffusion_engine.py` | DiffusionChessEngine (DPMSolverPP inference) |
| `tests/test_nn/test_diffusion.py` | v-prediction roundtrip tests, DPMSolverPP tests |
| `tests/test_training/test_diffusion_trainer.py` | DiffusionTrainer tests |
| `tests/test_training/test_phase2_trainer.py` | Phase2Trainer + gate tests |
| `tests/test_inference/test_diffusion_engine.py` | Inference engine tests |

## Reference: Mathematical identities

v-prediction target:

    v = sqrt(alpha_bar_t) * eps  -  sqrt(1 - alpha_bar_t) * x_0

Recover x_0 from v:

    x_0 = sqrt(alpha_bar_t) * x_t  -  sqrt(1 - alpha_bar_t) * v

Recover eps from v:

    eps = sqrt(1 - alpha_bar_t) * x_t  +  sqrt(alpha_bar_t) * v

DPM-Solver++ update:

    x_{t-1} = (sigma_{t-1}/sigma_t) * x_t  -  alpha_{t-1} * (exp(-h) - 1) * D

Log-SNR:

    lambda_t = log(sqrt(alpha_bar_t) / sqrt(1 - alpha_bar_t))

2nd-order correction:

    D = (1 + 1/(2r)) * eps_t  -  (1/(2r)) * eps_{t+1},  where r = h_prev/h
