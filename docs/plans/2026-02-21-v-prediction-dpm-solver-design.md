# v-Prediction + DPM-Solver++ Design

**Date:** 2026-02-21
**Goal:** Replace ε-prediction training with v-prediction and replace DDIM inference with DPM-Solver++ for faster, more stable diffusion in Denoisr.

## Context

Denoisr uses a DiT-based diffusion module (6-layer transformer with AdaLN-Zero) to imagine future board states in latent space. Currently:
- **Training**: DDPM ε-prediction objective (`||ε̂ - ε||²`)
- **Inference**: DDIM η=0 deterministic sampling (10 steps)

Problems with ε-prediction:
- At high noise levels (large t), the input is nearly pure noise, so predicting "which noise was added" gives vanishing gradient signal
- DDIM requires 10+ steps for acceptable quality
- The curriculum (25% → 100% of timesteps) suffers most when reaching the high-noise regime

## Design

### v-Prediction Training

The velocity target blends noise and signal:

```
v = √ᾱ_t · ε - √(1-ᾱ_t) · x₀
```

This keeps gradients informative across all timesteps. From the predicted v̂, both x₀ and ε are recoverable:

```
x₀_pred = √ᾱ_t · x_t - √(1-ᾱ_t) · v̂
ε_pred  = √(1-ᾱ_t) · x_t + √ᾱ_t · v̂
```

The DiT architecture is completely unchanged — same input shape `[B, 64, d_s]`, same output shape. Only the loss target and inference recovery formula change.

Training loss becomes:
```
v = schedule.compute_v_target(x₀, ε, t)
v̂ = diffusion_model(x_t, t, condition)
loss = MSE(v̂, v)
```

### DPM-Solver++ Inference

DPM-Solver++ (Lu et al. 2022) is a 2nd-order ODE solver for the diffusion probability flow ODE. It achieves DDIM-quality results in 3-5 steps instead of 10-20.

The algorithm operates in log-SNR space:
```
λ_t = log(√ᾱ_t / √(1-ᾱ_t))
```

First-order update (each step):
```
h = λ_{t-1} - λ_t
x_{t-1} = (σ_{t-1}/σ_t) · x_t - α_{t-1} · (e^{-h} - 1) · D
```

Second-order correction (reuses previous model output):
```
r = h_{prev} / h
D = (1 + 1/(2r)) · model_output_t - (1/(2r)) · model_output_{t+1}
```

With v-prediction, convert v̂ → ε̂ before plugging into the update formula.

### Fusion

The fusion strategy remains a simple average:
```
enriched = (current_latent + denoised_future) / 2
```

## Changes

### `nn/diffusion.py`

Add to `CosineNoiseSchedule`:
- `compute_v_target(x_0, noise, t) -> Tensor` — computes `v = √ᾱ·ε - √(1-ᾱ)·x₀`
- `predict_x0_from_v(x_t, v, t) -> Tensor` — recovers `x₀ = √ᾱ·x_t - √(1-ᾱ)·v`
- `predict_eps_from_v(x_t, v, t) -> Tensor` — recovers `ε = √(1-ᾱ)·x_t + √ᾱ·v`

Add new class `DPMSolverPP`:
- Constructor takes `CosineNoiseSchedule` and `num_steps`
- Precomputes evenly-spaced timestep schedule in log-SNR space
- `sample(model_fn, shape, condition, device) -> Tensor` — full sampling loop
- Implements 2nd-order multistep with model output buffer

### `training/phase2_trainer.py`

In `train_step()`, change diffusion loss computation:
```python
# Before (ε-prediction):
diffusion_loss = F.mse_loss(predicted_noise, noise)

# After (v-prediction):
v_target = self.schedule.compute_v_target(diff_target, noise, t)
v_pred = self.diffusion(noisy_target, t, cond)
diffusion_loss = F.mse_loss(v_pred, v_target)
```

In `evaluate_phase2_gate()`, replace DDIM loop with `DPMSolverPP.sample()`.

### `training/diffusion_trainer.py`

Same v-prediction change as phase2_trainer.

### `inference/diffusion_engine.py`

Replace `_diffusion_imagine()` DDIM loop with `DPMSolverPP.sample()`.

### Unchanged files

- `DiTBlock`, `ChessDiffusionModule` — architecture stays identical
- `world_model.py` — untouched
- `consistency.py` — untouched
- `loss.py` — untouched
- `encoder.py`, `policy_backbone.py`, `policy_head.py`, `value_head.py` — untouched

## Breaking changes

Checkpoints trained with ε-prediction are **incompatible** with v-prediction for the diffusion module weights only. Phase 1 checkpoints (no diffusion weights) are fully compatible. Phase 2 must be retrained.

## Test strategy

- `test_diffusion.py`: Add v-prediction roundtrip tests (compute v, recover x₀, verify match)
- `test_diffusion.py`: Add DPMSolverPP convergence test (denoise known signal in toy setup)
- `test_phase2_trainer.py`: Verify all 6 losses finite and decreasing with v-prediction
- `test_diffusion_trainer.py`: Update for v-prediction loss
- `test_diffusion_engine.py`: Verify inference produces legal moves with DPM-Solver++

## Dependencies

No new dependencies. All changes are pure PyTorch.
