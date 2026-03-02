# Phase 1 Training Enhancements — "Fix the Foundations"

**Date:** 2026-02-28
**Goal:** Break the ~47% top-1 accuracy plateau in Phase 1 supervised training
**Constraint:** Stay within RTX 3060 12GB VRAM budget

## Problem Analysis

Training logs from a 24-epoch run reveal:

1. **Post-warmup regression:** Top-1 accuracy peaks at 48.11% (EMA, epoch 10) then oscillates between 43-47% for epochs 11-24.
2. **Gradient norm explosion:** Mean gradient norms rise from 1.6 (E10) to 22.3 (E23), with peaks from 12 to 163 — a 14x increase while loss barely moves.
3. **Underpowered regularization:** The 26.6M parameter model trained on 4M positions (6.6 params/example) has zero dropout, only 2% label smoothing (baked into data), and one augmentation (color flip).
4. **LR schedule mismatch:** CosineAnnealingWarmRestarts with T_0=20 has its first cycle ending at epoch 30, but the plateau starts at epoch 11. The schedule doesn't match the learning dynamics.

**Root cause:** The model memorizes quickly during warmup, then oscillates as the cosine schedule interacts with an overfitting-prone landscape. The lack of regularization means the model learns sharp, unstable features rather than robust ones.

## Enhancements

### 1. Dropout + Stochastic Depth

**Files:** `nn/policy_backbone.py`, `nn/drop_path.py` (new)

Add dropout (p=0.1) after attention output projection and after FFN in each TransformerBlock. Add stochastic depth (DropPath) to residual connections, linearly scaled from 0.0 at layer 0 to `drop_path_rate` at the final layer.

DropPath randomly skips entire residual branches during training, forcing each layer to be independently useful and breaking inter-layer co-adaptation that causes gradient confusion.

**Config:** `ModelConfig.dropout=0.1`, `ModelConfig.drop_path_rate=0.1`

### 2. OneCycleLR Scheduler

**Files:** `training/supervised_trainer.py`

Replace CosineAnnealingWarmRestarts with OneCycleLR:
- Per-step scheduling (not per-epoch) for smoother LR trajectory
- 30% warmup (30 of 100 epochs) for gradual basin discovery
- Coupled LR/momentum annealing for super-convergence
- Automatic final annealing to near-zero

**Config:** `TrainingConfig.use_onecycle=True`, `TrainingConfig.onecycle_pct_start=0.3`

### 3. Gradient Accumulation

**Files:** `training/supervised_trainer.py`

Accumulate gradients over 4 micro-batches before stepping the optimizer, giving an effective batch size of 4096 (1024 x 4) without increasing VRAM. This smooths gradient noise that currently causes norm spikes up to 163.

**Config:** `TrainingConfig.gradient_accumulation_steps=4`

### 4. Legal-Move-Aware Label Smoothing

**Files:** `training/loss.py`

Add label smoothing in the loss function (not data generation), distributing smoothing probability only among legal moves (not all 4672 slots). This prevents overconfident predictions while preserving the chess-specific constraint that illegal moves should never be predicted.

Stacks with the 2% data-time smoothing for ~12% effective smoothing.

**Config:** `TrainingConfig.label_smoothing=0.1`

### 5. Soft Target Augmentation

**Files:** `training/dataset.py`, `training/augmentation.py`

Two lightweight augmentations that perturb targets rather than inputs (since chess has no spatial symmetries beyond color flip):

- **Value noise** (20% prob): Add small Gaussian noise to WDL targets, re-normalize. Prevents the value head from memorizing exact evaluation scores.
- **Policy temperature** (30% prob): Randomly sharpen or flatten policy distribution by applying temperature in [0.8, 1.2]. Simulates varying Stockfish analysis confidence.

**Config:** `TrainingConfig.value_noise_prob=0.2`, `TrainingConfig.value_noise_scale=0.02`, `TrainingConfig.policy_temp_augment_prob=0.3`

## Files to Modify

| File | Changes |
|---|---|
| `nn/policy_backbone.py` | Add dropout, DropPath to TransformerBlock |
| `nn/drop_path.py` (new) | DropPath module implementation |
| `training/supervised_trainer.py` | OneCycleLR, gradient accumulation |
| `training/loss.py` | Legal-move-aware label smoothing |
| `training/dataset.py` | Soft target augmentation dispatch |
| `training/augmentation.py` | Value noise + policy temperature functions |
| `scripts/config/__init__.py` | New config fields |
| `scripts/train_phase1.py` | Pass new config to trainer/model |
| `.env.example` | Document new environment variables |

## Expected Outcomes

- Gradient norms stabilize under 5.0 throughout training
- No regression after warmup — continuous improvement curve
- Top-1 accuracy ceiling raised from ~47% to ~55%+ (EMA)
- Training stability sufficient to reach epoch 100 without stability guard triggers

## Environment Variables

```env
DENOISR_MODEL_DROPOUT=0.1
DENOISR_MODEL_DROP_PATH_RATE=0.1
DENOISR_TRAIN_USE_ONECYCLE=1
DENOISR_TRAIN_ONECYCLE_PCT_START=0.3
DENOISR_TRAIN_GRADIENT_ACCUMULATION_STEPS=4
DENOISR_TRAIN_LABEL_SMOOTHING=0.1
DENOISR_TRAIN_VALUE_NOISE_PROB=0.2
DENOISR_TRAIN_VALUE_NOISE_SCALE=0.02
DENOISR_TRAIN_POLICY_TEMP_AUGMENT_PROB=0.3
```
