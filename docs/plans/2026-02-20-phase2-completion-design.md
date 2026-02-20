# Phase 2 Completion Design

## Problem

Phase 2 currently trains only the diffusion module (noise prediction MSE). Three modules are built but left untrained: world model, consistency projector, and the existing policy/value heads receive no continued supervision. The README describes a 6-term HarmonyDream loss that does not exist in Phase 2.

## Approach: Unified Phase2Trainer

A single `Phase2Trainer` class that combines all 6 loss terms in one training step per batch, flowing through `ChessLossComputer` with optional HarmonyDream balancing.

## Data: Enriched Trajectories

The `extract_trajectories` function is expanded to produce richer data:

```python
@dataclass(frozen=True)
class TrajectoryBatch:
    boards: Tensor      # [N, T, C, 8, 8]  — consecutive board states
    actions_from: Tensor # [N, T-1]          — from-square per move
    actions_to: Tensor   # [N, T-1]          — to-square per move
    policies: Tensor     # [N, T, 64, 64]    — one-hot from played move
    values: Tensor       # [N, T, 3]         — WDL from game result
```

Each trajectory contains T consecutive board states connected by T-1 actions. Policy targets are one-hot from the move played. Value targets come from the game result (no Stockfish required). This is cheaper than Stockfish-based targets but sufficient — Phase 1 already taught the strong targets, Phase 2 supervision just prevents catastrophic forgetting.

## Training Step

```
Input: TrajectoryBatch slice [B, T, ...]
                    |
    ┌───────────────┴───────────────┐
    │   Frozen encoder (no grad)     │
    │   boards → latent [B,T,64,d_s] │
    └───────────────┬───────────────┘
                    |
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
 Diffusion      World Model     Backbone→Heads
 (noise pred    (state,action   (per-position
  on future     → next state    policy+value
  latent)       + reward)       supervision)
    │               │               │
    ▼               ▼               ▼
 MSE loss       MSE(state)      CE(policy)
                + MSE(reward)   + CE(value)
                    │
                    ▼
              Consistency
              (SimSiam on
               predicted vs
               actual next)
                    │
                    ▼
              -cos_sim loss
```

### Loss terms

1. **Policy** — cross-entropy between predicted logits and one-hot target (with legal move masking)
2. **Value** — cross-entropy between predicted WDL and game-result WDL
3. **Diffusion** — MSE between predicted noise and actual noise on corrupted future latents
4. **World model state** — MSE between predicted next latent state and actual next latent state
5. **World model reward** — MSE between predicted reward and game outcome signal
6. **Consistency** — negative cosine similarity between consistency projector outputs of predicted and actual next states (stop-gradient on actual)

The reward signal is derived from the game result: +1 for the winning side's moves, -1 for the losing side's moves, 0 for draws.

All 6 losses flow through `ChessLossComputer.compute()` with auxiliary terms passed as kwargs. HarmonyDream balancing is available via `--harmony-dream`.

## Optimizer Configuration

| Module | Phase 2 Status | LR |
|---|---|---|
| Encoder | Frozen (no grad) | 0 |
| Backbone | Trainable (slow) | lr × encoder_lr_multiplier (0.3×) |
| Policy head | Trainable | lr |
| Value head | Trainable | lr |
| World model | New, trainable | lr |
| Diffusion | New, trainable | lr |
| Consistency projector | New, trainable | lr |

A single AdamW optimizer with parameter groups handles differential learning rates.

## Diffusion Curriculum

Unchanged from current: timesteps grow from 25% → 100% over ~70 epochs via `curriculum_growth=1.02`.

## Phase 2 Gate (Automated)

After training completes, automatically evaluate:

1. Encode a holdout set of positions with the frozen encoder
2. Run backbone → policy_head → compute top-1 accuracy (single-step baseline)
3. Run diffusion denoising → backbone → policy_head → compute top-1 accuracy (diffusion-conditioned)
4. Compute delta = diffusion_accuracy - single_accuracy
5. If delta > phase2_gate (default 5pp) → log success, save "phase2_passed" flag
6. If not → log warning with actual delta, save anyway (user decides)

The gate evaluation uses the same holdout fraction as Phase 1 (default 5%).

## Files Changed

| File | Change |
|---|---|
| `training/phase2_trainer.py` | **New** — Phase2Trainer with 6-loss training step |
| `scripts/train_phase2.py` | Modify — enriched trajectory extraction, Phase2Trainer, automated gate |
| `tests/test_training/test_phase2_trainer.py` | **New** — unit tests for Phase2Trainer |

## CLI Flags

No new flags needed. All existing Phase 2 flags apply. The `--harmony-dream` flag from TrainingConfig now takes effect in Phase 2 (currently it has no effect).

## Success Criteria

1. All 6 loss terms are computed and logged per step
2. World model, diffusion, and consistency modules receive gradient updates
3. Encoder is frozen, backbone trains at reduced LR
4. HarmonyDream dynamically rebalances when enabled
5. Phase 2 gate automatically evaluated and logged
6. All existing tests pass, new trainer tests pass
7. TensorBoard shows all 6 loss curves + HarmonyDream coefficients
