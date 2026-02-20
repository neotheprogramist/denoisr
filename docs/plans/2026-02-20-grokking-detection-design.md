# Grokking Detection for Phase 1 Training

## Problem

Grokking — the dramatic delayed jump from memorization to generalization — is a well-documented phenomenon in transformer training on structured data (Power et al. 2022). A chess engine trained on fixed supervised data exhibits exactly the conditions under which grokking occurs: algorithmic structure in the domain, fixed dataset, train/test split. Detecting grokking requires metrics beyond standard loss curves because the visible accuracy jump is a lagging indicator — the generalizing circuit forms gradually during a hidden-progress phase that precedes it by thousands of epochs.

## Scope

- Phase 1 supervised training only (fixed dataset with train/holdout split)
- Tier 1 metrics: weight norms, gradient norms, train/holdout loss gap (every step or epoch)
- Tier 2 metrics: effective rank, spectral norms, HTSR alpha (every 1K-5K steps)
- Multiple structured holdout splits (game-level, opening-family, piece-count)
- Adaptive evaluation frequency with console alerts
- Optional Grokfast acceleration (50x grokking speedup via EMA gradient filtering)

## Architecture

### Approach: standalone GrokTracker

A purpose-built `GrokTracker` class integrated into `SupervisedTrainer`. Registers forward hooks on backbone layers to capture activations. Computes metrics at configurable frequencies. Contains an adaptive-frequency state machine. Fires console alerts. Delegates logging to the existing `TrainingLogger`.

Rejected alternatives:
- **Hook-only observer** — zero trainer modification but fragile with autocast/gradient scaling, can't access loss breakdown without duplication.
- **Generic callback system** — extensible but premature abstraction for a single use case (YAGNI).

## New files

### `src/denoisr/training/grok_tracker.py`

Central class. Responsibilities:

1. Receives model at construction, registers forward hooks on all 15 backbone `TransformerBlock` layers
2. Stores captured activations in a ring buffer (latest batch only — no memory bloat)
3. Exposes `step(global_step, loss_breakdown, grad_norm)` — called after every training step
4. Exposes `epoch(epoch, holdout_metrics)` — called after every epoch
5. Manages tiered evaluation schedule with adaptive frequency
6. Writes to `TrainingLogger` and prints console alerts via `logging.WARNING`

### `src/denoisr/training/grokfast.py`

`GrokfastFilter` dataclass. Stateless except for the EMA gradient dictionary. Applied between `loss.backward()` (after `scaler.unscale_()`) and `optimizer.step()` in the training loop.

Implementation: for each parameter with a gradient, maintain an EMA of gradient history. Add `lamb * ema_grad` to the current gradient, amplifying slow-varying (generalizing) components while leaving fast-varying (memorizing) components alone.

Mixed-precision interaction: `scaler.unscale_()` must be called before applying the filter so that gradient magnitudes are correct.

### `src/denoisr/data/holdout_splitter.py`

`StratifiedHoldoutSplitter` class. Takes the full list of `TrainingExample` objects and produces four independent holdout sets:

1. **Random holdout** — existing 5% split, baseline generalization test
2. **Game-level holdout** — all positions from ~10% of games reserved for testing (prevents positional continuity leakage)
3. **Opening-family holdout** — entire ECO opening families held out (tests cross-structure generalization)
4. **Piece-count holdout** — all positions with <=6 pieces reserved (tests endgame generalization)

Requires `TrainingExample` to carry `game_id`, `eco_code`, and `piece_count` metadata (populated during data generation).

## Modified files

### `src/denoisr/types/` — TrainingExample

Add three optional fields for backward compatibility:

```python
game_id: int | None = None
eco_code: str | None = None
piece_count: int | None = None
```

Existing `.pt` files without these fields fall back to random-only holdout.

### `src/denoisr/scripts/config.py` — TrainingConfig

Add grokking detection and Grokfast configuration:

| Flag | Default | Description |
|------|---------|-------------|
| `--grok-tracking` | `false` | Enable grokking detection |
| `--grok-erank-freq` | `1000` | Effective rank computation frequency (steps) |
| `--grok-spectral-freq` | `5000` | Spectral norm / HTSR alpha frequency (steps) |
| `--grok-onset-threshold` | `0.95` | Weight norm ratio threshold for onset detection |
| `--grokfast` | `false` | Enable Grokfast EMA gradient filtering |
| `--grokfast-alpha` | `0.98` | Grokfast EMA decay rate |
| `--grokfast-lamb` | `2.0` | Grokfast amplification factor |

### `src/denoisr/training/supervised_trainer.py`

- Accept optional `GrokTracker` and `GrokfastFilter` in constructor
- Call `tracker.step()` after each training step
- Call `grokfast_filter.apply()` between `scaler.unscale_()` and `clip_grad_norm_()`

### `src/denoisr/scripts/train_phase1.py`

- Use `StratifiedHoldoutSplitter` to create structured holdout sets
- Evaluate each holdout set independently at epoch end
- Pass holdout metrics to `tracker.epoch()`
- Construct `GrokTracker` and `GrokfastFilter` from config

### `src/denoisr/scripts/generate_data.py`

- Assign sequential `game_id` per PGN game
- Extract ECO code from PGN headers (fall back to None if absent)
- Compute `piece_count` from board state

### `src/denoisr/training/logger.py`

- Add `log_grok_metrics(step, metrics_dict)` method for grokking-specific scalars
- Add `log_grok_state_transition(step, old_state, new_state, trigger)` for marking transitions

## Metrics

### Tier 1 — every step (near-zero cost)

- `grok/weight_norm_total` — total L2 norm of all parameters
- `grok/weight_norm/{encoder,backbone,policy_head,value_head}` — per-module norms
- Gradient norms reuse existing `gradients/norm`

### Tier 1 — every epoch

- `grok/loss_gap` — train_loss minus holdout_loss
- `grok/holdout/{random,game_level,opening_family,piece_count}/accuracy`
- `grok/holdout/{random,game_level,opening_family,piece_count}/loss`

### Tier 2 — every N steps (moderate cost)

- `grok/erank/layer_{i}` — effective rank of activations per backbone layer (default every 1000 steps)
- `grok/spectral_norm/layer_{i}/{attn,ffn}` — largest singular value per weight matrix (default every 5000 steps)
- `grok/alpha/layer_{i}` — HTSR power-law exponent of weight eigenvalue spectrum (default every 5000 steps)

### State transitions

- `grok/state` — integer encoding (0=BASELINE, 1=ONSET_DETECTED, 2=TRANSITIONING, 3=GROKKED)
- `grok/grokking_gap_steps` — steps between train saturation and holdout threshold (logged once)
- `grok/grokking_gap_epochs` — epochs between train saturation and holdout threshold (logged once)

## State machine

```
BASELINE → ONSET_DETECTED → TRANSITIONING → GROKKED
```

**BASELINE → ONSET_DETECTED** when any of:
- Weight norm mean (last 50 steps) < 95% of mean (steps [-100, -50])
- Any holdout accuracy improves >2 percentage points in a 10-epoch window
- Effective rank drops >10% in a 1000-step window

**ONSET_DETECTED → TRANSITIONING** when:
- Holdout accuracy improves >5 percentage points in a 20-epoch window while train accuracy is saturated (>95%)

**TRANSITIONING → GROKKED** when:
- Holdout accuracy exceeds a configurable threshold (default 25%, well above random ~1%)

Actions on transition:
- ONSET_DETECTED: increase Tier 2 eval frequencies 5x, print console alert
- TRANSITIONING: increase Tier 2 eval frequencies 10x, print console alert
- GROKKED: log final grokking gap metrics, print console alert

## Grokfast

Opt-in EMA gradient filter (Lee et al. 2024). For each parameter:

```
ema_grad = alpha * ema_grad + (1 - alpha) * current_grad
current_grad += lamb * ema_grad
```

This amplifies slow-varying gradient components (the generalizing circuit's signal) by factor `lamb` while leaving fast-varying components (memorization) alone. Achieves ~50x speedup of grokking in published results.

Applied after `scaler.unscale_()` and before `clip_grad_norm_()` in the training step. Controlled by `--grokfast` flag (default off).

## References

- Power et al. (2022) — "Grokking: Generalization beyond Overfitting on Small Algorithmic Datasets"
- Nanda et al. (2023) — "Progress Measures for Grokking via Mechanistic Interpretability" (ICLR Oral)
- Liu et al. (2023) — "Omnigrok: Grokking Beyond Algorithmic Data"
- Lee et al. (2024) — "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"
- Zunkovic & Ilievski (2024) — "Effective Dimensionality Reduction for Grokking Detection" (JMLR)
- McGrath et al. (2022) — "Acquisition of Chess Knowledge in AlphaZero" (PNAS)
- Karvonen (2024) — linear probes for chess transformer representations
