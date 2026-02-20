# Phase 1 Training Optimization — Implementation Design

## Context

Implements all 12 changes from `docs/plans/phase1_changes_summary.md`. The codebase is currently in its pre-optimization state — none of the changes exist yet. Must integrate cleanly with grokking detection features already on the `feat/grokking-detection` branch (GrokfastFilter, GrokTracker, grokking logger methods).

## Grokking Integration Analysis

The 12 planned changes and existing grokking code touch orthogonal surfaces:
- `breakdown` dict gets a new `"overflow"` key — grokking code ignores `breakdown` contents
- Batched `measure_accuracy()` doesn't affect grokking hooks (hooks fire during training, accuracy is measured after)
- Value head returning logits doesn't affect GrokTracker (reads weight tensors, never calls `forward()`)
- Logger inf-filtering touches `log_training_dynamics()`, grokking uses separate `log_grok_metrics()` — no overlap

**No conflicts. No special integration work required.**

## Execution Strategy

**Approach B: 3 logical commits, each including its test updates so tests always pass.**

---

## Commit 1: Numerical Stability + API Adaptation

**Items**: #1, #2, #3, #4, #12 | **Commit message**: `fix: return raw logits from value head for numerical stability`

### `src/denoisr/nn/value_head.py`

- **`forward()`**: Remove `torch.softmax()`. Add `torch.amp.autocast("cuda", enabled=False)` around WDL linear projection with `pooled.float()` for AMP safety. Return raw logits.
- **`infer()`**: New method — calls `forward()`, applies `softmax` to WDL logits. Used by inference callers that need probabilities.

### `src/denoisr/training/loss.py`

- Line ~69-70: Replace `torch.log(pred_value.clamp(min=1e-8))` with `F.log_softmax(pred_value, dim=-1)`.

### `src/denoisr/inference/engine.py` + `src/denoisr/inference/diffusion_engine.py`

- Change `self._value_head(features)` → `self._value_head.infer(features)`.

### `src/denoisr/scripts/train_phase3.py`

- In `policy_value_fn` closure: add `torch.softmax(wdl_logits, dim=-1)` before returning WDL to MCTS.

### `tests/test_nn/test_value_head.py`

- Probability tests (`sums_to_one`, `in_zero_one`) call `head.infer()` instead of `head()`.
- New tests: `forward()` shape check, logits are finite (not constrained to [0,1]), gradient flow using `wdl_logits.sum() + ply.sum()`.

---

## Commit 2: Bug Fixes + Observability

**Items**: #5, #6, #7, #11 | **Commit message**: `fix: warmup LR initialization, overflow detection, and grad norm filtering`

### `src/denoisr/training/supervised_trainer.py`

- **Warmup LR init**: After recording `_base_lrs`, set `g["lr"] = base_lr / warmup_epochs` for each param group.
- **Overflow detection**: After `clip_grad_norm_`, set `breakdown["overflow"] = not math.isfinite(grad_norm)`.

### `src/denoisr/training/logger.py`

- **`log_training_dynamics()`**: Filter inf/nan from grad norms before computing mean/peak.
- **`log_train_step()`**: Skip `"overflow"` key (boolean, not a TensorBoard scalar).

### `src/denoisr/scripts/train_phase1.py`

- Exclude overflow steps from `step_grad_norms`.
- Count overflow events per epoch.
- Report `overflows=N` in epoch summary.

### `tests/test_training/test_supervised_trainer.py`

- `test_scheduler_reduces_lr`: Compare against `_base_lrs` (peak LRs), run `warmup_epochs + 5` steps.

---

## Commit 3: Performance + Tuning

**Items**: #8, #9, #10 | **Commit message**: `perf: optimize phase 1 defaults, batch accuracy evaluation, per-group LR logging`

### Hyperparameter Defaults

| Parameter | Old | New | Location |
|-----------|-----|-----|----------|
| `--batch-size` | 64 | 1024 | `train_phase1.py` argparse |
| `--lr` | 1e-4 | 3e-4 | `train_phase1.py` argparse |
| `encoder_lr_multiplier` | 0.3 | 1.0 | `config.py` or `train_phase1.py` CLI |
| `warmup_epochs` | 3 | 5 | `config.py` or `train_phase1.py` CLI |
| `max_grad_norm` | 1.0 | 5.0 | `config.py` or `train_phase1.py` CLI |

### Batched Accuracy Evaluation (`train_phase1.py`)

Rewrite `measure_accuracy()` to process in batches of 256:
- Stack examples into tensors
- Batched forward pass: encoder → backbone → policy_head
- Compute top-1/top-5 from batched logits
- Also benefits grokking holdout evaluation (same function, 4 splits)

### Per-Group LR Logging (`train_phase1.py`)

- Log encoder LR (`param_groups[0]`) and head LR (`param_groups[2]`) in epoch summary + TensorBoard.

---

## Verification

After all 3 commits: `uv run pytest tests/ -x -q` — all tests must pass.
