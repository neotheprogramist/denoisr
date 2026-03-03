# Fail-Fast Audit: Remove Fallbacks, Enforce Single Happy Path

**Date:** 2026-03-03
**Goal:** Remove all fallback code paths and enforce a single happy path with fail-fast descriptive errors throughout the codebase.

## Principle

Every function should have exactly one execution path. When something goes wrong, it raises a descriptive error immediately rather than silently degrading to an alternative behavior. The only exceptions are:

1. Mathematical edge cases that are inherent to the algorithm (not error recovery)
2. Training/eval mode differences (DropPath, gradient checkpointing)
3. Numerical precision requirements (AMP autocast disable for WDL)

## Changes

### 1. SupervisedTrainer: Remove extra LR schedulers

**File:** `src/denoisr/training/supervised_trainer.py`
**Keep:** CosineAnnealingWarmRestarts (T_0=20, T_mult=2) with linear warmup
**Remove:**
- OneCycleLR path and `use_onecycle` parameter
- Plain CosineAnnealingLR fallback path
- All `use_onecycle`/`use_warm_restarts` boolean flags

**Config changes:** Remove `use_warm_restarts`, `use_onecycle`, `onecycle_pct_start` from TrainingConfig and CLI args.

### 2. ResourceMonitor: Fail-fast GPU monitoring

**File:** `src/denoisr/training/resource_monitor.py`
**Current:** Silently skips NVML metrics if pynvml not installed or queries fail.
**Change:** Constructor takes `require_gpu: bool`. If True, pynvml must be importable and nvmlInit() must succeed, or raise RuntimeError. Individual metric queries raise instead of silently returning empty dicts.

### 3. StockfishOracle: Remove sigmoid WDL fallback

**File:** `src/denoisr/data/stockfish_oracle.py`
**Current:** Falls back to sigmoid approximation if WDL not in engine info.
**Change:** Raise ValueError("Stockfish did not return WDL. Requires Stockfish 14+ with WDL support.") if WDL missing.

### 4. MCTS: Require board-state tracking

**File:** `src/denoisr/training/mcts.py`
**Current:** Two paths for legal mask — board-aware (A) and empirical policy support (B).
**Change:** Remove Path B entirely. `transition_fn` and `legal_mask_fn` become required constructor parameters (not Optional). Every expanded node must have a chess.Board.

### 5. ChessLossComputer: Require explicit legal mask

**File:** `src/denoisr/training/loss.py`
**Current:** `policy_legal_mask` is Optional; infers from target when None.
**Change:** Make `policy_legal_mask` required (Tensor, not Optional). Remove the `target_flat > 0` inference path. Remove the OR merge on line 74.

### 6. Training scripts: Remove DataLoader retry

**Files:** `src/denoisr/scripts/train_phase1.py`, `train_phase2.py`
**Current:** Catches worker crash, retries with workers=0.
**Change:** Remove try/except around DataLoader iteration. Let worker crashes propagate with their original traceback.

### 7. ReplayBuffer: Remove SimpleReplayBuffer

**File:** `src/denoisr/training/replay_buffer.py`
**Current:** Two buffer implementations.
**Change:** Delete SimpleReplayBuffer class. PriorityReplayBuffer is the only buffer.

### 8. Device detection: Log loudly

**File:** `src/denoisr/scripts/config/__init__.py`
**Current:** Silent MPS->CUDA->CPU cascade.
**Change:** Keep auto-detection order but log at WARNING level: "Selected device: cuda (detected CUDA)" or "Selected device: cpu (no GPU detected)".

### 9. GrokfastFilter: Fail-fast on mismatches

**File:** `src/denoisr/training/grokfast.py`
**Current:** 4 recovery paths for shape mismatch, device mismatch, non-finite EMA.
**Change:** Raise ValueError on shape/device mismatch (these indicate bugs). For non-finite gradients: skip the parameter (log warning), don't silently reset EMA state. For non-finite EMA: raise RuntimeError (indicates training divergence).

### 10. Self-play / Reanalyse: Fail-fast on zero visit distribution

**Files:** `src/denoisr/training/self_play.py`, `reanalyse.py`
**Current:** Falls back to uniform distribution when visit_dist.sum() == 0.
**Change:** Raise RuntimeError("MCTS produced zero visit distribution") — this means MCTS is broken.

### 11. Phase gate evaluation: EMA only

**Files:** `src/denoisr/scripts/train_phase1.py`, `train_phase2.py`
**Current:** Tries base -> SWA -> EMA, picks first passing.
**Change:** Evaluate EMA only. Remove SWA entirely (class ModelSWA, all SWA config, CLI flags). Single evaluation path.

### 12. Runtime env loading: Fail if path provided but missing

**File:** `src/denoisr/scripts/runtime.py`
**Current:** `load_env_file()` silently succeeds if file missing.
**Change:** If explicit path provided and file doesn't exist, raise FileNotFoundError. Default `.env` path: log INFO if missing, continue (it's optional by convention).

### 13. Oracle cleanup: Remove bare except

**File:** `src/denoisr/scripts/generate_data.py`
**Current:** `_cleanup_oracle()` has `except: pass`.
**Change:** `except Exception as e: logging.warning(f"Oracle cleanup failed: {e}")` — log but don't crash (cleanup is best-effort in atexit handlers).

### 14. maybe_compile: Make explicit

**File:** `src/denoisr/scripts/config/__init__.py`
**Current:** Silently returns uncompiled module on non-CUDA.
**Change:** Log at INFO level: "Skipping torch.compile (requires CUDA, got {device.type})". The behavior is correct but should be visible.

## Files to delete entirely

- `src/denoisr/training/swa.py` (ModelSWA removed — EMA is the single path)

## Config parameters to remove

From TrainingConfig:
- `use_warm_restarts` (always True now)
- `use_onecycle` (removed)
- `onecycle_pct_start` (removed)
- `phase1_swa_eval_every` (SWA removed)

From CLI args across all training scripts:
- `--warm-restarts` / `--no-warm-restarts`
- `--onecycle`
- `--onecycle-pct-start`
- `--swa-eval-every`

## Mathematical soundness verification

The following are confirmed correct and require no changes:

1. **v-prediction system** (diffusion.py): All 4 equations are algebraically consistent
2. **DPM-Solver++ 2nd-order** (diffusion.py): Correct log-SNR computation, 1st-order bootstrap is inherent to multistep methods
3. **SimSiam consistency** (phase2_trainer.py): Stop-gradient prevents collapse
4. **HarmonyDream** (loss.py): Inverse-magnitude balancing is mathematically sound
5. **Cosine noise schedule** (diffusion.py): Double-precision computation with float32 clamping prevents accumulation errors
6. **Policy head bilinear scoring** (policy_head.py): Q*K^T with correct sqrt(d_head) scaling
7. **Value head attention pooling** (value_head.py): Correct sqrt(d_s) scaling with AMP precision guard
8. **MCTS UCB** (mcts.py): Standard PUCT formula with Dirichlet noise

## Items intentionally kept (not fallbacks)

1. DPM-Solver++ 1st-order on first step — inherent to multistep solvers
2. DropPath identity in eval mode — standard stochastic depth behavior
3. Gradient checkpointing train/eval toggle — memory optimization, same computation
4. AMP autocast disable for WDL in value_head — numerical precision requirement
5. Augmentation sum=0 guards — mathematical edge case handling
6. Diffusion curriculum initial fraction — intentional easy-to-hard progression
