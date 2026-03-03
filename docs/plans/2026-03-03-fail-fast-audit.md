# Fail-Fast Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all fallback code paths and enforce a single happy path with fail-fast descriptive errors throughout the codebase.

**Architecture:** Systematic removal of alternative code paths, fallback mechanisms, and silent degradation. Each change enforces exactly one execution path per function. When something goes wrong, a descriptive error is raised immediately.

**Tech Stack:** Python 3.14, PyTorch, uv, pytest

**Test runner:** `uv run pytest tests/ -x -q`
**Linter:** `uvx ruff check`

---

### Task 1: Remove SimpleReplayBuffer

**Files:**
- Modify: `src/denoisr/training/replay_buffer.py:7-29` (delete SimpleReplayBuffer)
- Modify: `tests/test_training/test_replay_buffer.py` (remove SimpleReplayBuffer tests)

**Step 1: Read and identify SimpleReplayBuffer test references**

Find all test functions referencing SimpleReplayBuffer in `tests/test_training/test_replay_buffer.py`.

**Step 2: Delete SimpleReplayBuffer class**

Remove lines 7-29 from `src/denoisr/training/replay_buffer.py` (the entire SimpleReplayBuffer class).

**Step 3: Update tests**

Remove any test functions that test SimpleReplayBuffer. Update imports if needed. PriorityReplayBuffer tests stay.

**Step 4: Search for other SimpleReplayBuffer imports**

```bash
uv run python -c "import ast; print('ok')"  # verify env
```

Search all source and test files for `SimpleReplayBuffer` references and remove them.

**Step 5: Run tests**

```bash
uv run pytest tests/test_training/test_replay_buffer.py -x -q
```

Expected: All remaining PriorityReplayBuffer tests pass.

**Step 6: Run linter**

```bash
uvx ruff check src/denoisr/training/replay_buffer.py
```

**Step 7: Commit**

```bash
git add src/denoisr/training/replay_buffer.py tests/test_training/test_replay_buffer.py
git commit -m "refactor!: remove SimpleReplayBuffer, keep PriorityReplayBuffer as single path"
```

---

### Task 2: Remove SWA (ModelSWA) entirely

**Files:**
- Delete: `src/denoisr/training/swa.py`
- Delete: `tests/test_training/test_swa.py`
- Modify: `src/denoisr/scripts/train_phase1.py` (remove SWA import, init, update, eval, save)
- Modify: `src/denoisr/scripts/train_phase2.py` (remove SWA import, init, update, eval, save)
- Modify: `src/denoisr/scripts/config/__init__.py` (remove phase1_swa_eval_every field and CLI arg)
- Modify: tests referencing SWA

**Step 1: Delete swa.py and its test**

Delete `src/denoisr/training/swa.py` and `tests/test_training/test_swa.py`.

**Step 2: Remove SWA from train_phase1.py**

Remove these elements:
- Line 54: `from denoisr.training.swa import ModelSWA`
- Line 64: `_SWA_START_FRACTION = 0.75`
- Lines 752-753: SWA eval validation
- Lines 911-926: `swa_model = ModelSWA(...)` initialization and logging
- Lines 964-968: SWA cadence logging
- Line 1001: SWA eval every in hparams dict
- Lines 1185-1186: `swa_model.update()` call
- Lines 1199-1217: SWA evaluation block (swa_top1, eval_swa, swa_model.apply)
- Lines 1243-1245: SWA selection in gate comparison
- Lines 1521-1522: SWA model apply for checkpoint save

In the gate evaluation, change from trying base/SWA/EMA to EMA only:
- Keep EMA evaluation as the single gate check
- Remove the `selected_source` / `selected_top1` comparison logic
- Use EMA result directly

**Step 3: Remove SWA from train_phase2.py**

Remove these elements:
- Line 59: `from denoisr.training.swa import ModelSWA`
- Line 62: `_SWA_START_FRACTION = 0.75`
- Lines 334-352: SWA model initialization and logging
- Lines 537-538: SWA update
- Lines 650-672: SWA gate evaluation
- Lines 693-694: SWA model apply for checkpoint save

Same simplification: EMA is the single gate evaluation path.

**Step 4: Remove SWA config from config/__init__.py**

- Line 329-330: Remove `phase1_swa_eval_every` field from TrainingConfig
- Lines 796-801: Remove `--phase1-swa-eval-every` CLI arg registration
- Line 1106: Remove `phase1_swa_eval_every=args.phase1_swa_eval_every` from config construction

**Step 5: Update test files**

Search and update tests referencing SWA:
- `tests/test_scripts/test_config_enhancements.py`
- `tests/test_scripts/test_config_grok.py`
- `tests/test_integration/test_enhancements_smoke.py`
- `tests/test_training/test_supervised_trainer.py`
- Any other files found by grep

Remove SWA-specific test assertions. Keep EMA test assertions.

**Step 6: Run full test suite**

```bash
uv run pytest tests/ -x -q
```

**Step 7: Run linter**

```bash
uvx ruff check src/denoisr/training/ src/denoisr/scripts/
```

**Step 8: Commit**

```bash
git add -A
git commit -m "refactor!: remove SWA, use EMA as single evaluation path"
```

---

### Task 3: Remove OneCycleLR and plain CosineAnnealingLR scheduler paths

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py:37-39,74-75,80-99,112-120,233-234,311-314`
- Modify: `src/denoisr/scripts/config/__init__.py` (remove use_onecycle, onecycle_pct_start, use_warm_restarts fields)
- Modify: tests referencing these scheduler options

**Step 1: Simplify SupervisedTrainer constructor**

Remove parameters: `use_warm_restarts`, `use_onecycle`, `onecycle_pct_start`, `steps_per_epoch`.
Remove instance variables: `self._use_onecycle`, `self._onecycle_total_steps`.

Replace the scheduler selection block (lines 80-120) with just CosineAnnealingWarmRestarts:

```python
        # Start at 1/N of peak LR; warmup will ramp up from here
        for g, base_lr in zip(param_groups, self._base_lrs):
            g["lr"] = base_lr / max(self._warmup_epochs, 1)
        self._scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=20,
                T_mult=2,
                eta_min=min_lr,
            )
        )
```

**Step 2: Simplify _forward_backward**

Remove lines 233-234 (OneCycleLR per-step scheduling).

**Step 3: Simplify scheduler_step**

Remove the `if self._use_onecycle: return` branch (lines 313-314). Keep warmup + scheduler.step().

```python
    def scheduler_step(self) -> None:
        self._epoch += 1
        if self._epoch <= self._warmup_epochs:
            frac = self._epoch / self._warmup_epochs
            for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                group["lr"] = base_lr * frac
        else:
            self._scheduler.step()
```

**Step 4: Remove config fields**

From `src/denoisr/scripts/config/__init__.py`:
- Remove `use_warm_restarts` field (line 137)
- Remove `use_onecycle` field (line 146)
- Remove `onecycle_pct_start` field (lines 149-150)
- Remove corresponding CLI args (lines 589-594, 821-833)
- Remove from `training_config_from_args` construction (lines 1079, 1109-1110)
- Remove from env validation specs (lines 1011-1012)

**Step 5: Update callers**

Search for `use_warm_restarts`, `use_onecycle`, `onecycle_pct_start` in train_phase1.py, train_phase2.py and remove the arguments passed to SupervisedTrainer.

**Step 6: Update tests**

Update `tests/test_training/test_supervised_trainer.py` and any other tests that pass these removed parameters.

**Step 7: Run tests**

```bash
uv run pytest tests/ -x -q
```

**Step 8: Run linter**

```bash
uvx ruff check src/denoisr/training/supervised_trainer.py src/denoisr/scripts/config/__init__.py
```

**Step 9: Commit**

```bash
git add -A
git commit -m "refactor!: single LR scheduler (CosineAnnealingWarmRestarts), remove OneCycleLR and plain cosine"
```

---

### Task 4: Require explicit legal mask in loss computation

**Files:**
- Modify: `src/denoisr/training/loss.py:61,69-76`
- Modify: `src/denoisr/scripts/train_phase1.py` (pass legal mask to loss)
- Modify: tests

**Step 1: Make policy_legal_mask required**

In `src/denoisr/training/loss.py`, change the `compute` signature:

```python
    def compute(
        self,
        pred_policy: Tensor,
        pred_value: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
        policy_legal_mask: Tensor,  # Required, not Optional
        **auxiliary_losses: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
```

Replace lines 69-76 with:

```python
        legal_mask = policy_legal_mask.reshape(B, -1).to(
            device=pred_flat.device, dtype=torch.bool
        )
```

Remove the `else` branch (line 75-76) and the OR merge (line 74).

**Step 2: Update Phase 1 training to pass legal mask**

In `train_phase1.py`, the `_forward_backward` call currently passes no legal mask. The legal mask must be derived from the target policy (`target_policies > 0`) at the caller level and passed explicitly. Add this to the training step:

```python
legal_mask = target_policies > 0
total_loss, breakdown = self.loss_fn.compute(
    pred_policy, pred_value, target_policies, target_values,
    policy_legal_mask=legal_mask,
)
```

Note: For Phase 1, deriving from targets is correct since Stockfish targets always have nonzero probability on all legal moves (label smoothing ensures this). The key change is making the caller explicit about what it considers legal.

**Step 3: Verify Phase 2 already passes legal mask**

Check `src/denoisr/training/phase2_trainer.py` — it already passes `policy_legal_mask=target_legal_mask`. No change needed.

**Step 4: Update tests**

All tests calling `loss_fn.compute()` must now pass `policy_legal_mask`. Update test fixtures.

**Step 5: Run tests**

```bash
uv run pytest tests/test_training/test_loss.py -x -q
uv run pytest tests/ -x -q
```

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor!: require explicit legal mask in loss computation"
```

---

### Task 5: Require WDL from Stockfish (remove sigmoid fallback)

**Files:**
- Modify: `src/denoisr/data/stockfish_oracle.py:71-83`
- Modify: `tests/test_data/test_stockfish_oracle.py`

**Step 1: Remove sigmoid fallback**

Replace lines 71-83 in `stockfish_oracle.py`:

```python
        wdl = info.get("wdl")
        if wdl is None:
            raise ValueError(
                "Stockfish did not return WDL data. "
                "Requires Stockfish 14+ compiled with WDL support. "
                "Check your Stockfish binary version."
            )
        wdl_white = wdl.white()
        total = wdl_white.wins + wdl_white.draws + wdl_white.losses
        value = ValueTarget(
            win=wdl_white.wins / total,
            draw=wdl_white.draws / total,
            loss=wdl_white.losses / total,
        )
```

**Step 2: Update tests**

If there are tests covering the sigmoid fallback, remove them. Add a test that verifies ValueError is raised when WDL is missing.

**Step 3: Run tests**

```bash
uv run pytest tests/test_data/test_stockfish_oracle.py -x -q
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor!: require WDL from Stockfish, remove sigmoid approximation fallback"
```

---

### Task 6: MCTS require board-state tracking

**Files:**
- Modify: `src/denoisr/training/mcts.py:79-80,175-180,185-208`
- Modify: `tests/test_training/test_mcts.py`

**Step 1: Make transition_fn and legal_mask_fn required**

In MCTS.__init__ (line 74-86), change:

```python
    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        world_model_fn: WorldModelFn,
        config: MCTSConfig,
        legal_mask_fn: LegalMaskFn,
        transition_fn: TransitionFn,
    ) -> None:
```

Remove `| None` from both. Remove default `= None`.

**Step 2: Remove empirical policy path in _simulate**

Replace lines 185-208 with just the board-aware path:

```python
            legal_mask = self._legal_mask_fn(node.board).to(
                device=policy_logits.device, dtype=torch.bool
            )
            masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
            if legal_mask.any():
                policy = torch.softmax(masked_logits.reshape(-1), dim=0).reshape(
                    64, 64
                )
                legal_indices = [
                    (int(f), int(t))
                    for f, t in legal_mask.nonzero(as_tuple=False).tolist()
                ]
            else:
                policy = torch.zeros_like(policy_logits)
                legal_indices = []
```

**Step 3: Require board propagation in expansion**

In the expansion block (lines 168-180), replace the conditional with an assertion:

```python
        if node.state is None and actions:
            parent = path[-2]
            f, t = actions[-1]
            assert parent.state is not None
            state, reward = self._wm(parent.state, f, t)
            node.state = state
            node.reward_from_parent = reward
            assert parent.board is not None
            node.board = self._transition_fn(parent.board, f, t)
```

**Step 4: Update tests**

All MCTS tests must now pass `legal_mask_fn` and `transition_fn` as required args.

**Step 5: Run tests**

```bash
uv run pytest tests/test_training/test_mcts.py -x -q
```

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor!: require board-state tracking in MCTS, remove empirical policy fallback"
```

---

### Task 7: Fail-fast on zero visit distribution in self-play

**Files:**
- Modify: `src/denoisr/training/self_play.py:128-131`
- Modify: `tests/test_training/test_self_play.py`

**Step 1: Replace fallback with error**

Replace lines 128-131 in self_play.py:

```python
            flat_dist = visit_dist.reshape(-1)
            if flat_dist.sum() == 0:
                raise RuntimeError(
                    "MCTS produced zero visit distribution. "
                    "This indicates a bug in MCTS search or legal mask generation. "
                    f"Board FEN: {board.fen()}, move_num: {move_num}"
                )
```

**Step 2: Update tests**

If tests rely on the uniform fallback, update them. Add a test verifying RuntimeError on zero visit dist.

**Step 3: Run tests**

```bash
uv run pytest tests/test_training/test_self_play.py -x -q
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor!: fail-fast on zero MCTS visit distribution in self-play"
```

---

### Task 8: Fail-fast ResourceMonitor

**Files:**
- Modify: `src/denoisr/training/resource_monitor.py` (entire file)
- Modify: `tests/test_training/test_resource_monitor.py`

**Step 1: Remove _try_init_nvml and _get_nvml_handle fallbacks**

Rewrite ResourceMonitor constructor to require GPU monitoring to work if CUDA is available:

```python
import logging
import pynvml

log = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent()
        self._has_cuda = torch.cuda.is_available()
        if self._has_cuda:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self._nvml_handle = None
        # ... sample lists ...
```

**Step 2: Remove try/except in _sample_nvml**

Replace the try/except with direct calls. If pynvml queries fail, let the exception propagate.

```python
    def _sample_nvml(self) -> None:
        import pynvml
        rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        self._gpu_util_samples.append(float(rates.gpu))
        temp = pynvml.nvmlDeviceGetTemperature(
            self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
        )
        self._gpu_temp_samples.append(float(temp))
        power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
        self._gpu_power_samples.append(power / 1000.0)
```

**Step 3: Remove _try_init_nvml and _get_nvml_handle functions**

Delete the standalone functions (lines 16-34).

**Step 4: Update tests**

Update tests to account for the new fail-fast behavior.

**Step 5: Run tests**

```bash
uv run pytest tests/test_training/test_resource_monitor.py -x -q
```

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor!: fail-fast ResourceMonitor, remove silent NVML degradation"
```

---

### Task 9: Fail-fast GrokfastFilter

**Files:**
- Modify: `src/denoisr/training/grokfast.py:36-58`
- Modify: `tests/test_training/test_grokfast.py`

**Step 1: Replace shape/device mismatch recovery with errors**

Replace the apply method (lines 36-58):

```python
    def apply(self, model: nn.Module, *, key_prefix: str = "") -> None:
        prefix = f"{key_prefix}." if key_prefix else ""
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            full_name = f"{prefix}{name}" if prefix else name
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                # Non-finite gradient: skip this parameter, don't corrupt EMA
                self.grads.pop(full_name, None)
                continue
            if full_name not in self.grads:
                self.grads[full_name] = grad.clone()
            else:
                ema = self.grads[full_name]
                if ema.shape != grad.shape:
                    raise ValueError(
                        f"Grokfast EMA shape mismatch for {full_name}: "
                        f"expected {ema.shape}, got {grad.shape}. "
                        "This indicates a model architecture change mid-training."
                    )
                if ema.device != grad.device:
                    raise ValueError(
                        f"Grokfast EMA device mismatch for {full_name}: "
                        f"expected {ema.device}, got {grad.device}."
                    )
                if not torch.isfinite(ema).all():
                    raise RuntimeError(
                        f"Grokfast EMA buffer for {full_name} contains non-finite values. "
                        "Training has diverged."
                    )
                ema.mul_(self.alpha).add_(grad, alpha=1 - self.alpha)
            param.grad.add_(self.grads[full_name], alpha=self.lamb)
```

**Step 2: Update tests**

Add tests for ValueError on shape/device mismatch. Add test for RuntimeError on non-finite EMA.

**Step 3: Run tests**

```bash
uv run pytest tests/test_training/test_grokfast.py -x -q
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor!: fail-fast GrokfastFilter on shape/device mismatch"
```

---

### Task 10: Device detection logging and maybe_compile logging

**Files:**
- Modify: `src/denoisr/scripts/config/__init__.py:370-393`

**Step 1: Add logging to detect_device**

```python
import logging

_log = logging.getLogger(__name__)

def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        _log.warning("Selected device: mps (Apple Silicon GPU detected)")
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        _log.warning("Selected device: cuda (NVIDIA GPU detected, TF32 enabled)")
        return torch.device("cuda")
    _log.warning("Selected device: cpu (no GPU detected)")
    return torch.device("cpu")
```

**Step 2: Add logging to maybe_compile**

```python
def maybe_compile(module: _M, device: torch.device) -> _M:
    """Compile module on CUDA; skip with log on other devices."""
    if device.type != "cuda":
        _log.info("Skipping torch.compile (requires CUDA, got %s)", device.type)
        return module
    return torch.compile(module)  # type: ignore[return-value]
```

**Step 3: Run tests**

```bash
uv run pytest tests/ -x -q
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: log device selection and torch.compile decisions explicitly"
```

---

### Task 11: Fail-fast env file loading

**Files:**
- Modify: `src/denoisr/scripts/runtime.py:34-56`

**Step 1: Add fail-fast for explicit paths**

```python
def load_env_file(path: str | Path | None = None) -> Path:
    """Load KEY=VALUE pairs from an env file into os.environ.

    Existing environment variables are not overwritten.
    If an explicit path is given and doesn't exist, raises FileNotFoundError.
    The default .env path is optional (logs info if missing).
    """
    env_path = Path(path) if path is not None else Path(DEFAULT_ENV_FILE)
    if not env_path.exists():
        if path is not None:
            raise FileNotFoundError(
                f"Env file not found: {env_path}. "
                "Pass an existing file path or omit to use optional default .env"
            )
        return env_path

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path
```

**Step 2: Run tests**

```bash
uv run pytest tests/ -x -q
```

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor!: fail-fast when explicit env file path doesn't exist"
```

---

### Task 12: Fix bare except in oracle cleanup

**Files:**
- Modify: `src/denoisr/scripts/generate_data.py:56-62`

**Step 1: Replace bare except with logged warning**

```python
def _cleanup_oracle() -> None:
    if _oracle is None:
        return
    try:
        _oracle.close()
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "Oracle cleanup failed", exc_info=True
        )
```

Note: atexit handlers should not raise (Python runtime behavior), so we log but don't re-raise. This is best-effort cleanup, not a fallback.

**Step 2: Run tests**

```bash
uv run pytest tests/ -x -q
```

**Step 3: Commit**

```bash
git add -A
git commit -m "fix: log oracle cleanup failures instead of silently swallowing"
```

---

### Task 13: Remove DataLoader worker crash retry

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py` (remove retry try/except around DataLoader)
- Modify: `src/denoisr/scripts/train_phase2.py` (same)

**Step 1: Identify and remove retry blocks**

In train_phase1.py, find the try/except block that catches DataLoader worker crashes and retries with workers=0. Remove the try/except entirely, keeping only the normal iteration path.

In train_phase2.py, find the same pattern and remove it.

**Step 2: Run tests**

```bash
uv run pytest tests/test_scripts/ -x -q
```

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor!: remove DataLoader worker crash retry, fail-fast on worker errors"
```

---

### Task 14: Full integration verification

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -x -q
```

**Step 2: Run linter on all changed files**

```bash
uvx ruff check src/denoisr/ tests/
```

**Step 3: Run type checker**

```bash
uv run --with mypy mypy --strict src/denoisr/training/replay_buffer.py src/denoisr/training/loss.py src/denoisr/training/grokfast.py src/denoisr/training/resource_monitor.py src/denoisr/training/mcts.py src/denoisr/training/self_play.py src/denoisr/data/stockfish_oracle.py src/denoisr/scripts/runtime.py
```

**Step 4: Verify no remaining SimpleReplayBuffer, ModelSWA, or use_onecycle references**

```bash
grep -r "SimpleReplayBuffer\|ModelSWA\|use_onecycle\|onecycle_pct_start\|use_warm_restarts" src/ tests/ --include="*.py"
```

Expected: No matches.

**Step 5: Final commit if any fixups needed**

```bash
git add -A
git commit -m "chore: final cleanup from fail-fast audit"
```
