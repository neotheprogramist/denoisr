# Phase 1 Training Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement all 12 optimization changes from `docs/plans/phase1_changes_summary.md` — numerical stability, bug fixes, observability, and performance tuning for Phase 1 supervised training.

**Architecture:** Three logical commits following the dependency chain: (1) value head returns raw logits + all callers updated, (2) warmup LR fix + overflow detection + logger hardening, (3) hyperparameter defaults + batched accuracy + per-group LR logging. Each commit leaves all tests passing.

**Tech Stack:** PyTorch, torch.amp (AMP/GradScaler), F.log_softmax, CosineAnnealingLR

---

### Task 1: Update value head to return raw logits

**Files:**
- Modify: `src/denoisr/nn/value_head.py:19-23`

**Step 1: Modify `forward()` to return raw logits with AMP float32 override, and add `infer()` method**

Replace the entire `forward` method and add `infer`:

```python
# value_head.py -- new forward() + infer()

def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    pooled = self.norm(x.mean(dim=1))
    with torch.amp.autocast("cuda", enabled=False):
        wdl_logits = self.wdl_linear(pooled.float())
    ply = torch.nn.functional.softplus(self.ply_linear(pooled))
    return wdl_logits, ply

def infer(self, x: Tensor) -> tuple[Tensor, Tensor]:
    """Return WDL probabilities + ply for inference callers."""
    wdl_logits, ply = self.forward(x)
    return torch.softmax(wdl_logits, dim=-1), ply
```

Also update the docstring: replace "produces WDL probabilities (via softmax)" with "produces WDL logits (softmax applied only via `infer()` for inference)".

**Step 2: Run value head tests to verify they fail (expected -- tests still call `head()` and assert probabilities)**

Run: `uv run pytest tests/test_nn/test_value_head.py -v`
Expected: `test_wdl_sums_to_one` and `test_wdl_in_zero_one` FAIL (logits don't sum to 1 / aren't in [0,1])

---

### Task 2: Update loss function to use F.log_softmax

**Files:**
- Modify: `src/denoisr/training/loss.py:69-70`

**Step 1: Replace clamped-log with F.log_softmax**

Change lines 69-70 from:
```python
pred_log = torch.log(pred_value.clamp(min=1e-8))
value_loss = -(target_value * pred_log).sum(dim=-1).mean()
```
To:
```python
pred_log = F.log_softmax(pred_value, dim=-1)
value_loss = -(target_value * pred_log).sum(dim=-1).mean()
```

**Step 2: Run loss tests to verify nothing breaks**

Run: `uv run pytest tests/test_training/test_loss.py -v`
Expected: PASS (loss tests use the compute interface, which now receives logits from the value head)

---

### Task 3: Update inference engines to use `.infer()`

**Files:**
- Modify: `src/denoisr/inference/engine.py:79`
- Modify: `src/denoisr/inference/diffusion_engine.py:78`

**Step 1: Change both engines from `self._value_head(features)` to `self._value_head.infer(features)`**

In `engine.py` line 79:
```python
# Old:
wdl, _ = self._value_head(features)
# New:
wdl, _ = self._value_head.infer(features)
```

In `diffusion_engine.py` line 78:
```python
# Old:
wdl, _ = self._value_head(features)
# New:
wdl, _ = self._value_head.infer(features)
```

**Step 2: Run inference tests**

Run: `uv run pytest tests/test_inference/ -v`
Expected: PASS

---

### Task 4: Update Phase 3 MCTS closure to apply softmax

**Files:**
- Modify: `src/denoisr/scripts/train_phase3.py:89-93`

**Step 1: Add softmax to WDL logits in `policy_value_fn`**

Change lines 89-93 from:
```python
@torch.no_grad()
def policy_value_fn(latent: Tensor) -> tuple[Tensor, Tensor]:
    features = backbone(latent.unsqueeze(0))
    policy = policy_head(features).squeeze(0)
    wdl, _ = value_head(features)
    return policy, wdl.squeeze(0)
```
To:
```python
@torch.no_grad()
def policy_value_fn(latent: Tensor) -> tuple[Tensor, Tensor]:
    features = backbone(latent.unsqueeze(0))
    policy = policy_head(features).squeeze(0)
    wdl_logits, _ = value_head(features)
    wdl = torch.softmax(wdl_logits, dim=-1)
    return policy, wdl.squeeze(0)
```

**Step 2: Run phase 3 / MCTS tests**

Run: `uv run pytest tests/test_training/test_mcts.py tests/test_training/test_self_play.py -v`
Expected: PASS

---

### Task 5: Update value head tests

**Files:**
- Modify: `tests/test_nn/test_value_head.py`

**Step 1: Rewrite test file**

```python
import pytest
import torch

from denoisr.nn.value_head import ChessValueHead

from conftest import SMALL_D_S


class TestChessValueHead:
    @pytest.fixture
    def head(self, device: torch.device) -> ChessValueHead:
        return ChessValueHead(d_s=SMALL_D_S).to(device)

    def test_forward_output_shape(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl_logits, ply = head(small_latent)
        assert wdl_logits.shape == (2, 3)

    def test_ply_output_shape(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        _, ply = head(small_latent)
        assert ply.shape == (2, 1)

    def test_forward_returns_finite_logits(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl_logits, _ = head(small_latent)
        assert torch.isfinite(wdl_logits).all()

    def test_forward_logits_not_constrained_to_unit(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        """Logits from forward() are raw -- not necessarily in [0, 1]."""
        wdl_logits, _ = head(small_latent)
        # Logits can be negative or > 1; just verify they're not all in [0,1]
        # (with random weights, it's near-certain some logit is outside [0,1])
        assert not ((wdl_logits >= 0).all() and (wdl_logits <= 1).all())

    def test_infer_wdl_sums_to_one(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, _ = head.infer(small_latent)
        sums = wdl.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_infer_wdl_in_zero_one(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, _ = head.infer(small_latent)
        assert (wdl >= 0).all()
        assert (wdl <= 1).all()

    def test_ply_non_negative(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        _, ply = head(small_latent)
        assert (ply >= 0).all()

    def test_gradient_flows(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl_logits, ply = head(small_latent)
        (wdl_logits.sum() + ply.sum()).backward()
        for p in head.parameters():
            assert p.grad is not None
```

**Step 2: Run all tests to verify commit 1 is green**

Run: `uv run pytest tests/ -x -q`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/denoisr/nn/value_head.py src/denoisr/training/loss.py \
  src/denoisr/inference/engine.py src/denoisr/inference/diffusion_engine.py \
  src/denoisr/scripts/train_phase3.py tests/test_nn/test_value_head.py
git commit -m "fix: return raw logits from value head for numerical stability

- forward() returns WDL logits (removed softmax), with float32 AMP override
- Added infer() method for inference callers that need probabilities
- Loss function uses F.log_softmax (numerically stable log-sum-exp)
- Inference engines call .infer() instead of .__call__()
- Phase 3 MCTS closure applies softmax to WDL logits
- Updated value head tests for logits vs probabilities API

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Fix warmup LR initialization

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py:59-64`

**Step 1: After recording `_base_lrs`, set initial LRs to `base_lr / warmup_epochs`**

After line 60 (`self._base_lrs = ...`), before the `_scheduler` creation (line 61), add:
```python
# Start at 1/N of peak LR; warmup will ramp up from here
for g, base_lr in zip(param_groups, self._base_lrs):
    g["lr"] = base_lr / max(self._warmup_epochs, 1)
```

**Step 2: Run tests to see which break**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v`
Expected: `test_scheduler_reduces_lr` FAILS (initial LRs are now lower, so after 5 steps they may be higher -- not lower)

---

### Task 7: Add AMP overflow detection

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py:1-6,98-106`

**Step 1: Add `import math` at top of file**

Add `import math` to the imports.

**Step 2: Add overflow flag to breakdown**

Replace lines 105-106:
```python
breakdown["grad_norm"] = total_norm.item()
return total_loss.item(), breakdown
```
With:
```python
grad_norm = total_norm.item()
breakdown["grad_norm"] = grad_norm
breakdown["overflow"] = not math.isfinite(grad_norm)
return total_loss.item(), breakdown
```

**Step 3: Run trainer tests**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v`
Expected: Same as before (scheduler test still fails from Task 6)

---

### Task 8: Harden logger against inf grad norms

**Files:**
- Modify: `src/denoisr/training/logger.py:11,46-57,143-166`

**Step 1: Add `import math` to logger.py imports**

Add `import math` after the existing `import logging` line.

**Step 2: In `log_train_step()`, skip the `"overflow"` key**

Change line 52:
```python
if key == "total":
    continue
```
To:
```python
if key in ("total", "overflow"):
    continue
```

**Step 3: In `log_training_dynamics()`, filter inf/nan from grad norms**

Change lines 155-156:
```python
gn_avg = mean(grad_norms)
gn_peak = max(grad_norms)
```
To:
```python
finite_norms = [n for n in grad_norms if math.isfinite(n)]
if not finite_norms:
    return
gn_avg = mean(finite_norms)
gn_peak = max(finite_norms)
```

**Step 4: Run logger tests**

Run: `uv run pytest tests/test_training/test_logger.py -v`
Expected: PASS

---

### Task 9: Add overflow handling to Phase 1 training loop

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py:288-330`

**Step 1: Add overflow counter initialization**

After line 289 (`step_grad_norms: list[float] = []`), add:
```python
overflow_count = 0
```

**Step 2: Filter overflow steps from grad norms**

Replace line 317:
```python
step_grad_norms.append(breakdown.get("grad_norm", 0.0))
```
With:
```python
if breakdown.get("overflow", False):
    overflow_count += 1
else:
    step_grad_norms.append(breakdown.get("grad_norm", 0.0))
```

**Step 3: Add overflow count to epoch summary**

In the summary dict (around line 376), after `"data_pct"`, add:
```python
"overflows": str(overflow_count),
```

**Step 4: Run tests**

Run: `uv run pytest tests/ -x -q --ignore=tests/test_training/test_supervised_trainer.py`
Expected: PASS (skipping the broken scheduler test)

---

### Task 10: Fix scheduler test

**Files:**
- Modify: `tests/test_training/test_supervised_trainer.py:111-121`

**Step 1: Rewrite `test_scheduler_reduces_lr`**

Replace:
```python
def test_scheduler_reduces_lr(self, trainer: SupervisedTrainer) -> None:
    """After stepping the scheduler, learning rates should decrease."""
    initial_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
    batch = _make_batch(8)
    # Simulate several epochs
    for _ in range(5):
        trainer.train_step(batch)
        trainer.scheduler_step()
    current_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
    # At least one group should have a lower LR
    assert any(c < i for c, i in zip(current_lrs, initial_lrs))
```
With:
```python
def test_scheduler_reduces_lr(self, trainer: SupervisedTrainer) -> None:
    """After warmup + cosine decay, LRs should be below peak."""
    peak_lrs = trainer._base_lrs
    batch = _make_batch(8)
    # Run through warmup + 5 cosine decay steps
    for _ in range(trainer._warmup_epochs + 5):
        trainer.train_step(batch)
        trainer.scheduler_step()
    current_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
    # After cosine decay, LRs should be below peak
    assert all(c < p for c, p in zip(current_lrs, peak_lrs))
```

**Step 2: Run all tests to verify commit 2 is green**

Run: `uv run pytest tests/ -x -q`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/denoisr/training/supervised_trainer.py \
  src/denoisr/training/logger.py \
  src/denoisr/scripts/train_phase1.py \
  tests/test_training/test_supervised_trainer.py
git commit -m "fix: warmup LR initialization, overflow detection, and grad norm filtering

- Warmup starts at 1/N of peak LR instead of full peak
- Added overflow detection flag to training breakdown dict
- Logger filters inf/nan from grad norm stats and skips overflow key
- Training loop excludes overflow steps from grad norm stats, reports overflow count
- Updated scheduler test to compare against _base_lrs after full warmup + decay

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 11: Update hyperparameter defaults

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py:98,100` (argparse defaults)
- Modify: `src/denoisr/scripts/config.py:98,109,119,383,391,399` (dataclass + CLI defaults)

**Step 1: Change batch-size and lr defaults in train_phase1.py**

Line 98: change `default=64` to `default=1024`
Line 100: change `default=1e-4` to `default=3e-4`

**Step 2: Change training config dataclass defaults in config.py**

Line 98: `max_grad_norm: float = 1.0` to `max_grad_norm: float = 5.0`
Line 109: `encoder_lr_multiplier: float = 0.3` to `encoder_lr_multiplier: float = 1.0`
Line 119: `warmup_epochs: int = 3` to `warmup_epochs: int = 5`

**Step 3: Change CLI defaults in config.py `add_training_args()`**

Line 383: `default=1.0` to `default=5.0` (max-grad-norm)
Line 391: `default=0.3` to `default=1.0` (encoder-lr-multiplier)
Line 399: `default=3` to `default=5` (warmup-epochs)

Update help strings to reflect new defaults.

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v`
Expected: PASS (trainer fixture uses explicit `lr=1e-3`, doesn't rely on config defaults)

---

### Task 12: Batch accuracy measurement

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py:48-82`

**Step 1: Rewrite `measure_accuracy()` to use batched inference**

Replace the entire function:
```python
def measure_accuracy(
    trainer: SupervisedTrainer,
    examples: list[TrainingExample],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, float]:
    trainer.encoder.eval()
    trainer.backbone.eval()
    trainer.policy_head.eval()

    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    autocast_enabled = device.type == "cuda"

    correct_1 = 0
    correct_5 = 0
    total = len(examples)

    with torch.no_grad(), autocast(autocast_device, enabled=autocast_enabled):
        for i in range(0, total, batch_size):
            batch = examples[i : i + batch_size]
            boards = torch.stack([ex.board.data for ex in batch]).to(device)
            targets = torch.stack([ex.policy.data for ex in batch])

            latent = trainer.encoder(boards)
            features = trainer.backbone(latent)
            logits = trainer.policy_head(features)

            pred_flat = logits.reshape(len(batch), -1)
            target_flat = targets.reshape(len(batch), -1)
            target_idx = target_flat.argmax(dim=-1)  # (B,)

            top5 = pred_flat.topk(5, dim=-1).indices  # (B, 5)
            correct_1 += (top5[:, 0] == target_idx).sum().item()
            correct_5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()

    return correct_1 / max(total, 1), correct_5 / max(total, 1)
```

**Step 2: Run tests**

Run: `uv run pytest tests/ -x -q`
Expected: PASS

---

### Task 13: Add per-group LR logging

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py:337,363-376`

**Step 1: Capture head LR**

After line 337 (`current_lr = trainer.optimizer.param_groups[0]["lr"]`), add:
```python
head_lr = trainer.optimizer.param_groups[2]["lr"]
```

**Step 2: Add TensorBoard scalars for both LR groups**

After the `logger.log_epoch(...)` call (line 339), add:
```python
logger._writer.add_scalar("lr/encoder", current_lr, epoch)
logger._writer.add_scalar("lr/head", head_lr, epoch)
```

**Step 3: Update epoch summary**

In the summary dict, change:
```python
"lr": f"{current_lr:.2e}",
```
To:
```python
"lr_enc": f"{current_lr:.2e}",
"lr_head": f"{head_lr:.2e}",
```

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/denoisr/scripts/train_phase1.py src/denoisr/scripts/config.py
git commit -m "perf: optimize phase 1 defaults, batch accuracy evaluation, per-group LR logging

- batch-size 64->1024, lr 1e-4->3e-4
- max-grad-norm 1.0->5.0, encoder-lr-multiplier 0.3->1.0, warmup-epochs 3->5
- measure_accuracy() processes holdout in batches of 256
- Log both encoder LR and head LR in epoch summary and TensorBoard

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 14: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All tests PASS

**Step 2: Run linter**

Run: `uvx ruff check src/ tests/`
Expected: No errors

**Step 3: Run type checker**

Run: `uv run --with mypy mypy --strict src/denoisr/nn/value_head.py src/denoisr/training/loss.py src/denoisr/training/supervised_trainer.py src/denoisr/training/logger.py`
Expected: No errors (or only pre-existing ones)
