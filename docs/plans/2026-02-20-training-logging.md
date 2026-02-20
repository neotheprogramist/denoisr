# Training Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TensorBoard-based training logging with gradient norms, GPU metrics, timing, and hyperparameter comparison, writing event files to `logs/<run-name>/`.

**Architecture:** A `TrainingLogger` class wrapping `torch.utils.tensorboard.SummaryWriter`, called from training scripts at natural logging points. Trainers return gradient norms in their breakdown dicts. No new dependencies (tensorboard ships with torch).

**Tech Stack:** PyTorch `torch.utils.tensorboard.SummaryWriter`, TensorBoard (view with `uvx tensorboard --logdir logs/`).

---

### Task 1: Create TrainingLogger with tests

**Files:**
- Create: `src/denoisr/training/logger.py`
- Create: `tests/test_training/test_logger.py`

**Step 1: Write the failing test**

Create `tests/test_training/test_logger.py`:

```python
"""Tests for TrainingLogger TensorBoard integration."""

import pathlib

from denoisr.training.logger import TrainingLogger


class TestTrainingLogger:
    def test_creates_log_directory(self, tmp_path: pathlib.Path) -> None:
        """Logger should create run directory inside log_dir."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test_run")
        logger.close()
        assert (tmp_path / "test_run").is_dir()

    def test_auto_generates_run_name(self, tmp_path: pathlib.Path) -> None:
        """Without run_name, logger should create a timestamped directory."""
        logger = TrainingLogger(log_dir=tmp_path)
        logger.close()
        dirs = list(tmp_path.iterdir())
        assert len(dirs) == 1
        # Timestamped name should match YYYY-MM-DD_HH-MM-SS pattern
        assert len(dirs[0].name) == 19

    def test_log_train_step_writes_scalars(self, tmp_path: pathlib.Path) -> None:
        """log_train_step should write loss scalars without error."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        breakdown = {"policy": 1.5, "value": 0.8, "total": 2.3, "grad_norm": 0.42}
        logger.log_train_step(step=0, loss=2.3, breakdown=breakdown)
        logger.close()

    def test_log_epoch_writes_scalars(self, tmp_path: pathlib.Path) -> None:
        """log_epoch should write accuracy and lr scalars without error."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch(epoch=1, avg_loss=2.0, top1=0.05, top5=0.15, lr=1e-4)
        logger.close()

    def test_log_epoch_timing(self, tmp_path: pathlib.Path) -> None:
        """log_epoch_timing should write timing scalars without error."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_timing(epoch=1, duration_s=42.5, samples_per_sec=1500.0)
        logger.close()

    def test_log_gpu_no_error_on_cpu(self, tmp_path: pathlib.Path) -> None:
        """log_gpu should be a no-op on CPU without raising."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_gpu(step=0)
        logger.close()

    def test_log_diffusion(self, tmp_path: pathlib.Path) -> None:
        """log_diffusion should write diffusion-specific scalars."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_diffusion(epoch=1, avg_loss=0.5, curriculum_steps=25)
        logger.close()

    def test_log_hparams(self, tmp_path: pathlib.Path) -> None:
        """log_hparams should write without error."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        hparams = {"lr": 1e-4, "batch_size": 64, "d_s": 256}
        metrics = {"best_top1": 0.35}
        logger.log_hparams(hparams, metrics)
        logger.close()

    def test_context_manager(self, tmp_path: pathlib.Path) -> None:
        """Logger should support with-statement for automatic cleanup."""
        with TrainingLogger(log_dir=tmp_path, run_name="ctx") as logger:
            logger.log_train_step(step=0, loss=1.0, breakdown={"total": 1.0})
        assert (tmp_path / "ctx").is_dir()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_logger.py -v`
Expected: FAIL (ModuleNotFoundError — `denoisr.training.logger` doesn't exist)

**Step 3: Implement TrainingLogger**

Create `src/denoisr/training/logger.py`:

```python
"""TensorBoard training logger for denoisr."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]


class TrainingLogger:
    """Thin wrapper around SummaryWriter for structured training metrics.

    Usage:
        with TrainingLogger(Path("logs"), run_name="lr1e-4") as logger:
            logger.log_train_step(step, loss, breakdown)
    """

    def __init__(
        self,
        log_dir: Path,
        run_name: str | None = None,
    ) -> None:
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = log_dir / run_name
        self._writer = SummaryWriter(str(self._run_dir))

    def log_train_step(
        self, step: int, loss: float, breakdown: dict[str, float]
    ) -> None:
        """Log per-batch training metrics."""
        self._writer.add_scalar("loss/total", loss, step)
        for key, value in breakdown.items():
            if key == "total":
                continue
            if key == "grad_norm":
                self._writer.add_scalar("gradients/norm", value, step)
            else:
                self._writer.add_scalar(f"loss/{key}", value, step)

    def log_epoch(
        self,
        epoch: int,
        avg_loss: float,
        top1: float,
        top5: float,
        lr: float,
    ) -> None:
        """Log per-epoch summary metrics."""
        self._writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
        self._writer.add_scalar("accuracy/top1", top1, epoch)
        self._writer.add_scalar("accuracy/top5", top5, epoch)
        self._writer.add_scalar("lr", lr, epoch)

    def log_epoch_timing(
        self, epoch: int, duration_s: float, samples_per_sec: float
    ) -> None:
        """Log epoch timing metrics."""
        self._writer.add_scalar("timing/epoch_duration_s", duration_s, epoch)
        self._writer.add_scalar("timing/samples_per_sec", samples_per_sec, epoch)

    def log_gpu(self, step: int) -> None:
        """Log GPU memory metrics. No-op on CPU/MPS."""
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        self._writer.add_scalar("gpu/memory_allocated_mb", allocated, step)
        self._writer.add_scalar("gpu/memory_reserved_mb", reserved, step)

    def log_diffusion(
        self, epoch: int, avg_loss: float, curriculum_steps: int
    ) -> None:
        """Log diffusion training metrics."""
        self._writer.add_scalar("diffusion/loss", avg_loss, epoch)
        self._writer.add_scalar(
            "diffusion/curriculum_steps", curriculum_steps, epoch
        )

    def log_hparams(
        self, hparams: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        """Log hyperparameters for TensorBoard HParams tab."""
        self._writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        """Flush and close the underlying SummaryWriter."""
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_logger.py -v`
Expected: ALL PASS (9 tests)

**Step 5: Run full verification**

Run: `uvx ruff check src/denoisr/training/logger.py tests/test_training/test_logger.py && uv run --with mypy mypy --strict src/denoisr/training/logger.py`
Expected: 0 errors

**Step 6: Commit**

```bash
git add src/denoisr/training/logger.py tests/test_training/test_logger.py
git commit -m "feat: add TrainingLogger with TensorBoard backend"
```

---

### Task 2: Add gradient norm to trainer breakdowns

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py:80-90`
- Modify: `src/denoisr/training/diffusion_trainer.py:42-86`
- Modify: `tests/test_training/test_amp.py`
- Modify: `tests/test_training/test_diffusion_trainer.py`

**Step 1: Write the failing tests**

Add to `tests/test_training/test_amp.py` inside `TestSupervisedTrainerAMP`:

```python
    def test_breakdown_includes_grad_norm(
        self, trainer: SupervisedTrainer
    ) -> None:
        """Training breakdown should include gradient norm."""
        batch = _make_batch(4)
        _, breakdown = trainer.train_step(batch)
        assert "grad_norm" in breakdown
        assert breakdown["grad_norm"] >= 0
```

Add to `tests/test_training/test_diffusion_trainer.py`, a new test (inside the existing test class):

```python
    def test_train_step_returns_breakdown_with_grad_norm(
        self, diff_trainer: DiffusionTrainer, trajectories: torch.Tensor
    ) -> None:
        """DiffusionTrainer.train_step should return loss and breakdown with grad_norm."""
        loss, breakdown = diff_trainer.train_step(trajectories)
        assert isinstance(loss, float)
        assert "grad_norm" in breakdown
        assert breakdown["grad_norm"] >= 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_amp.py::TestSupervisedTrainerAMP::test_breakdown_includes_grad_norm tests/test_training/test_diffusion_trainer.py::TestDiffusionTrainer::test_train_step_returns_breakdown_with_grad_norm -v`
Expected: FAIL (breakdown has no "grad_norm" key; DiffusionTrainer returns float not tuple)

**Step 3: Modify SupervisedTrainer to capture grad norm**

In `src/denoisr/training/supervised_trainer.py`, replace lines 82-86:

```python
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.max_grad_norm,
        )
```

With:

```python
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.max_grad_norm,
        )
```

And replace line 90 (`return total_loss.item(), breakdown`) with:

```python
        breakdown["grad_norm"] = total_norm.item()
        return total_loss.item(), breakdown
```

**Step 4: Modify DiffusionTrainer to return breakdown with grad norm**

In `src/denoisr/training/diffusion_trainer.py`:

Change line 42 return type:
```python
    def train_step(self, trajectories: Tensor) -> tuple[float, dict[str, float]]:
```

Replace lines 78-82:

```python
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.max_grad_norm,
        )
```

With:

```python
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.max_grad_norm,
        )
```

Replace line 86 (`return loss.item()`) with:

```python
        breakdown = {"grad_norm": total_norm.item()}
        return loss.item(), breakdown
```

**Step 5: Update train_phase2.py for new return type**

In `src/denoisr/scripts/train_phase2.py`, change line 165:
```python
            loss = diff_trainer.train_step(batch)
```
To:
```python
            loss, _breakdown = diff_trainer.train_step(batch)
```

**Step 6: Update existing DiffusionTrainer tests for new return type**

In `tests/test_training/test_diffusion_trainer.py`, find any test that calls `diff_trainer.train_step(trajectories)` expecting a bare float and update to unpack the tuple. For example, change:
```python
loss = diff_trainer.train_step(trajectories)
assert isinstance(loss, float)
```
To:
```python
loss, breakdown = diff_trainer.train_step(trajectories)
assert isinstance(loss, float)
```

**Step 7: Run all tests**

Run: `uv run pytest tests/test_training/ -v`
Expected: ALL PASS

**Step 8: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 9: Commit**

```bash
git add src/denoisr/training/supervised_trainer.py src/denoisr/training/diffusion_trainer.py src/denoisr/scripts/train_phase2.py tests/test_training/
git commit -m "feat: capture gradient norm in trainer breakdowns"
```

---

### Task 3: Add --run-name CLI flag and wire logging into train_phase1.py

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py`

**Step 1: Add --run-name flag, import logger, and wire logging calls**

In `src/denoisr/scripts/train_phase1.py`:

Add import:
```python
import time
from denoisr.training.logger import TrainingLogger
```

Add CLI flag after `--output` (around line 90):
```python
    parser.add_argument("--run-name", type=str, default=None, help="TensorBoard run name (default: timestamp)")
```

After `trainer = SupervisedTrainer(...)` (after line 144), create the logger:
```python
    logger = TrainingLogger(Path("logs"), run_name=args.run_name)
```

Log hyperparameters once (before training loop):
```python
    logger.log_hparams(
        {"lr": args.lr, "batch_size": bs, "d_s": cfg.d_s,
         "num_heads": cfg.num_heads, "num_layers": cfg.num_layers,
         "ffn_dim": cfg.ffn_dim, "num_planes": cfg.num_planes,
         "gradient_checkpointing": cfg.gradient_checkpointing},
        {"best_top1": 0.0},
    )
```

Inside the training loop, add a global step counter and per-step logging. Track epoch timing. The loop becomes:

```python
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.monotonic()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False,
            smoothing=0.3,
        )
        for boards_batch, policies_batch, values_batch in pbar:
            loss, breakdown = trainer.train_step_tensors(
                boards_batch, policies_batch, values_batch
            )
            logger.log_train_step(global_step, loss, breakdown)
            if global_step % 100 == 0:
                logger.log_gpu(global_step)
            global_step += 1
            epoch_loss += loss
            num_batches += 1
            pbar.set_postfix(
                loss=f"{loss:.4f}",
                policy=f"{breakdown['policy']:.4f}",
                value=f"{breakdown['value']:.4f}",
            )
        pbar.close()
        trainer.scheduler_step()

        epoch_duration = time.monotonic() - epoch_start
        samples_per_sec = len(train) / epoch_duration
        avg_loss = epoch_loss / max(num_batches, 1)
        top1, top5 = measure_accuracy(trainer, holdout, device)
        current_lr = trainer.optimizer.param_groups[0]["lr"]

        logger.log_epoch(epoch, avg_loss, top1, top5, current_lr)
        logger.log_epoch_timing(epoch, epoch_duration, samples_per_sec)

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"avg_loss={avg_loss:.4f} top1={top1:.1%} top5={top5:.1%}"
        )

        if top1 > best_acc:
            best_acc = top1
            save_checkpoint(...)

        if top1 > 0.30:
            print(f"PHASE 1 GATE PASSED: top-1 accuracy {top1:.1%} > 30%")
            print("Ready for Phase 2.")
            break

    logger.close()
    print(f"Best top-1 accuracy: {best_acc:.1%}")
```

**Step 2: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 3: Commit**

```bash
git add src/denoisr/scripts/train_phase1.py
git commit -m "feat: add TensorBoard logging to Phase 1 training"
```

---

### Task 4: Wire logging into train_phase2.py

**Files:**
- Modify: `src/denoisr/scripts/train_phase2.py`

**Step 1: Add --run-name flag, import logger, and wire logging calls**

In `src/denoisr/scripts/train_phase2.py`:

Add import:
```python
import time
from denoisr.training.logger import TrainingLogger
```

Add CLI flag after `--output` (around line 94):
```python
    parser.add_argument("--run-name", type=str, default=None, help="TensorBoard run name (default: timestamp)")
```

After `diff_trainer = DiffusionTrainer(...)` (after line 132), create the logger:
```python
    logger = TrainingLogger(Path("logs"), run_name=args.run_name)
```

Wire logging into training loop:

```python
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.monotonic()

        pbar = tqdm(...)
        for (batch,) in pbar:
            batch = batch.to(device, non_blocking=True)
            loss, breakdown = diff_trainer.train_step(batch)
            logger.log_train_step(global_step, loss, breakdown)
            if global_step % 100 == 0:
                logger.log_gpu(global_step)
            global_step += 1
            epoch_loss += loss
            num_batches += 1
            pbar.set_postfix(loss=f"{loss:.4f}")
        pbar.close()

        diff_trainer.advance_curriculum()
        epoch_duration = time.monotonic() - epoch_start
        num_samples = len(dataset)
        avg_loss = epoch_loss / max(num_batches, 1)

        logger.log_diffusion(epoch, avg_loss, diff_trainer._current_max_steps)
        logger.log_epoch_timing(epoch, epoch_duration, num_samples / epoch_duration)

        print(...)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(...)

    logger.close()
```

**Step 2: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 3: Commit**

```bash
git add src/denoisr/scripts/train_phase2.py
git commit -m "feat: add TensorBoard logging to Phase 2 training"
```

---

### Task 5: Add logs/ to .gitignore and verify end-to-end

**Files:**
- Modify: `.gitignore`

**Step 1: Add logs/ to .gitignore**

Append to `.gitignore`:
```
logs/
```

**Step 2: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 3: Verify CLI help strings**

Run:
```bash
uv run denoisr-train-phase1 --help
uv run denoisr-train-phase2 --help
```
Expected: Both show `--run-name` flag.

**Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add logs/ to .gitignore"
```
