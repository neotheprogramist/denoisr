# Resource & Diagnostics Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-epoch CPU/GPU utilization, RAM usage, GPU thermals/power, training dynamics, and data pipeline efficiency metrics to the training logger.

**Architecture:** A new `ResourceMonitor` class samples system metrics every 100 steps and reports avg/peak per epoch. Three new `TrainingLogger` methods write these to TensorBoard + text logs. Phase 1 and Phase 2 training scripts integrate the monitor with minimal loop changes.

**Tech Stack:** `psutil` (CPU/RAM), `pynvml` via `nvidia-ml-py3` (GPU util/temp/power), existing `torch.cuda` (GPU memory), `statistics.mean`/`max` for aggregation.

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add psutil and nvidia-ml-py3 as runtime dependencies**

Run:
```bash
uv add psutil 'nvidia-ml-py3>=7.352.0'
```

psutil is needed on all platforms. nvidia-ml-py3 provides pynvml for GPU monitoring (gracefully unused on non-NVIDIA systems).

**Step 2: Verify lock file updated**

Run: `uv lock --check`
Expected: no errors

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add psutil and nvidia-ml-py3 for resource monitoring"
```

---

### Task 2: Create ResourceMonitor with CPU/RAM sampling

**Files:**
- Create: `src/denoisr/training/resource_monitor.py`
- Test: `tests/test_training/test_resource_monitor.py`

**Step 1: Write the failing tests**

Create `tests/test_training/test_resource_monitor.py`:

```python
"""Tests for ResourceMonitor system metrics collection."""

from denoisr.training.resource_monitor import ResourceMonitor


class TestResourceMonitorCPURAM:
    def test_summarize_empty_returns_empty(self) -> None:
        """Summarize with no samples should return empty dict."""
        monitor = ResourceMonitor()
        result = monitor.summarize()
        assert result == {}

    def test_sample_and_summarize_has_cpu_keys(self) -> None:
        """After sampling, summary should contain CPU avg/peak keys."""
        monitor = ResourceMonitor()
        monitor.sample()
        monitor.sample()
        result = monitor.summarize()
        assert "cpu_percent_avg" in result
        assert "cpu_percent_peak" in result
        assert result["cpu_percent_avg"] >= 0.0
        assert result["cpu_percent_peak"] >= result["cpu_percent_avg"]

    def test_sample_and_summarize_has_ram_keys(self) -> None:
        """After sampling, summary should contain RAM avg/peak keys."""
        monitor = ResourceMonitor()
        monitor.sample()
        result = monitor.summarize()
        assert "ram_mb_avg" in result
        assert "ram_mb_peak" in result
        assert result["ram_mb_avg"] > 0.0
        assert result["ram_mb_peak"] >= result["ram_mb_avg"]

    def test_reset_clears_samples(self) -> None:
        """After reset, summarize should return empty dict."""
        monitor = ResourceMonitor()
        monitor.sample()
        assert monitor.summarize() != {}
        monitor.reset()
        assert monitor.summarize() == {}

    def test_peak_is_max_of_samples(self) -> None:
        """Peak should be the max across samples, not just the last."""
        monitor = ResourceMonitor()
        # Take several samples — peak should be >= avg
        for _ in range(5):
            monitor.sample()
        result = monitor.summarize()
        assert result["cpu_percent_peak"] >= result["cpu_percent_avg"]
        assert result["ram_mb_peak"] >= result["ram_mb_avg"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_resource_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'denoisr.training.resource_monitor'`

**Step 3: Write minimal implementation**

Create `src/denoisr/training/resource_monitor.py`:

```python
"""System resource monitor for training metrics.

Collects CPU/RAM/GPU samples during training and reports
per-epoch averages and peaks. Gracefully degrades when
subsystems are unavailable (no CUDA, no pynvml).
"""

from __future__ import annotations

from statistics import mean

import psutil
import torch


def _try_init_nvml() -> bool:
    """Attempt to initialize NVML. Returns True on success."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def _get_nvml_handle() -> object | None:
    """Get NVML handle for device 0, or None."""
    try:
        import pynvml

        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


class ResourceMonitor:
    """Samples system resources and computes per-epoch avg/peak.

    Usage:
        monitor = ResourceMonitor()

        for epoch in range(epochs):
            monitor.reset()
            for step in range(steps):
                if step % 100 == 0:
                    monitor.sample()
            metrics = monitor.summarize()
            # metrics = {"cpu_percent_avg": 45.2, "cpu_percent_peak": 98.1, ...}
    """

    def __init__(self) -> None:
        self._process = psutil.Process()
        # Prime cpu_percent (first call always returns 0.0)
        self._process.cpu_percent()

        self._has_cuda = torch.cuda.is_available()
        self._has_nvml = self._has_cuda and _try_init_nvml()
        self._nvml_handle = _get_nvml_handle() if self._has_nvml else None

        self._cpu_samples: list[float] = []
        self._ram_samples: list[float] = []
        self._gpu_util_samples: list[float] = []
        self._gpu_mem_samples: list[float] = []
        self._gpu_temp_samples: list[float] = []
        self._gpu_power_samples: list[float] = []

    def reset(self) -> None:
        """Clear all accumulated samples for a new epoch."""
        self._cpu_samples.clear()
        self._ram_samples.clear()
        self._gpu_util_samples.clear()
        self._gpu_mem_samples.clear()
        self._gpu_temp_samples.clear()
        self._gpu_power_samples.clear()

    def sample(self) -> None:
        """Take a snapshot of all available resource metrics."""
        self._cpu_samples.append(self._process.cpu_percent())
        self._ram_samples.append(
            self._process.memory_info().rss / (1024 * 1024)
        )

        if self._has_cuda:
            self._gpu_mem_samples.append(
                torch.cuda.memory_allocated() / (1024 * 1024)
            )

        if self._has_nvml and self._nvml_handle is not None:
            self._sample_nvml()

    def _sample_nvml(self) -> None:
        """Sample GPU utilization, temperature, and power via NVML."""
        try:
            import pynvml

            rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            self._gpu_util_samples.append(float(rates.gpu))

            temp = pynvml.nvmlDeviceGetTemperature(
                self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            self._gpu_temp_samples.append(float(temp))

            power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
            self._gpu_power_samples.append(power / 1000.0)  # mW -> W
        except Exception:
            pass

    def summarize(self) -> dict[str, float]:
        """Compute avg/peak for all collected metrics.

        Returns empty dict if no samples were collected.
        """
        if not self._cpu_samples:
            return {}

        result: dict[str, float] = {
            "cpu_percent_avg": mean(self._cpu_samples),
            "cpu_percent_peak": max(self._cpu_samples),
            "ram_mb_avg": mean(self._ram_samples),
            "ram_mb_peak": max(self._ram_samples),
        }

        if self._gpu_mem_samples:
            result["gpu_mem_mb_avg"] = mean(self._gpu_mem_samples)
            result["gpu_mem_mb_peak"] = max(self._gpu_mem_samples)

        if self._gpu_util_samples:
            result["gpu_util_avg"] = mean(self._gpu_util_samples)
            result["gpu_util_peak"] = max(self._gpu_util_samples)

        if self._gpu_temp_samples:
            result["gpu_temp_avg"] = mean(self._gpu_temp_samples)
            result["gpu_temp_peak"] = max(self._gpu_temp_samples)

        if self._gpu_power_samples:
            result["gpu_power_avg"] = mean(self._gpu_power_samples)
            result["gpu_power_peak"] = max(self._gpu_power_samples)

        return result
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_resource_monitor.py -v`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/resource_monitor.py tests/test_training/test_resource_monitor.py
git commit -m "feat: add ResourceMonitor with CPU/RAM/GPU sampling"
```

---

### Task 3: Add new logger methods

**Files:**
- Modify: `src/denoisr/training/logger.py`
- Modify: `tests/test_training/test_logger.py`

**Step 1: Write the failing tests**

Append to `tests/test_training/test_logger.py`:

```python
    def test_log_resource_metrics_writes_scalars_and_text(
        self, tmp_path: pathlib.Path
    ) -> None:
        """log_resource_metrics should write resource metrics to TensorBoard and text."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        metrics = {
            "cpu_percent_avg": 45.2,
            "cpu_percent_peak": 98.1,
            "ram_mb_avg": 2341.0,
            "ram_mb_peak": 2567.0,
        }
        logger.log_resource_metrics(epoch=0, metrics=metrics)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "cpu_avg=45.2%" in text
        assert "ram_peak=2567mb" in text

    def test_log_training_dynamics(self, tmp_path: pathlib.Path) -> None:
        """log_training_dynamics should compute and write dynamics stats."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        losses = [1.0, 2.0, 3.0, 4.0, 5.0]
        grad_norms = [0.1, 0.5, 0.3, 0.8, 0.2]
        logger.log_training_dynamics(epoch=0, losses=losses, grad_norms=grad_norms)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "grad_norm_avg=" in text
        assert "grad_norm_peak=0.800" in text
        assert "loss_std=" in text

    def test_log_pipeline_timing(self, tmp_path: pathlib.Path) -> None:
        """log_pipeline_timing should write pipeline efficiency metrics."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_pipeline_timing(epoch=0, data_time=2.0, compute_time=8.0)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "data_wait_frac=0.20" in text
        assert "compute_frac=0.80" in text

    def test_log_pipeline_timing_zero_total(self, tmp_path: pathlib.Path) -> None:
        """log_pipeline_timing should handle zero total time gracefully."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_pipeline_timing(epoch=0, data_time=0.0, compute_time=0.0)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "data_wait_frac=0.00" in text
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_logger.py -v -k "resource or dynamics or pipeline"`
Expected: FAIL — `AttributeError: 'TrainingLogger' object has no attribute 'log_resource_metrics'`

**Step 3: Add methods to TrainingLogger**

Add these methods to `src/denoisr/training/logger.py` in the `TrainingLogger` class, before `_write_text`:

```python
    def log_resource_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log per-epoch resource utilization avg/peak."""
        for key, value in metrics.items():
            self._writer.add_scalar(f"resources/{key}", value, epoch)

        # Text log — CPU/RAM line
        parts: list[str] = [f"epoch={epoch}"]
        if "cpu_percent_avg" in metrics:
            parts.append(f"cpu_avg={metrics['cpu_percent_avg']:.1f}%")
            parts.append(f"cpu_peak={metrics['cpu_percent_peak']:.1f}%")
        if "ram_mb_avg" in metrics:
            parts.append(f"ram_avg={metrics['ram_mb_avg']:.0f}mb")
            parts.append(f"ram_peak={metrics['ram_mb_peak']:.0f}mb")
        self._write_text("\t".join(parts))

        # GPU line (if present)
        gpu_parts: list[str] = [f"epoch={epoch}"]
        has_gpu = False
        if "gpu_util_avg" in metrics:
            gpu_parts.append(f"gpu_util_avg={metrics['gpu_util_avg']:.1f}%")
            gpu_parts.append(f"gpu_util_peak={metrics['gpu_util_peak']:.1f}%")
            has_gpu = True
        if "gpu_mem_mb_avg" in metrics:
            gpu_parts.append(f"gpu_mem_avg={metrics['gpu_mem_mb_avg']:.0f}mb")
            gpu_parts.append(f"gpu_mem_peak={metrics['gpu_mem_mb_peak']:.0f}mb")
            has_gpu = True
        if "gpu_temp_avg" in metrics:
            gpu_parts.append(f"gpu_temp_avg={metrics['gpu_temp_avg']:.0f}C")
            has_gpu = True
        if "gpu_power_avg" in metrics:
            gpu_parts.append(f"gpu_power_avg={metrics['gpu_power_avg']:.0f}W")
            has_gpu = True
        if has_gpu:
            self._write_text("\t".join(gpu_parts))

    def log_training_dynamics(
        self,
        epoch: int,
        losses: list[float],
        grad_norms: list[float],
    ) -> None:
        """Log per-epoch training dynamics (grad norm stats, loss std dev)."""
        from statistics import mean, stdev

        if not grad_norms:
            return

        gn_avg = mean(grad_norms)
        gn_peak = max(grad_norms)
        self._writer.add_scalar("dynamics/grad_norm_avg", gn_avg, epoch)
        self._writer.add_scalar("dynamics/grad_norm_peak", gn_peak, epoch)

        loss_sd = stdev(losses) if len(losses) > 1 else 0.0
        self._writer.add_scalar("dynamics/loss_stddev", loss_sd, epoch)

        self._write_text(
            f"epoch={epoch}\tgrad_norm_avg={gn_avg:.3f}\t"
            f"grad_norm_peak={gn_peak:.3f}\tloss_std={loss_sd:.4f}"
        )

    def log_pipeline_timing(
        self, epoch: int, data_time: float, compute_time: float
    ) -> None:
        """Log data pipeline vs. compute time split."""
        total = data_time + compute_time
        data_frac = data_time / total if total > 0 else 0.0
        compute_frac = compute_time / total if total > 0 else 0.0

        self._writer.add_scalar("pipeline/data_wait_s", data_time, epoch)
        self._writer.add_scalar("pipeline/data_wait_frac", data_frac, epoch)
        self._writer.add_scalar("pipeline/compute_frac", compute_frac, epoch)

        self._write_text(
            f"epoch={epoch}\tdata_wait_s={data_time:.2f}\t"
            f"data_wait_frac={data_frac:.2f}\tcompute_frac={compute_frac:.2f}"
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_logger.py -v`
Expected: all tests PASS (both old and new)

**Step 5: Commit**

```bash
git add src/denoisr/training/logger.py tests/test_training/test_logger.py
git commit -m "feat: add resource, dynamics, and pipeline logging methods to TrainingLogger"
```

---

### Task 4: Integrate ResourceMonitor into Phase 1 training

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py`

**Step 1: Add imports**

At the top of `train_phase1.py`, add:

```python
from denoisr.training.resource_monitor import ResourceMonitor
```

**Step 2: Create monitor before the training loop**

After `global_step = 0` (line 218), add:

```python
        monitor = ResourceMonitor()
```

**Step 3: Instrument the epoch loop**

Replace the epoch training loop (lines 220-256) with the instrumented version. The key changes are:
1. Add `monitor.reset()` and accumulator lists at epoch start
2. Wrap DataLoader iteration with timing
3. Wrap `train_step_tensors` with timing
4. Accumulate losses and grad norms
5. Call `monitor.sample()` alongside `logger.log_gpu()`
6. Log new metrics at epoch end

The modified loop:

```python
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.monotonic()
            monitor.reset()
            step_losses: list[float] = []
            step_grad_norms: list[float] = []
            data_time = 0.0
            compute_time = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
                smoothing=0.3,
            )
            data_start = time.monotonic()
            for boards_batch, policies_batch, values_batch in pbar:
                data_time += time.monotonic() - data_start

                compute_start = time.monotonic()
                loss, breakdown = trainer.train_step_tensors(
                    boards_batch, policies_batch, values_batch
                )
                compute_time += time.monotonic() - compute_start

                logger.log_train_step(global_step, loss, breakdown)
                step_losses.append(loss)
                step_grad_norms.append(breakdown.get("grad_norm", 0.0))
                if global_step % 100 == 0:
                    logger.log_gpu(global_step)
                    monitor.sample()
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    policy=f"{breakdown['policy']:.4f}",
                    value=f"{breakdown['value']:.4f}",
                )
                data_start = time.monotonic()
            pbar.close()
            trainer.scheduler_step()

            epoch_duration = time.monotonic() - epoch_start
            samples_per_sec = len(train) / epoch_duration
            avg_loss = epoch_loss / max(num_batches, 1)
            top1, top5 = measure_accuracy(trainer, holdout, device)
            current_lr = trainer.optimizer.param_groups[0]["lr"]

            logger.log_epoch(epoch, avg_loss, top1, top5, current_lr)
            logger.log_epoch_timing(epoch, epoch_duration, samples_per_sec)
            logger.log_resource_metrics(epoch, monitor.summarize())
            logger.log_training_dynamics(epoch, step_losses, step_grad_norms)
            logger.log_pipeline_timing(epoch, data_time, compute_time)
```

**Step 4: Verify existing tests still pass**

Run: `uv run pytest tests/ -x -q`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/denoisr/scripts/train_phase1.py
git commit -m "feat: integrate resource monitoring into Phase 1 training"
```

---

### Task 5: Integrate ResourceMonitor into Phase 2 training

**Files:**
- Modify: `src/denoisr/scripts/train_phase2.py`

**Step 1: Add imports**

At the top of `train_phase2.py`, add:

```python
from denoisr.training.resource_monitor import ResourceMonitor
```

**Step 2: Create monitor before training loop**

After `global_step = 0` (line 183), add:

```python
        monitor = ResourceMonitor()
```

**Step 3: Instrument the epoch loop**

Replace the epoch training loop (lines 185-214) with the instrumented version. Same pattern as Phase 1:

```python
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.monotonic()
            monitor.reset()
            step_losses: list[float] = []
            step_grad_norms: list[float] = []
            data_time = 0.0
            compute_time = 0.0

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
                smoothing=0.3,
            )
            data_start = time.monotonic()
            for (batch,) in pbar:
                data_time += time.monotonic() - data_start

                batch = batch.to(device, non_blocking=True)
                compute_start = time.monotonic()
                loss, breakdown = diff_trainer.train_step(batch)
                compute_time += time.monotonic() - compute_start

                logger.log_train_step(global_step, loss, breakdown)
                step_losses.append(loss)
                step_grad_norms.append(breakdown.get("grad_norm", 0.0))
                if global_step % 100 == 0:
                    logger.log_gpu(global_step)
                    monitor.sample()
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix(loss=f"{loss:.4f}")
                data_start = time.monotonic()
            pbar.close()

            diff_trainer.advance_curriculum()
            epoch_duration = time.monotonic() - epoch_start
            num_samples = len(dataset)
            avg_loss = epoch_loss / max(num_batches, 1)

            logger.log_diffusion(epoch, avg_loss, diff_trainer.current_max_steps)
            logger.log_epoch_timing(epoch, epoch_duration, num_samples / epoch_duration)
            logger.log_resource_metrics(epoch, monitor.summarize())
            logger.log_training_dynamics(epoch, step_losses, step_grad_norms)
            logger.log_pipeline_timing(epoch, data_time, compute_time)
```

**Step 4: Verify existing tests still pass**

Run: `uv run pytest tests/ -x -q`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/denoisr/scripts/train_phase2.py
git commit -m "feat: integrate resource monitoring into Phase 2 training"
```

---

### Task 6: Update README metrics table

**Files:**
- Modify: `README.md` (lines 263-272, the "What gets logged" table)

**Step 1: Update the table**

Replace the existing metrics table with the expanded version:

```markdown
| Metric                                                | Frequency       | Phase |
| ----------------------------------------------------- | --------------- | ----- |
| `loss/total`, `loss/policy`, `loss/value`             | Every batch     | 1     |
| `gradients/norm` (pre-clip L2 norm)                   | Every batch     | 1, 2  |
| `accuracy/top1`, `accuracy/top5`                      | Every epoch     | 1     |
| `lr` (learning rate)                                  | Every epoch     | 1     |
| `diffusion/loss`, `diffusion/curriculum_steps`        | Every epoch     | 2     |
| `timing/epoch_duration_s`, `timing/samples_per_sec`   | Every epoch     | 1, 2  |
| `resources/cpu_percent_avg`, `cpu_percent_peak`       | Every epoch     | 1, 2  |
| `resources/ram_mb_avg`, `ram_mb_peak`                 | Every epoch     | 1, 2  |
| `resources/gpu_util_avg`, `gpu_util_peak`             | Every epoch     | 1, 2  |
| `resources/gpu_mem_mb_avg`, `gpu_mem_mb_peak`         | Every epoch     | 1, 2  |
| `resources/gpu_temp_avg`, `gpu_power_avg`             | Every epoch     | 1, 2  |
| `dynamics/grad_norm_avg`, `grad_norm_peak`            | Every epoch     | 1, 2  |
| `dynamics/loss_stddev`                                | Every epoch     | 1, 2  |
| `pipeline/data_wait_frac`, `compute_frac`             | Every epoch     | 1, 2  |
| `gpu/memory_allocated_mb`, `gpu/memory_reserved_mb`   | Every 100 steps | 1, 2  |
| Hyperparameters (lr, batch_size, d_s, num_heads, ...) | Once at start   | 1, 2  |
```

Also update the example `metrics.log` output (around line 303) to include resource lines:

```
epoch=0   avg_loss=6.566337   top1=0.0000   top5=0.0000   lr=1.00e-04
epoch=0   duration_s=3.83     samples_per_sec=496.1
epoch=0   cpu_avg=45.2%   cpu_peak=98.1%   ram_avg=2341mb   ram_peak=2567mb
epoch=0   grad_norm_avg=0.342   grad_norm_peak=1.000   loss_std=0.0512
epoch=0   data_wait_s=0.45   data_wait_frac=0.12   compute_frac=0.88
```

**Step 2: Verify tests still pass**

Run: `uv run pytest tests/ -x -q`
Expected: all tests PASS

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with new resource and diagnostics metrics"
```

---

### Task 7: Run full verification

**Step 1: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: all tests PASS

**Step 2: Run linter**

Run: `uvx ruff check`
Expected: no errors

**Step 3: Run type checker**

Run: `uv run --with mypy mypy --strict src/denoisr/training/resource_monitor.py src/denoisr/training/logger.py`
Expected: no errors (or only pre-existing ones)

**Step 4: Commit any fixes from linting/type-checking**

If needed:
```bash
git add -A && git commit -m "fix: address lint and type issues in resource metrics"
```
