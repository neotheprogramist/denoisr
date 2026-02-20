# Resource & Diagnostics Metrics Design

## Goal

Add comprehensive resource monitoring and training diagnostics to the existing logging infrastructure. Currently we log loss, accuracy, timing, and GPU memory. This extends coverage to CPU/GPU utilization, system RAM, GPU thermals/power, training dynamics, and data pipeline efficiency.

## Metrics

### Resource metrics (per-epoch avg/peak via sampling every 100 steps)

| TensorBoard tag | Unit | Source |
|---|---|---|
| `resources/cpu_percent_avg` | % | `psutil.Process().cpu_percent()` |
| `resources/cpu_percent_peak` | % | max of samples |
| `resources/ram_mb_avg` | MB | `psutil.Process().memory_info().rss` |
| `resources/ram_mb_peak` | MB | max of samples |
| `resources/gpu_util_avg` | % | `pynvml` utilization rates |
| `resources/gpu_util_peak` | % | max of samples |
| `resources/gpu_mem_mb_avg` | MB | `torch.cuda.memory_allocated()` |
| `resources/gpu_mem_mb_peak` | MB | max of samples |
| `resources/gpu_temp_avg` | C | `pynvml` temperature |
| `resources/gpu_temp_peak` | C | max of samples |
| `resources/gpu_power_avg` | W | `pynvml` power usage / 1000 |
| `resources/gpu_power_peak` | W | max of samples |

### Training dynamics (per-epoch from in-loop accumulation)

| TensorBoard tag | Unit | Source |
|---|---|---|
| `dynamics/grad_norm_avg` | - | mean of per-step gradient norms |
| `dynamics/grad_norm_peak` | - | max of per-step gradient norms |
| `dynamics/loss_stddev` | - | std dev of per-step losses within epoch |

### Data pipeline (per-epoch from timing instrumentation)

| TensorBoard tag | Unit | Source |
|---|---|---|
| `pipeline/data_wait_s` | seconds | time waiting for DataLoader |
| `pipeline/data_wait_frac` | 0-1 | fraction of epoch spent waiting |
| `pipeline/compute_frac` | 0-1 | 1 - data_wait_frac |

## Architecture

### New file: `src/denoisr/training/resource_monitor.py`

`ResourceMonitor` class:
- `__init__()` — detect available subsystems (psutil, pynvml), initialize NVML if CUDA available
- `reset()` — clear accumulated samples, called at epoch start
- `sample()` — snapshot CPU%, RAM, GPU util/mem/temp/power; append to internal lists
- `summarize() -> dict[str, float]` — compute avg/peak for all collected metrics
- Graceful degradation: each subsystem independently optional via internal flags

### New logger methods in `TrainingLogger`

- `log_resource_metrics(epoch, metrics_dict)` — write resource avg/peak to TensorBoard + text
- `log_training_dynamics(epoch, losses, grad_norms)` — compute and write dynamics stats
- `log_pipeline_timing(epoch, data_time, compute_time)` — write pipeline efficiency

### Training script changes (Phase 1 and Phase 2)

```python
# Before epoch loop
monitor = ResourceMonitor()

# At epoch start
monitor.reset()

# In batch loop
data_start = time.monotonic()
batch = next(iterator)  # or for-loop iteration
data_time += time.monotonic() - data_start

compute_start = time.monotonic()
loss, breakdown = trainer.train_step(...)
compute_time += time.monotonic() - compute_start

step_losses.append(loss)
step_grad_norms.append(breakdown.get("grad_norm", 0.0))

if global_step % 100 == 0:
    monitor.sample()

# At epoch end
logger.log_resource_metrics(epoch, monitor.summarize())
logger.log_training_dynamics(epoch, step_losses, step_grad_norms)
logger.log_pipeline_timing(epoch, data_time, compute_time)
```

### Text log format

```
epoch=0  cpu_avg=45.2%  cpu_peak=98.1%  ram_avg=2341mb  ram_peak=2567mb
epoch=0  gpu_util_avg=87.3%  gpu_util_peak=100.0%  gpu_temp_avg=72C  gpu_power_avg=185W
epoch=0  grad_norm_avg=0.342  grad_norm_peak=1.000  loss_std=0.0512
epoch=0  data_wait_frac=0.12  compute_frac=0.88
```

## Dependencies

- `psutil` — CPU and RAM monitoring (add as dev dependency)
- `nvidia-ml-py3` — GPU utilization, temperature, power (add as dev dependency, optional at runtime)

## Graceful degradation

- No CUDA: GPU metrics silently omitted
- No pynvml: GPU util/temp/power omitted, GPU memory still works via torch.cuda
- MPS (Apple Silicon): CPU and RAM work, GPU metrics omitted
- Import errors caught at ResourceMonitor init, sets availability flags

## Files to create/modify

1. **Create** `src/denoisr/training/resource_monitor.py` — ResourceMonitor class
2. **Modify** `src/denoisr/training/logger.py` — add 3 new logging methods
3. **Modify** `src/denoisr/scripts/train_phase1.py` — integrate monitor + timing
4. **Modify** `src/denoisr/scripts/train_phase2.py` — integrate monitor + timing
5. **Modify** `README.md` — update "What gets logged" table
6. **Create** `tests/test_training/test_resource_monitor.py` — unit tests
7. **Modify** `tests/test_training/test_logger.py` — tests for new logger methods
