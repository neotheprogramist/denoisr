# Training Logging Design

**Goal:** Full training observability — loss curves, accuracy tracking, GPU utilization, gradient health, and hyperparameter comparison — persisted to TensorBoard event files in `logs/`.

**Approach:** A thin `TrainingLogger` class wrapping `SummaryWriter`, called from training scripts. Trainers stay unchanged except returning gradient norm in their breakdown dicts.

**Backend:** TensorBoard (ships with PyTorch, no new dependencies). View with `uvx tensorboard --logdir logs/`.

---

## TrainingLogger API

Single class in `src/denoisr/training/logger.py`:

```python
class TrainingLogger:
    def __init__(self, log_dir: Path, run_name: str | None = None) -> None:
        # run_name defaults to timestamp "YYYY-MM-DD_HH-MM-SS"
        # Creates SummaryWriter at log_dir / run_name

    def log_train_step(self, step: int, loss: float, breakdown: dict[str, float]) -> None:
        # loss/total, loss/policy, loss/value, gradients/norm (from breakdown)

    def log_epoch(self, epoch: int, avg_loss: float, top1: float, top5: float, lr: float) -> None:
        # epoch/avg_loss, accuracy/top1, accuracy/top5, lr

    def log_epoch_timing(self, epoch: int, duration_s: float, samples_per_sec: float) -> None:
        # timing/epoch_duration_s, timing/samples_per_sec

    def log_gpu(self, step: int) -> None:
        # gpu/memory_allocated_mb, gpu/memory_reserved_mb (no-op on CPU/MPS)

    def log_diffusion(self, epoch: int, avg_loss: float, curriculum_steps: int) -> None:
        # diffusion/loss, diffusion/curriculum_steps

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None:
        # TensorBoard HParams tab for run comparison

    def close(self) -> None:
        # Flush and close writer
```

## What Gets Logged

**Per training step (every batch):**
- `loss/total`, `loss/policy`, `loss/value` — from breakdown dict
- `gradients/norm` — from clip_grad_norm_ return value, added to breakdown

**Per epoch:**
- `epoch/avg_loss`
- `accuracy/top1`, `accuracy/top5` (Phase 1)
- `lr` — read from optimizer param groups
- `timing/epoch_duration_s`, `timing/samples_per_sec`
- `gpu/memory_allocated_mb`, `gpu/memory_reserved_mb`
- `diffusion/loss`, `diffusion/curriculum_steps` (Phase 2)

**Once at run start:**
- HParams: batch_size, lr, d_s, num_heads, num_layers, ffn_dim, gradient_checkpointing, num_planes

## Gradient Norm Capture

Both trainers call `clip_grad_norm_` but discard its return value (the pre-clip total norm). Change to capture and include in breakdown:

- `SupervisedTrainer._forward_backward`: add `breakdown["grad_norm"] = total_norm.item()`
- `DiffusionTrainer.train_step`: change return type to `tuple[float, dict[str, float]]`, return loss + `{"grad_norm": total_norm.item()}`

## CLI Integration

Add `--run-name` flag to `train_phase1.py` and `train_phase2.py`:
```
--run-name NAME    TensorBoard run name (default: auto-generated timestamp)
```

Log directory: `logs/<run_name>/`

## Directory Layout

```
logs/
├── lr1e-4_bs64/
│   └── events.out.tfevents.*
├── lr1e-4_bs256/
│   └── events.out.tfevents.*
└── 2026-02-20_02-15-30/     # auto-generated when no --run-name
    └── events.out.tfevents.*
```

## What We're NOT Doing

- No Weights & Biases, MLflow, or other cloud services
- No custom dashboard
- No logging in the trainer classes themselves (scripts control logging)
- No per-parameter gradient histograms (expensive, rarely needed)
- No image logging (board visualizations)
