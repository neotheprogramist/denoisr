"""TensorBoard training logger for denoisr."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter


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
