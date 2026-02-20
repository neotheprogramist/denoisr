"""TensorBoard training logger for denoisr.

Writes both TensorBoard event files (for interactive visualization)
and human-readable text logs (for quick inspection without a viewer).

Text logs written:
    logs/<run-name>/metrics.log   — tab-separated epoch metrics
    logs/<run-name>/hparams.txt   — hyperparameter dump
"""

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
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(str(self._run_dir))
        self._log_file = open(  # noqa: SIM115
            self._run_dir / "metrics.log", "a", encoding="utf-8"
        )

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
        self._write_text(
            f"epoch={epoch}\tavg_loss={avg_loss:.6f}\t"
            f"top1={top1:.4f}\ttop5={top5:.4f}\tlr={lr:.2e}"
        )

    def log_epoch_timing(
        self, epoch: int, duration_s: float, samples_per_sec: float
    ) -> None:
        """Log epoch timing metrics."""
        self._writer.add_scalar("timing/epoch_duration_s", duration_s, epoch)
        self._writer.add_scalar("timing/samples_per_sec", samples_per_sec, epoch)
        self._write_text(
            f"epoch={epoch}\tduration_s={duration_s:.2f}\t"
            f"samples_per_sec={samples_per_sec:.1f}"
        )

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
        self._write_text(
            f"epoch={epoch}\tdiffusion_loss={avg_loss:.6f}\t"
            f"curriculum_steps={curriculum_steps}"
        )

    def log_hparams(
        self, hparams: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        """Log hyperparameters for TensorBoard HParams tab and text file."""
        self._writer.add_hparams(hparams, metrics)
        hparams_path = self._run_dir / "hparams.txt"
        with open(hparams_path, "w", encoding="utf-8") as f:
            for key, value in hparams.items():
                f.write(f"{key}={value}\n")

    def _write_text(self, line: str) -> None:
        """Append a line to the text log file."""
        self._log_file.write(line + "\n")
        self._log_file.flush()

    def close(self) -> None:
        """Flush and close all writers."""
        self._writer.flush()
        self._writer.close()
        self._log_file.close()

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
