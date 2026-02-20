"""TensorBoard training logger for denoisr.

Writes both TensorBoard event files (for interactive visualization)
and human-readable text logs (for quick inspection without a viewer).

Text logs written:
    logs/<run-name>/metrics.log   — tab-separated epoch metrics
    logs/<run-name>/hparams.txt   — hyperparameter dump
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


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

    def log_resource_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log per-epoch resource utilization avg/peak."""
        for key, value in metrics.items():
            self._writer.add_scalar(f"resources/{key}", value, epoch)

        parts: list[str] = [f"epoch={epoch}"]
        if "cpu_percent_avg" in metrics:
            parts.append(f"cpu_avg={metrics['cpu_percent_avg']:.1f}%")
            parts.append(f"cpu_peak={metrics['cpu_percent_peak']:.1f}%")
        if "ram_mb_avg" in metrics:
            parts.append(f"ram_avg={metrics['ram_mb_avg']:.0f}mb")
            parts.append(f"ram_peak={metrics['ram_mb_peak']:.0f}mb")
        self._write_text("\t".join(parts))

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

    def log_epoch_summary(self, parts: dict[str, str]) -> None:
        """Emit a single consolidated epoch summary via logging.

        Designed for agent-friendly output: one structured line per epoch
        with all metrics, easily parsed via key=value splitting.
        """
        line = "  ".join(f"{k}={v}" for k, v in parts.items())
        log.info(line)

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
