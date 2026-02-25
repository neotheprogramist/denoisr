"""Structured training logger for denoisr.

Writes both compact epoch metrics and scalar/hparam events into the
shared process log stream (typically ``logs/denoisr.log`` configured by
``denoisr.scripts.runtime.configure_logging``).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

log = logging.getLogger(__name__)


class _StructuredEventWriter:
    """SummaryWriter-like adapter that emits JSON event lines via logging."""

    def __init__(self, logger: logging.Logger, run_name: str) -> None:
        self._logger = logger
        self._run_name = run_name

    def _emit(self, payload: dict[str, Any]) -> None:
        full_payload = {"run": self._run_name, **payload}
        self._logger.info(
            "EVENT %s",
            json.dumps(full_payload, sort_keys=True, default=str),
        )

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        *_: Any,
        **__: Any,
    ) -> None:
        try:
            value: float | str = float(scalar_value)
        except (TypeError, ValueError):
            value = str(scalar_value)
        self._emit(
            {
                "kind": "scalar",
                "tag": tag,
                "step": int(global_step) if global_step is not None else None,
                "value": value,
            }
        )

    def add_hparams(
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, float],
        *_: Any,
        **__: Any,
    ) -> None:
        self._emit(
            {
                "kind": "hparams",
                "hparams": hparam_dict,
                "metrics": metric_dict,
            }
        )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class TrainingLogger:
    """Thin wrapper for structured training metrics.

    Usage:
        with TrainingLogger(Path("logs"), run_name="lr1e-4") as logger:
            logger.log_train_step(step, loss, breakdown)
            logger.log_epoch_line(epoch=1, total_epochs=100, ...)
    """

    def __init__(
        self,
        log_dir: Path,
        run_name: str | None = None,
    ) -> None:
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_name = run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_root_logging(log_dir)

        # Dedicated metrics logger; records propagate to root handlers.
        self._metrics_logger = logging.getLogger(f"denoisr.metrics.{self._run_name}")
        self._metrics_logger.setLevel(logging.INFO)
        self._metrics_logger.propagate = True

        self._events_logger = logging.getLogger("denoisr.events")
        self._events_logger.setLevel(logging.INFO)
        self._events_logger.propagate = True
        self._writer = _StructuredEventWriter(self._events_logger, self._run_name)

    @staticmethod
    def _ensure_root_logging(log_dir: Path) -> None:
        """Install a fallback file logger when scripts forgot to configure logging."""
        root = logging.getLogger()
        if root.handlers:
            return
        fallback_path = log_dir / "denoisr.log"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(fallback_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    # ------------------------------------------------------------------
    # Per-batch logging (unchanged)
    # ------------------------------------------------------------------

    def log_train_step(
        self, step: int, loss: float, breakdown: dict[str, float]
    ) -> None:
        """Log per-batch training metrics."""
        self._writer.add_scalar("loss/total", loss, step)
        for key, value in breakdown.items():
            if key in ("total", "overflow"):
                continue
            if key == "grad_norm":
                self._writer.add_scalar("gradients/norm", value, step)
            else:
                self._writer.add_scalar(f"loss/{key}", value, step)

    # ------------------------------------------------------------------
    # Single-line epoch logging (replaces 6 per-epoch methods)
    # ------------------------------------------------------------------

    def log_epoch_line(
        self,
        *,
        epoch: int,
        total_epochs: int,
        losses: dict[str, float],
        step_losses: list[float] | None = None,
        lr: float,
        grad_norms: list[float],
        samples_per_sec: float,
        duration_s: float,
        accuracy: dict[str, float] | None = None,
        resources: dict[str, str] | None = None,
        data_pct: float = 0.0,
        overflows: int = 0,
        phase: str = "phase1",
        # Phase 2 specific
        generation: int | None = None,
        total_generations: int | None = None,
    ) -> None:
        """Emit one compact line per epoch and write scalar events.

        Consolidates the previous log_epoch, log_epoch_timing,
        log_resource_metrics, log_training_dynamics, log_pipeline_timing,
        and log_diffusion methods into a single call.
        """
        # --- Build the human-readable line ---
        parts: list[str] = []

        # Epoch
        parts.append(f"E={epoch}/{total_epochs}")

        # Losses
        for k, v in losses.items():
            parts.append(f"{k}={v:.4f}" if abs(v) < 10 else f"{k}={v:.2f}")

        # Accuracy (Phase 1 only)
        if accuracy:
            for k, v in accuracy.items():
                parts.append(f"{k}={v:.1f}%")

        # Learning rate
        parts.append(f"lr={lr:.1e}")

        # Grad norms (avg/peak)
        if grad_norms:
            finite_norms = [n for n in grad_norms if math.isfinite(n)]
            if finite_norms:
                avg_gn = sum(finite_norms) / len(finite_norms)
                peak_gn = max(finite_norms)
                parts.append(f"gnorm={avg_gn:.1f}/{peak_gn:.1f}")

        # Overflows
        if overflows > 0:
            parts.append(f"ovf={overflows}")

        # Speed
        parts.append(f"sps={samples_per_sec:.0f}")

        # Duration
        parts.append(f"t={duration_s:.1f}s")

        # Resource metrics
        if resources:
            r = resources
            if "cpu_pct" in r:
                parts.append(f"cpu={r['cpu_pct']}/{r.get('cpu_max', 'n/a')}")
            if "ram_mb" in r:
                parts.append(f"ram={r['ram_mb']}mb")
            if "gpu_util" in r:
                gpu_part = f"gpu={r['gpu_util']}/{r.get('gpu_mem_pct', 'n/a')}"
                if "gpu_mem_mb" in r:
                    gpu_part += f" {r['gpu_mem_mb']}mb"
                if "gpu_temp" in r:
                    gpu_part += f" {r['gpu_temp']}C"
                if "gpu_power" in r:
                    gpu_part += f" {r['gpu_power']}W"
                parts.append(gpu_part)

        # Data pipeline percentage
        parts.append(f"data={data_pct:.0f}%")

        line = " ".join(parts)
        self._metrics_logger.info(line)

        # --- Structured scalar event writes ---
        self._tb_losses(epoch, losses)
        self._tb_accuracy(epoch, accuracy)
        self._tb_lr(epoch, lr)
        self._tb_timing(epoch, duration_s, samples_per_sec)
        self._tb_dynamics(epoch, losses, grad_norms, step_losses)
        self._tb_resources(epoch, resources)
        self._tb_pipeline(epoch, data_pct, duration_s)

    # ------------------------------------------------------------------
    # Scalar event helpers (internal)
    # ------------------------------------------------------------------

    def _tb_losses(self, epoch: int, losses: dict[str, float]) -> None:
        """Write loss scalars."""
        if "loss" in losses:
            self._writer.add_scalar("epoch/avg_loss", losses["loss"], epoch)
        if "diff" in losses or "diffusion" in losses:
            diff_val = losses.get("diff", losses.get("diffusion", 0.0))
            self._writer.add_scalar("diffusion/loss", diff_val, epoch)
        for k, v in losses.items():
            if k not in ("loss",):
                self._writer.add_scalar(f"epoch/{k}_loss", v, epoch)

    def _tb_accuracy(self, epoch: int, accuracy: dict[str, float] | None) -> None:
        """Write accuracy scalars."""
        if not accuracy:
            return
        if "top1" in accuracy:
            self._writer.add_scalar("accuracy/top1", accuracy["top1"] / 100, epoch)
        if "top5" in accuracy:
            self._writer.add_scalar("accuracy/top5", accuracy["top5"] / 100, epoch)

    def _tb_lr(self, epoch: int, lr: float) -> None:
        self._writer.add_scalar("lr", lr, epoch)

    def _tb_timing(self, epoch: int, duration_s: float, samples_per_sec: float) -> None:
        self._writer.add_scalar("timing/epoch_duration_s", duration_s, epoch)
        self._writer.add_scalar("timing/samples_per_sec", samples_per_sec, epoch)

    def _tb_dynamics(
        self,
        epoch: int,
        losses: dict[str, float],
        grad_norms: list[float],
        step_losses: list[float] | None,
    ) -> None:
        if not grad_norms:
            return
        finite_norms = [n for n in grad_norms if math.isfinite(n)]
        if not finite_norms:
            return
        gn_avg = mean(finite_norms)
        gn_peak = max(finite_norms)
        self._writer.add_scalar("dynamics/grad_norm_avg", gn_avg, epoch)
        self._writer.add_scalar("dynamics/grad_norm_peak", gn_peak, epoch)

        loss_vals = (
            [v for v in step_losses if math.isfinite(v)]
            if step_losses is not None
            else list(losses.values())
        )
        loss_sd = stdev(loss_vals) if len(loss_vals) > 1 else 0.0
        self._writer.add_scalar("dynamics/loss_stddev", loss_sd, epoch)

    def _tb_resources(self, epoch: int, resources: dict[str, str] | None) -> None:
        """Write resource metrics as scalar events.

        Accepts the string-valued dict from log_epoch_line and parses
        numeric values where possible.
        """
        if not resources:
            return
        # Try to parse and write numeric resource values
        for key, val in resources.items():
            try:
                numeric = float(val.rstrip("%CWmb"))
                self._writer.add_scalar(f"resources/{key}", numeric, epoch)
            except ValueError, AttributeError:
                pass

    def _tb_pipeline(self, epoch: int, data_pct: float, duration_s: float) -> None:
        data_frac = data_pct / 100.0
        compute_frac = 1.0 - data_frac
        data_time = duration_s * data_frac
        self._writer.add_scalar("pipeline/data_wait_s", data_time, epoch)
        self._writer.add_scalar("pipeline/data_wait_frac", data_frac, epoch)
        self._writer.add_scalar("pipeline/compute_frac", compute_frac, epoch)

    # ------------------------------------------------------------------
    # GPU memory (unchanged)
    # ------------------------------------------------------------------

    def log_gpu(self, step: int) -> None:
        """Log GPU memory metrics. No-op on CPU/MPS."""
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        self._writer.add_scalar("gpu/memory_allocated_mb", allocated, step)
        self._writer.add_scalar("gpu/memory_reserved_mb", reserved, step)

    # ------------------------------------------------------------------
    # Epoch summary (redirected to _metrics_logger)
    # ------------------------------------------------------------------

    def log_epoch_summary(self, parts: dict[str, str]) -> None:
        """Emit a single consolidated epoch summary via logging.

        Designed for agent-friendly output: one structured line per epoch
        with all metrics, easily parsed via key=value splitting.
        """
        line = "  ".join(f"{k}={v}" for k, v in parts.items())
        self._metrics_logger.info(line)

    # ------------------------------------------------------------------
    # Hyperparameters (unchanged)
    # ------------------------------------------------------------------

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None:
        """Log hyperparameters as structured events in the shared log."""
        self._writer.add_hparams(hparams, metrics)

    # ------------------------------------------------------------------
    # Grokking detection (redirected to _metrics_logger)
    # ------------------------------------------------------------------

    def log_grok_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log grokking detection metrics as scalar events."""
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def log_grok_state_transition(
        self,
        step: int,
        old_state: str,
        new_state: str,
        trigger: str,
    ) -> None:
        """Log grokking state transition to text log."""
        self._metrics_logger.info(
            f"GROKKING step={step}\t{old_state}->{new_state}\ttrigger={trigger}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all writers."""
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
