"""Human-readable training logger for denoisr."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class TrainingLogger:
    """Thin wrapper for concise epoch-level training logs."""

    def __init__(
        self,
        log_dir: Path,
        run_name: str | None = None,
    ) -> None:
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_name = run_name
        self._last_warned_grok_state: int | None = None

        log_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_root_logging(log_dir)

        self._metrics_logger = logging.getLogger(f"denoisr.metrics.{self._run_name}")
        self._metrics_logger.setLevel(logging.INFO)
        self._metrics_logger.propagate = True

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

    def log_epoch_line(
        self,
        *,
        epoch: int,
        total_epochs: int,
        losses: dict[str, float],
        lr: float,
        grad_norms: list[float],
        samples_per_sec: float,
        duration_s: float,
        accuracy: dict[str, float] | None = None,
        resources: dict[str, str] | None = None,
        data_pct: float = 0.0,
        overflows: int = 0,
        phase: str = "phase1",
        generation: int | None = None,
        total_generations: int | None = None,
    ) -> None:
        """Emit one compact line per epoch."""
        _ = (phase, generation, total_generations)
        parts: list[str] = [f"E={epoch}/{total_epochs}"]

        for key, value in losses.items():
            parts.append(f"{key}={value:.4f}" if abs(value) < 10 else f"{key}={value:.2f}")

        if accuracy:
            for key, value in accuracy.items():
                parts.append(f"{key}={value:.1f}%")

        parts.append(f"lr={lr:.1e}")

        finite_norms = [n for n in grad_norms if math.isfinite(n)]
        if finite_norms:
            avg_gn = sum(finite_norms) / len(finite_norms)
            peak_gn = max(finite_norms)
            parts.append(f"gnorm={avg_gn:.1f}/{peak_gn:.1f}")

        if overflows > 0:
            parts.append(f"ovf={overflows}")

        parts.append(f"sps={samples_per_sec:.0f}")
        parts.append(f"t={duration_s:.1f}s")

        if resources:
            if "cpu_pct" in resources:
                parts.append(f"cpu={resources['cpu_pct']}/{resources.get('cpu_max', 'n/a')}")
            if "ram_mb" in resources:
                parts.append(f"ram={resources['ram_mb']}mb")
            if "gpu_util" in resources:
                gpu_part = f"gpu={resources['gpu_util']}/{resources.get('gpu_mem_pct', 'n/a')}"
                if "gpu_mem_mb" in resources:
                    gpu_part += f" {resources['gpu_mem_mb']}mb"
                if "gpu_temp" in resources:
                    gpu_part += f" {resources['gpu_temp']}C"
                if "gpu_power" in resources:
                    gpu_part += f" {resources['gpu_power']}W"
                parts.append(gpu_part)

        parts.append(f"data={data_pct:.0f}%")
        self._metrics_logger.info(" ".join(parts))

    def log_epoch_summary(self, parts: dict[str, str]) -> None:
        """Emit a free-form summary line."""
        line = "  ".join(f"{k}={v}" for k, v in parts.items())
        self._metrics_logger.info(line)

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None:
        """Log configured hyperparameters and tracked targets as one line."""
        hp = " ".join(f"{k}={v}" for k, v in sorted(hparams.items()))
        mt = " ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
        if mt:
            self._metrics_logger.info("HPARAMS %s TARGETS %s", hp, mt)
        else:
            self._metrics_logger.info("HPARAMS %s", hp)

    def log_grok_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log compact grokking summaries at epoch granularity."""
        holdout_keys = [k for k in metrics if k.startswith("grok/holdout/")]
        if not holdout_keys:
            return

        state_value = int(metrics.get("grok/state", 0.0))
        state_name = {
            0: "BASELINE",
            1: "ONSET_DETECTED",
            2: "TRANSITIONING",
            3: "GROKKED",
        }.get(state_value, f"UNKNOWN({state_value})")

        random_acc = metrics.get("grok/holdout/random/accuracy")
        loss_gap = metrics.get("grok/loss_gap")

        parts = [f"epoch={step}", f"state={state_name}"]
        if random_acc is not None:
            parts.append(f"random_acc={random_acc * 100:.2f}%")
        if loss_gap is not None:
            parts.append(f"loss_gap={loss_gap:.4f}")

        split_acc_parts: list[str] = []
        for key in sorted(holdout_keys):
            if key.endswith("/accuracy") and "/random/" not in key:
                split = key.split("/")[2]
                split_acc_parts.append(f"{split}={metrics[key] * 100:.2f}%")
        if split_acc_parts:
            parts.append("splits=" + ",".join(split_acc_parts))

        self._metrics_logger.info("GROK-EPOCH %s", " ".join(parts))

        if state_value >= 1 and state_value != self._last_warned_grok_state:
            self._last_warned_grok_state = state_value
            self._metrics_logger.warning(
                "GROKKING state entered: %s at epoch=%d",
                state_name,
                step,
            )

    def log_grok_state_transition(
        self,
        step: int,
        old_state: str,
        new_state: str,
        trigger: str,
    ) -> None:
        """Log grokking state transitions as warnings."""
        self._metrics_logger.warning(
            "GROKKING transition step=%d %s->%s trigger=%s",
            step,
            old_state,
            new_state,
            trigger,
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
