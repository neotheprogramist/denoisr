"""Plateau detection for training loops.

Tracks gradient norm EMA, loss EMA, and effective update magnitude
to warn when training has stalled.
"""

import logging
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class PlateauDetector:
    """Detects training plateaus via gradient norm, loss stall, and update magnitude.

    Call ``update()`` after each epoch. Returns a list of warning messages
    (empty if everything looks healthy).
    """

    window: int = 10
    grad_threshold: float = 0.15
    loss_rel_threshold: float = 1e-3
    update_mag_threshold: float = 1e-4

    _grad_ema: float = field(default=0.0, init=False, repr=False)
    _loss_history: deque[float] = field(default_factory=deque, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def update(
        self,
        epoch: int,
        grad_norm: float,
        loss: float,
        lr: float,
    ) -> list[str]:
        warnings: list[str] = []
        alpha = 0.1

        if not self._initialized:
            self._grad_ema = grad_norm
            self._initialized = True
        else:
            self._grad_ema = alpha * grad_norm + (1 - alpha) * self._grad_ema

        self._loss_history.append(loss)
        if len(self._loss_history) > self.window:
            self._loss_history.popleft()

        # Check 1: gradient norm collapse
        if self._grad_ema < self.grad_threshold:
            msg = (
                f"Epoch {epoch}: gradient norm EMA collapsed to "
                f"{self._grad_ema:.4f} (threshold={self.grad_threshold})"
            )
            warnings.append(msg)
            log.warning(msg)

        # Check 2: loss stall over window
        if len(self._loss_history) >= self.window:
            oldest = self._loss_history[0]
            newest = self._loss_history[-1]
            if oldest > 0:
                rel_change = abs(newest - oldest) / oldest
                if rel_change < self.loss_rel_threshold:
                    msg = (
                        f"Epoch {epoch}: loss stalled — relative change "
                        f"{rel_change:.6f} over {self.window} epochs "
                        f"(threshold={self.loss_rel_threshold})"
                    )
                    warnings.append(msg)
                    log.warning(msg)

        # Check 3: effective update magnitude
        effective_update = lr * self._grad_ema
        if effective_update < self.update_mag_threshold:
            msg = (
                f"Epoch {epoch}: effective update magnitude "
                f"{effective_update:.2e} (lr={lr:.2e} × grad_norm_ema="
                f"{self._grad_ema:.4f}) below threshold "
                f"{self.update_mag_threshold:.2e}"
            )
            warnings.append(msg)
            log.warning(msg)

        return warnings
