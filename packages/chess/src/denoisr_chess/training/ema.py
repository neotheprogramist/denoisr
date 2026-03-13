"""Exponential Moving Average of model parameters for evaluation."""

import copy
import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class ModelEMA:
    """Maintains EMA shadow weights for multiple nn.Modules.

    Usage:
        ema = ModelEMA({"encoder": encoder, "backbone": backbone, ...}, decay=0.999)

        # After each optimizer step:
        ema.update()

        # For evaluation:
        with ema.apply():
            accuracy = evaluate(encoder, backbone, ...)
        # Original weights restored automatically

        # Save EMA weights:
        ema_state = ema.state_dicts()
        # Returns {"encoder": {...}, "backbone": {...}, ...}
    """

    def __init__(
        self,
        modules: dict[str, nn.Module],
        decay: float = 0.999,
    ) -> None:
        self._modules = modules
        self._decay = decay
        # Deep copy initial state dicts as shadow weights
        self._shadow: dict[str, dict[str, torch.Tensor]] = {
            name: copy.deepcopy(module.state_dict()) for name, module in modules.items()
        }
        self._steps = 0

    @property
    def decay(self) -> float:
        return self._decay

    def update(self) -> None:
        """Update shadow weights: theta_ema = decay * theta_ema + (1 - decay) * theta."""
        self._steps += 1
        d = self._decay
        with torch.no_grad():
            for name, module in self._modules.items():
                for key, param in module.state_dict().items():
                    if param.is_floating_point():
                        self._shadow[name][key].lerp_(param.data, 1.0 - d)
                    else:
                        # Non-float (e.g. num_batches_tracked) -- copy directly
                        self._shadow[name][key].copy_(param.data)

    def apply(self) -> "_EMAContext":
        """Context manager: swap in EMA weights, restore originals on exit."""
        return _EMAContext(self._modules, self._shadow)

    def state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return shadow state dicts for saving."""
        return {name: dict(sd) for name, sd in self._shadow.items()}

    def load_state_dicts(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        """Load previously saved shadow state dicts."""
        for name, sd in state.items():
            if name in self._shadow:
                self._shadow[name] = sd


class _EMAContext:
    """Context manager that swaps EMA weights in, restores originals on exit."""

    def __init__(
        self,
        modules: dict[str, nn.Module],
        shadow: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        self._modules = modules
        self._shadow = shadow
        self._backup: dict[str, dict[str, torch.Tensor]] = {}

    def __enter__(self) -> None:
        for name, module in self._modules.items():
            self._backup[name] = copy.deepcopy(module.state_dict())
            module.load_state_dict(self._shadow[name])

    def __exit__(self, *args: object) -> None:
        for name, module in self._modules.items():
            module.load_state_dict(self._backup[name])
        self._backup.clear()
