"""Stochastic Weight Averaging utilities for multi-module training stacks."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterator

from torch import nn
from torch.optim.swa_utils import AveragedModel


class ModelSWA:
    """Maintains SWA shadows for one or more modules."""

    def __init__(self, modules: dict[str, nn.Module]) -> None:
        if not modules:
            raise ValueError("ModelSWA requires at least one module")
        self._modules = modules
        self._averaged = {
            name: AveragedModel(module) for name, module in self._modules.items()
        }
        self._num_updates = 0

    @property
    def num_updates(self) -> int:
        return self._num_updates

    def has_batch_norm(self) -> bool:
        for module in self._modules.values():
            if any(
                isinstance(submodule, nn.modules.batchnorm._BatchNorm)
                for submodule in module.modules()
            ):
                return True
        return False

    def update(self) -> None:
        for name, module in self._modules.items():
            self._averaged[name].update_parameters(module)
        self._num_updates += 1

    @contextmanager
    def apply(self) -> Iterator[None]:
        backups = {
            name: copy.deepcopy(module.state_dict())
            for name, module in self._modules.items()
        }
        try:
            for name, module in self._modules.items():
                module.load_state_dict(self._averaged[name].module.state_dict())
            yield
        finally:
            for name, module in self._modules.items():
                module.load_state_dict(backups[name])

    def state_dicts(self) -> dict[str, dict[str, object]]:
        return {
            name: copy.deepcopy(averaged.module.state_dict())
            for name, averaged in self._averaged.items()
        }
