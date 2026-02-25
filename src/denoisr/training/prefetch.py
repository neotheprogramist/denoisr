"""Utilities for overlapping host->device transfer with compute."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

import torch

_BatchT = TypeVar("_BatchT")


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


class DevicePrefetcher(Iterator[_BatchT]):
    """Iterator that preloads the next batch onto a CUDA stream."""

    def __init__(self, loader: Iterable[_BatchT], device: torch.device) -> None:
        self._iter = iter(loader)
        self._device = device
        self._stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._next_batch: _BatchT | None = None
        if self._stream is not None:
            self._preload()

    def _preload(self) -> None:
        assert self._stream is not None
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = _to_device(batch, self._device)

    def __iter__(self) -> DevicePrefetcher[_BatchT]:
        return self

    def __next__(self) -> _BatchT:
        if self._stream is None:
            return next(self._iter)
        if self._next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(device=self._device).wait_stream(self._stream)
        batch = self._next_batch
        self._preload()
        return batch
