"""Tests for SWA helper utilities."""

from __future__ import annotations

import torch
from torch import nn

from denoisr.training.swa import ModelSWA


def test_model_swa_averages_parameters() -> None:
    module = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    swa = ModelSWA({"linear": module})

    with torch.no_grad():
        module.weight.fill_(3.0)
    swa.update()

    with torch.no_grad():
        module.weight.fill_(5.0)
    swa.update()

    with swa.apply():
        assert torch.allclose(module.weight, torch.full_like(module.weight, 4.0))

    assert torch.allclose(module.weight, torch.full_like(module.weight, 5.0))


def test_model_swa_requires_modules() -> None:
    try:
        ModelSWA({})
    except ValueError as exc:
        assert "at least one module" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty module dict")
