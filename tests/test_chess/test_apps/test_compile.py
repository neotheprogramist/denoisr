import pytest
import torch
from torch import nn

import denoisr_chess.config as config


class TestMaybeCompile:
    def test_returns_module_on_cpu(self) -> None:
        """On CPU, maybe_compile should return the original module."""
        m = nn.Linear(4, 4)
        result = config.maybe_compile(m, torch.device("cpu"))
        assert result is m

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_returns_module_on_mps(self) -> None:
        """On MPS, maybe_compile should return the original module."""
        m = nn.Linear(4, 4)
        result = config.maybe_compile(m, torch.device("mps"))
        assert result is m

    def test_cuda_uses_torch_compile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On CUDA, maybe_compile should delegate to torch.compile."""
        m = nn.Linear(4, 4)
        compiled = nn.Sequential(m)
        monkeypatch.setattr(config.torch, "compile", lambda module: compiled)

        result = config.maybe_compile(m, torch.device("cuda"))
        assert result is compiled
