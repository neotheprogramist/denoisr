import pytest
import torch
from torch import nn

from denoisr.scripts.config import maybe_compile


class TestMaybeCompile:
    def test_returns_module_on_cpu(self) -> None:
        """On CPU, maybe_compile should return the original module (no compile)."""
        m = nn.Linear(4, 4)
        result = maybe_compile(m, torch.device("cpu"))
        assert result is m  # exact same object

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_returns_module_on_mps(self) -> None:
        """On MPS, maybe_compile should return the original module."""
        m = nn.Linear(4, 4)
        result = maybe_compile(m, torch.device("mps"))
        assert result is m

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_compiled_module_on_cuda(self) -> None:
        """On CUDA, maybe_compile should return a compiled wrapper."""
        m = nn.Linear(4, 4).cuda()
        result = maybe_compile(m, torch.device("cuda"))
        assert result is not m  # should be wrapped
