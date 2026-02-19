import torch
from torch import nn

from denoisr.scripts.config import maybe_compile


class TestMaybeCompile:
    def test_returns_module_on_cpu(self) -> None:
        """On CPU, maybe_compile should return the original module (no compile)."""
        m = nn.Linear(4, 4)
        result = maybe_compile(m, torch.device("cpu"))
        assert result is m  # exact same object

    def test_returns_module_on_mps_if_available(self) -> None:
        """On MPS, maybe_compile should return the original module."""
        if not torch.backends.mps.is_available():
            return  # skip on non-Mac
        m = nn.Linear(4, 4)
        result = maybe_compile(m, torch.device("mps"))
        assert result is m

    def test_compiled_module_on_cuda_if_available(self) -> None:
        """On CUDA, maybe_compile should return a compiled wrapper."""
        if not torch.cuda.is_available():
            return  # skip if no CUDA
        m = nn.Linear(4, 4).cuda()
        result = maybe_compile(m, torch.device("cuda"))
        assert result is not m  # should be wrapped
