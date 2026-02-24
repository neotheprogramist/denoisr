import pytest
import torch
from torch import nn

import denoisr.scripts.config as config


class TestMaybeCompile:
    def teardown_method(self) -> None:
        config._probe_cuda_compile_support.cache_clear()

    def test_returns_module_on_cpu(self) -> None:
        """On CPU, maybe_compile should return the original module (no compile)."""
        m = nn.Linear(4, 4)
        result = config.maybe_compile(m, torch.device("cpu"))
        assert result is m  # exact same object

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_returns_module_on_mps(self) -> None:
        """On MPS, maybe_compile should return the original module."""
        m = nn.Linear(4, 4)
        result = config.maybe_compile(m, torch.device("mps"))
        assert result is m

    def test_compile_off_returns_original_module(self) -> None:
        """--compile=off should always bypass torch.compile."""
        m = nn.Linear(4, 4)
        result = config.maybe_compile(m, torch.device("cuda"), compile_mode="off")
        assert result is m

    def test_auto_falls_back_when_probe_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto mode should return eager module when CUDA compile probe fails."""
        m = nn.Linear(4, 4)
        monkeypatch.setattr(config.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            config,
            "_probe_cuda_compile_support",
            lambda: (False, "ptxas not executable"),
        )
        result = config.maybe_compile(m, torch.device("cuda"), compile_mode="auto")
        assert result is m

    def test_auto_uses_torch_compile_when_probe_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto mode should call torch.compile when CUDA compile probe succeeds."""
        m = nn.Linear(4, 4)
        compiled = nn.Sequential(m)
        monkeypatch.setattr(config.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            config,
            "_probe_cuda_compile_support",
            lambda: (True, "ptxas=/tmp/ptxas"),
        )
        monkeypatch.setattr(config.torch, "compile", lambda module: compiled)
        result = config.maybe_compile(m, torch.device("cuda"), compile_mode="auto")
        assert result is compiled

    def test_on_mode_raises_when_probe_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--compile=on should fail fast when compile prerequisites are missing."""
        m = nn.Linear(4, 4)
        monkeypatch.setattr(config.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            config,
            "_probe_cuda_compile_support",
            lambda: (False, "ptxas not executable"),
        )
        with pytest.raises(RuntimeError, match="ptxas not executable"):
            config.maybe_compile(m, torch.device("cuda"), compile_mode="on")
