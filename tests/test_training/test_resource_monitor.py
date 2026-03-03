"""Tests for ResourceMonitor system metrics collection."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from denoisr.training.resource_monitor import ResourceMonitor


class TestResourceMonitorCPURAM:
    def test_summarize_empty_returns_empty(self) -> None:
        """Summarize with no samples should return empty dict."""
        monitor = ResourceMonitor()
        result = monitor.summarize()
        assert result == {}

    def test_sample_and_summarize_has_cpu_keys(self) -> None:
        """After sampling, summary should contain CPU avg/peak keys."""
        monitor = ResourceMonitor()
        monitor.sample()
        monitor.sample()
        result = monitor.summarize()
        assert "cpu_percent_avg" in result
        assert "cpu_percent_peak" in result
        assert result["cpu_percent_avg"] >= 0.0
        assert result["cpu_percent_peak"] >= result["cpu_percent_avg"]

    def test_sample_and_summarize_has_ram_keys(self) -> None:
        """After sampling, summary should contain RAM avg/peak keys."""
        monitor = ResourceMonitor()
        monitor.sample()
        result = monitor.summarize()
        assert "ram_mb_avg" in result
        assert "ram_mb_peak" in result
        assert result["ram_mb_avg"] > 0.0
        assert result["ram_mb_peak"] >= result["ram_mb_avg"]

    def test_reset_clears_samples(self) -> None:
        """After reset, summarize should return empty dict."""
        monitor = ResourceMonitor()
        monitor.sample()
        assert monitor.summarize() != {}
        monitor.reset()
        assert monitor.summarize() == {}

    def test_peak_is_max_of_samples(self) -> None:
        """Peak should be the max across samples, not just the last."""
        monitor = ResourceMonitor()
        for _ in range(5):
            monitor.sample()
        result = monitor.summarize()
        assert result["cpu_percent_peak"] >= result["cpu_percent_avg"]
        assert result["ram_mb_peak"] >= result["ram_mb_avg"]


class TestResourceMonitorFailFast:
    """Verify that GPU monitoring fails fast when CUDA is available."""

    def test_nvml_init_failure_propagates(self) -> None:
        """When CUDA is available but NVML fails, constructor raises."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML init failed")

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            pytest.raises(RuntimeError, match="NVML init failed"),
        ):
            ResourceMonitor()

    def test_nvml_handle_failure_propagates(self) -> None:
        """When CUDA is available but device handle fails, constructor raises."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError(
            "No device"
        )

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            pytest.raises(RuntimeError, match="No device"),
        ):
            ResourceMonitor()

    def test_no_cuda_skips_nvml(self) -> None:
        """When CUDA is not available, NVML is not initialized."""
        monitor = ResourceMonitor()
        assert monitor._nvml_handle is None
        assert monitor._has_cuda is False

    def test_sample_nvml_failure_propagates(self) -> None:
        """NVML query failures during sampling propagate immediately."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "fake_handle"
        mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = RuntimeError(
            "GPU query failed"
        )

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024.0 * 1024),
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        ):
            monitor = ResourceMonitor()

        # Now sample -- the NVML query should raise
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            pytest.raises(RuntimeError, match="GPU query failed"),
        ):
            monitor.sample()


class TestResourceMonitorWithMockedGPU:
    """Full GPU path with mocked NVML and CUDA."""

    def test_sample_collects_gpu_metrics(self) -> None:
        """When CUDA + NVML work, sample collects all GPU metrics."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "fake_handle"
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(
            gpu=75.0
        )
        mock_pynvml.NVML_TEMPERATURE_GPU = 0
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150_000  # 150W in mW

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=512.0 * 1024 * 1024),
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        ):
            monitor = ResourceMonitor()
            monitor.sample()
            monitor.sample()
            result = monitor.summarize()

        assert result["gpu_util_avg"] == 75.0
        assert result["gpu_util_peak"] == 75.0
        assert result["gpu_temp_avg"] == 65.0
        assert result["gpu_temp_peak"] == 65.0
        assert result["gpu_power_avg"] == 150.0
        assert result["gpu_power_peak"] == 150.0
        assert result["gpu_mem_mb_avg"] == 512.0
        assert result["gpu_mem_mb_peak"] == 512.0

    def test_reset_clears_gpu_samples(self) -> None:
        """Reset should clear GPU metric samples as well."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "fake_handle"
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(
            gpu=50.0
        )
        mock_pynvml.NVML_TEMPERATURE_GPU = 0
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100_000

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=256.0 * 1024 * 1024),
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        ):
            monitor = ResourceMonitor()
            monitor.sample()
            assert monitor.summarize() != {}
            monitor.reset()
            assert monitor.summarize() == {}
