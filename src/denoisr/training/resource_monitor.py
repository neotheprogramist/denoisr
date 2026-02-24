"""System resource monitor for training metrics.

Collects CPU/RAM/GPU samples during training and reports
per-epoch averages and peaks. Gracefully degrades when
subsystems are unavailable (no CUDA, no pynvml).
"""

from __future__ import annotations

from statistics import mean

import psutil
import torch


def _try_init_nvml() -> bool:
    """Attempt to initialize NVML. Returns True on success."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return True
    except ImportError, OSError:
        return False


def _get_nvml_handle() -> object | None:
    """Get NVML handle for device 0, or None."""
    try:
        import pynvml

        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except ImportError, OSError:
        return None


class ResourceMonitor:
    """Samples system resources and computes per-epoch avg/peak.

    Usage::

        monitor = ResourceMonitor()

        for epoch in range(epochs):
            monitor.reset()
            for step in range(steps):
                if step % 100 == 0:
                    monitor.sample()
            metrics = monitor.summarize()
    """

    def __init__(self) -> None:
        self._process = psutil.Process()
        # Prime cpu_percent (first call always returns 0.0)
        self._process.cpu_percent()

        self._has_cuda = torch.cuda.is_available()
        self._has_nvml = self._has_cuda and _try_init_nvml()
        self._nvml_handle = _get_nvml_handle() if self._has_nvml else None

        self._cpu_samples: list[float] = []
        self._ram_samples: list[float] = []
        self._gpu_util_samples: list[float] = []
        self._gpu_mem_samples: list[float] = []
        self._gpu_temp_samples: list[float] = []
        self._gpu_power_samples: list[float] = []

    def reset(self) -> None:
        """Clear all accumulated samples for a new epoch."""
        self._cpu_samples.clear()
        self._ram_samples.clear()
        self._gpu_util_samples.clear()
        self._gpu_mem_samples.clear()
        self._gpu_temp_samples.clear()
        self._gpu_power_samples.clear()

    def sample(self) -> None:
        """Take a snapshot of all available resource metrics."""
        self._cpu_samples.append(self._process.cpu_percent())
        self._ram_samples.append(self._process.memory_info().rss / (1024 * 1024))

        if self._has_cuda:
            self._gpu_mem_samples.append(torch.cuda.memory_allocated() / (1024 * 1024))

        if self._has_nvml and self._nvml_handle is not None:
            self._sample_nvml()

    def _sample_nvml(self) -> None:
        """Sample GPU utilization, temperature, and power via NVML."""
        try:
            import pynvml

            rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            self._gpu_util_samples.append(float(rates.gpu))

            temp = pynvml.nvmlDeviceGetTemperature(
                self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            self._gpu_temp_samples.append(float(temp))

            power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
            self._gpu_power_samples.append(power / 1000.0)  # mW -> W
        except ImportError, OSError, RuntimeError:
            pass

    def summarize(self) -> dict[str, float]:
        """Compute avg/peak for all collected metrics.

        Returns empty dict if no samples were collected.
        """
        if not self._cpu_samples:
            return {}

        result: dict[str, float] = {
            "cpu_percent_avg": mean(self._cpu_samples),
            "cpu_percent_peak": max(self._cpu_samples),
            "ram_mb_avg": mean(self._ram_samples),
            "ram_mb_peak": max(self._ram_samples),
        }

        if self._gpu_mem_samples:
            result["gpu_mem_mb_avg"] = mean(self._gpu_mem_samples)
            result["gpu_mem_mb_peak"] = max(self._gpu_mem_samples)

        if self._gpu_util_samples:
            result["gpu_util_avg"] = mean(self._gpu_util_samples)
            result["gpu_util_peak"] = max(self._gpu_util_samples)

        if self._gpu_temp_samples:
            result["gpu_temp_avg"] = mean(self._gpu_temp_samples)
            result["gpu_temp_peak"] = max(self._gpu_temp_samples)

        if self._gpu_power_samples:
            result["gpu_power_avg"] = mean(self._gpu_power_samples)
            result["gpu_power_peak"] = max(self._gpu_power_samples)

        return result
