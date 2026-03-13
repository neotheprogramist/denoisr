"""System resource monitor for training metrics.

Collects CPU/RAM/GPU samples during training and reports
per-epoch averages and peaks. When CUDA is available, GPU
monitoring via NVML is required -- initialization failures
propagate immediately.
"""

from __future__ import annotations

import logging
from statistics import mean

import psutil
import torch

log = logging.getLogger(__name__)


class ResourceMonitor:
    """Samples system resources and computes per-epoch avg/peak.

    When CUDA is available, NVML *must* initialize successfully.
    Any NVML failure propagates immediately -- there is no silent
    degradation.

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
        if self._has_cuda:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self._nvml_handle = None

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
            self._sample_nvml()

    def _sample_nvml(self) -> None:
        """Sample GPU utilization, temperature, and power via NVML."""
        import pynvml

        rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        self._gpu_util_samples.append(float(rates.gpu))

        temp = pynvml.nvmlDeviceGetTemperature(
            self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
        )
        self._gpu_temp_samples.append(float(temp))

        power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
        self._gpu_power_samples.append(power / 1000.0)  # mW -> W

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
