"""Tests for ResourceMonitor system metrics collection."""

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
