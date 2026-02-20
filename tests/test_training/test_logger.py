"""Tests for TrainingLogger TensorBoard integration."""

import logging
import pathlib

import pytest

from denoisr.training.logger import TrainingLogger


class TestTrainingLogger:
    def test_creates_log_directory(self, tmp_path: pathlib.Path) -> None:
        """Logger should create run directory inside log_dir."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test_run")
        logger.close()
        assert (tmp_path / "test_run").is_dir()

    def test_auto_generates_run_name(self, tmp_path: pathlib.Path) -> None:
        """Without run_name, logger should create a timestamped directory."""
        logger = TrainingLogger(log_dir=tmp_path)
        logger.close()
        dirs = list(tmp_path.iterdir())
        assert len(dirs) == 1
        # Timestamped name should match YYYY-MM-DD_HH-MM-SS pattern
        assert len(dirs[0].name) == 19

    def test_log_train_step_writes_scalars(self, tmp_path: pathlib.Path) -> None:
        """log_train_step should write loss scalars without error."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        breakdown = {"policy": 1.5, "value": 0.8, "total": 2.3, "grad_norm": 0.42}
        logger.log_train_step(step=0, loss=2.3, breakdown=breakdown)
        logger.close()
        event_files = list((tmp_path / "test").glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_log_epoch_writes_scalars_and_text(self, tmp_path: pathlib.Path) -> None:
        """log_epoch should write scalars and a text log line."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch(epoch=1, avg_loss=2.0, top1=0.05, top5=0.15, lr=1e-4)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "epoch=1" in text
        assert "avg_loss=2.000000" in text
        assert "top1=0.0500" in text

    def test_log_epoch_timing(self, tmp_path: pathlib.Path) -> None:
        """log_epoch_timing should write timing scalars without error."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_timing(epoch=1, duration_s=42.5, samples_per_sec=1500.0)
        logger.close()

    def test_log_gpu_no_error_on_cpu(self, tmp_path: pathlib.Path) -> None:
        """log_gpu should be a no-op on CPU without raising."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_gpu(step=0)
        logger.close()

    def test_log_diffusion_writes_text(self, tmp_path: pathlib.Path) -> None:
        """log_diffusion should write diffusion metrics to text log."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_diffusion(epoch=1, avg_loss=0.5, curriculum_steps=25)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "diffusion_loss=0.500000" in text
        assert "curriculum_steps=25" in text

    def test_log_hparams_writes_text_file(self, tmp_path: pathlib.Path) -> None:
        """log_hparams should write hparams.txt alongside TensorBoard data."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        hparams = {"lr": 1e-4, "batch_size": 64, "d_s": 256}
        metrics = {"best_top1": 0.35}
        logger.log_hparams(hparams, metrics)
        logger.close()
        text = (tmp_path / "test" / "hparams.txt").read_text()
        assert "lr=0.0001" in text
        assert "batch_size=64" in text
        assert "d_s=256" in text

    def test_log_resource_metrics_writes_scalars_and_text(
        self, tmp_path: pathlib.Path
    ) -> None:
        """log_resource_metrics should write resource metrics to TensorBoard and text."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        metrics = {
            "cpu_percent_avg": 45.2,
            "cpu_percent_peak": 98.1,
            "ram_mb_avg": 2341.0,
            "ram_mb_peak": 2567.0,
        }
        logger.log_resource_metrics(epoch=0, metrics=metrics)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "cpu_avg=45.2%" in text
        assert "ram_peak=2567mb" in text

    def test_log_training_dynamics(self, tmp_path: pathlib.Path) -> None:
        """log_training_dynamics should compute and write dynamics stats."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        losses = [1.0, 2.0, 3.0, 4.0, 5.0]
        grad_norms = [0.1, 0.5, 0.3, 0.8, 0.2]
        logger.log_training_dynamics(epoch=0, losses=losses, grad_norms=grad_norms)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "grad_norm_avg=" in text
        assert "grad_norm_peak=0.800" in text
        assert "loss_std=" in text

    def test_log_pipeline_timing(self, tmp_path: pathlib.Path) -> None:
        """log_pipeline_timing should write pipeline efficiency metrics."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_pipeline_timing(epoch=0, data_time=2.0, compute_time=8.0)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "data_wait_frac=0.20" in text
        assert "compute_frac=0.80" in text

    def test_log_pipeline_timing_zero_total(self, tmp_path: pathlib.Path) -> None:
        """log_pipeline_timing should handle zero total time gracefully."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_pipeline_timing(epoch=0, data_time=0.0, compute_time=0.0)
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "data_wait_frac=0.00" in text

    def test_log_epoch_summary_emits_via_logging(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log_epoch_summary should emit key=value pairs via logging module."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        with caplog.at_level(logging.INFO, logger="denoisr.training.logger"):
            logger.log_epoch_summary({"epoch": "0", "loss": "2.13", "top1": "5.2%"})
        logger.close()
        assert "epoch=0" in caplog.text
        assert "loss=2.13" in caplog.text
        assert "top1=5.2%" in caplog.text

    def test_context_manager(self, tmp_path: pathlib.Path) -> None:
        """Logger should support with-statement for automatic cleanup."""
        with TrainingLogger(log_dir=tmp_path, run_name="ctx") as logger:
            logger.log_train_step(step=0, loss=1.0, breakdown={"total": 1.0})
        assert (tmp_path / "ctx").is_dir()
