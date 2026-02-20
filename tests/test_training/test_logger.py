"""Tests for TrainingLogger TensorBoard integration."""

import pathlib

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

    def test_log_epoch_writes_scalars(self, tmp_path: pathlib.Path) -> None:
        """log_epoch should write accuracy and lr scalars without error."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch(epoch=1, avg_loss=2.0, top1=0.05, top5=0.15, lr=1e-4)
        logger.close()

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

    def test_log_diffusion(self, tmp_path: pathlib.Path) -> None:
        """log_diffusion should write diffusion-specific scalars."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_diffusion(epoch=1, avg_loss=0.5, curriculum_steps=25)
        logger.close()

    def test_log_hparams(self, tmp_path: pathlib.Path) -> None:
        """log_hparams should write without error."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        hparams = {"lr": 1e-4, "batch_size": 64, "d_s": 256}
        metrics = {"best_top1": 0.35}
        logger.log_hparams(hparams, metrics)
        logger.close()

    def test_context_manager(self, tmp_path: pathlib.Path) -> None:
        """Logger should support with-statement for automatic cleanup."""
        with TrainingLogger(log_dir=tmp_path, run_name="ctx") as logger:
            logger.log_train_step(step=0, loss=1.0, breakdown={"total": 1.0})
        assert (tmp_path / "ctx").is_dir()
