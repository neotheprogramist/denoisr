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

    def test_log_epoch_line_phase1(self, tmp_path: pathlib.Path) -> None:
        """log_epoch_line should write a compact single line with Phase 1 metrics."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_line(
            epoch=1,
            total_epochs=100,
            losses={"loss": 2.0, "pol": 1.5, "val": 0.5},
            lr=1e-4,
            grad_norms=[0.1, 0.5, 0.3, 0.8, 0.2],
            samples_per_sec=1500.0,
            duration_s=42.5,
            accuracy={"top1": 5.0, "top5": 15.0},
            data_pct=20.0,
            phase="phase1",
        )
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "E=1/100" in text
        assert "loss=2.0000" in text
        assert "top1=5.0%" in text
        assert "top5=15.0%" in text
        assert "lr=1.0e-04" in text
        assert "sps=1500" in text
        assert "t=42.5s" in text
        assert "data=20%" in text

    def test_log_epoch_line_phase2(self, tmp_path: pathlib.Path) -> None:
        """log_epoch_line should handle Phase 2 format with diffusion losses."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_line(
            epoch=5,
            total_epochs=200,
            losses={
                "loss": 0.69,
                "pol": 0.45,
                "val": 0.12,
                "diff": -0.08,
                "cons": 0.34,
                "state": 0.01,
                "rew": 0.02,
            },
            lr=3e-4,
            grad_norms=[1.0, 2.0, 3.0],
            samples_per_sec=320.0,
            duration_s=148.7,
            data_pct=0.0,
            phase="phase2",
        )
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "E=5/200" in text
        assert "diff=-0.0800" in text
        assert "sps=320" in text

    def test_log_epoch_line_with_resources(self, tmp_path: pathlib.Path) -> None:
        """log_epoch_line should include resource metrics when provided."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_line(
            epoch=1,
            total_epochs=10,
            losses={"loss": 1.0},
            lr=1e-3,
            grad_norms=[0.5],
            samples_per_sec=100.0,
            duration_s=10.0,
            resources={
                "cpu_pct": "101%",
                "cpu_max": "134%",
                "ram_mb": "12334",
                "gpu_util": "88%",
                "gpu_mem_pct": "90%",
                "gpu_mem_mb": "1039",
                "gpu_temp": "57",
                "gpu_power": "136",
            },
        )
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "cpu=101%/134%" in text
        assert "ram=12334mb" in text
        assert "gpu=88%/90% 1039mb 57C 136W" in text

    def test_log_epoch_line_with_overflows(self, tmp_path: pathlib.Path) -> None:
        """Overflows should appear in line only when > 0."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_line(
            epoch=1,
            total_epochs=10,
            losses={"loss": 1.0},
            lr=1e-3,
            grad_norms=[0.5],
            samples_per_sec=100.0,
            duration_s=10.0,
            overflows=3,
        )
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "ovf=3" in text

    def test_log_epoch_line_no_overflow_when_zero(self, tmp_path: pathlib.Path) -> None:
        """No overflow token when overflows=0."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_line(
            epoch=1,
            total_epochs=10,
            losses={"loss": 1.0},
            lr=1e-3,
            grad_norms=[0.5],
            samples_per_sec=100.0,
            duration_s=10.0,
            overflows=0,
        )
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "ovf=" not in text

    def test_log_epoch_line_empty_grad_norms(self, tmp_path: pathlib.Path) -> None:
        """Empty grad_norms should not produce gnorm= token."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_line(
            epoch=1,
            total_epochs=10,
            losses={"loss": 1.0},
            lr=1e-3,
            grad_norms=[],
            samples_per_sec=100.0,
            duration_s=10.0,
        )
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "gnorm=" not in text

    def test_log_gpu_no_error_on_cpu(self, tmp_path: pathlib.Path) -> None:
        """log_gpu should be a no-op on CPU without raising."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_gpu(step=0)
        logger.close()

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

    def test_log_epoch_summary_emits_via_logging(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log_epoch_summary should emit key=value pairs via metrics logger.

        The metrics logger has propagate=False so caplog (which hooks
        into the root logger) cannot capture records directly.  We
        install caplog's handler on the metrics logger for this test.
        """
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        metrics_log = logging.getLogger("denoisr.metrics.test")
        with caplog.at_level(logging.INFO, logger="denoisr.metrics.test"):
            metrics_log.addHandler(caplog.handler)
            try:
                logger.log_epoch_summary({"epoch": "0", "loss": "2.13", "top1": "5.2%"})
            finally:
                metrics_log.removeHandler(caplog.handler)
        logger.close()
        assert "epoch=0" in caplog.text
        assert "loss=2.13" in caplog.text
        assert "top1=5.2%" in caplog.text

    def test_log_epoch_summary_writes_to_file(self, tmp_path: pathlib.Path) -> None:
        """log_epoch_summary should also write to metrics.log file."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_summary({"epoch": "0", "loss": "2.13"})
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "epoch=0" in text
        assert "loss=2.13" in text

    def test_context_manager(self, tmp_path: pathlib.Path) -> None:
        """Logger should support with-statement for automatic cleanup."""
        with TrainingLogger(log_dir=tmp_path, run_name="ctx") as logger:
            logger.log_train_step(step=0, loss=1.0, breakdown={"total": 1.0})
        assert (tmp_path / "ctx").is_dir()


class TestGrokLogging:
    def test_log_grok_metrics_writes_scalars(self, tmp_path: pathlib.Path) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        metrics = {
            "grok/weight_norm_total": 42.0,
            "grok/erank/layer_0": 15.3,
            "grok/state": 0.0,
        }
        logger.log_grok_metrics(step=100, metrics=metrics)
        logger.close()
        event_files = list((tmp_path / "test").glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_log_grok_state_transition_writes_text(
        self, tmp_path: pathlib.Path
    ) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_grok_state_transition(
            step=5000,
            old_state="BASELINE",
            new_state="ONSET_DETECTED",
            trigger="weight_norm decreased 6.2%",
        )
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "GROKKING" in text
        assert "ONSET_DETECTED" in text
