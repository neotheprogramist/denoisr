"""Tests for TrainingLogger human-readable log integration."""

import logging
import pathlib

import pytest

from denoisr_chess.training.logger import TrainingLogger


@pytest.fixture
def log_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Route root logs to a deterministic per-test file."""
    path = tmp_path / "denoisr.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    yield path
    logging.shutdown()
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)


def _read_log(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


class TestTrainingLogger:
    def test_requires_root_logging_configuration(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        logging.shutdown()
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        with pytest.raises(RuntimeError, match="Root logging is not configured"):
            TrainingLogger(log_dir=tmp_path, run_name="test")

    def test_does_not_create_run_directory(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        """Logger should keep everything in denoisr.log (no logs/<run>/ dirs)."""
        logger = TrainingLogger(log_dir=tmp_path, run_name="test_run")
        logger.close()
        assert not (tmp_path / "test_run").exists()
        assert log_path.exists()

    def test_auto_generates_timestamp_run_name(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        """Without run_name, logger should still create a timestamp label."""
        logger = TrainingLogger(log_dir=tmp_path)
        run_name = logger._run_name
        logger.close()
        assert len(run_name) == 19  # YYYY-MM-DD_HH-MM-SS
        assert log_path.exists()

    def test_log_epoch_line_phase1(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
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
        text = _read_log(log_path)
        assert "E=1/100" in text
        assert "loss=2.0000" in text
        assert "top1=5.0%" in text
        assert "top5=15.0%" in text
        assert "lr=1.0e-04" in text
        assert "sps=1500" in text
        assert "t=42.5s" in text
        assert "data=20%" in text

    def test_log_epoch_line_phase2(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
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
        text = _read_log(log_path)
        assert "E=5/200" in text
        assert "diff=-0.0800" in text
        assert "sps=320" in text

    def test_log_epoch_line_with_resources(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
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
        text = _read_log(log_path)
        assert "cpu=101%/134%" in text
        assert "ram=12334mb" in text
        assert "gpu=88%/90% 1039mb 57C 136W" in text

    def test_log_epoch_line_with_overflows(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
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
        text = _read_log(log_path)
        assert "ovf=3" in text

    def test_log_epoch_line_no_overflow_when_zero(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
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
        text = _read_log(log_path)
        assert "ovf=" not in text

    def test_log_epoch_line_empty_grad_norms(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
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
        text = _read_log(log_path)
        assert "gnorm=" not in text

    def test_log_hparams_writes_human_line(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        hparams = {"lr": 1e-4, "batch_size": 64, "d_s": 256}
        metrics = {"best_top1": 0.35}
        logger.log_hparams(hparams, metrics)
        logger.close()
        text = _read_log(log_path)
        assert "HPARAMS" in text
        assert "batch_size=64" in text
        assert "d_s=256" in text
        assert "lr=0.0001" in text

    def test_log_epoch_summary_emits_via_logging(
        self,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
        log_path: pathlib.Path,
    ) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        with caplog.at_level(logging.INFO, logger="denoisr.metrics.test"):
            logger.log_epoch_summary({"epoch": "0", "loss": "2.13", "top1": "5.2%"})
        logger.close()
        assert "epoch=0" in caplog.text
        assert "loss=2.13" in caplog.text
        assert "top1=5.2%" in caplog.text
        assert "epoch=0" in _read_log(log_path)

    def test_log_epoch_summary_writes_to_file(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_epoch_summary({"epoch": "0", "loss": "2.13"})
        logger.close()
        text = _read_log(log_path)
        assert "epoch=0" in text
        assert "loss=2.13" in text

    def test_context_manager(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        with TrainingLogger(log_dir=tmp_path, run_name="ctx") as logger:
            logger.log_epoch_summary({"epoch": "0", "loss": "1.0"})
        assert not (tmp_path / "ctx").exists()
        assert "epoch=0" in _read_log(log_path)


class TestGrokLogging:
    def test_step_level_grok_metrics_do_not_bloat_logs(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        metrics = {
            "grok/weight_norm_total": 42.0,
            "grok/erank/layer_0": 15.3,
            "grok/state": 0.0,
        }
        logger.log_grok_metrics(step=100, metrics=metrics)
        logger.close()
        text = _read_log(log_path)
        assert "GROK-EPOCH" not in text

    def test_epoch_level_grok_metrics_write_summary_and_warning(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        metrics = {
            "grok/holdout/random/accuracy": 0.26,
            "grok/holdout/game_level/accuracy": 0.22,
            "grok/loss_gap": -0.15,
            "grok/state": 1.0,
        }
        logger.log_grok_metrics(step=12, metrics=metrics)
        logger.close()
        text = _read_log(log_path)
        assert "GROK-EPOCH" in text
        assert "state=ONSET_DETECTED" in text
        assert "random_acc=26.00%" in text
        assert "GROKKING state entered" in text

    def test_log_grok_state_transition_writes_warning(
        self, tmp_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_grok_state_transition(
            step=5000,
            old_state="BASELINE",
            new_state="ONSET_DETECTED",
            trigger="weight_norm decreased 6.2%",
        )
        logger.close()
        text = _read_log(log_path)
        assert "GROKKING transition" in text
        assert "ONSET_DETECTED" in text
