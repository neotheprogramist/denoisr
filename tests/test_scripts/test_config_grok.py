import argparse

import pytest

from denoisr.scripts.config import (
    TrainingConfig,
    add_training_args,
    training_config_from_args,
)


def _set_required_training_env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "DENOISR_TRAIN_MAX_GRAD_NORM": "5.0",
        "DENOISR_TRAIN_WEIGHT_DECAY": "0.0001",
        "DENOISR_TRAIN_ENCODER_LR_MULTIPLIER": "1.0",
        "DENOISR_TRAIN_MIN_LR": "0.000001",
        "DENOISR_TRAIN_WARMUP_EPOCHS": "10",
        "DENOISR_TRAIN_WARM_RESTARTS": "1",
        "DENOISR_TRAIN_THREAT_WEIGHT": "0.1",
        "DENOISR_TRAIN_POLICY_WEIGHT": "2.0",
        "DENOISR_TRAIN_VALUE_WEIGHT": "0.5",
        "DENOISR_TRAIN_CONSISTENCY_WEIGHT": "1.0",
        "DENOISR_TRAIN_DIFFUSION_WEIGHT": "1.0",
        "DENOISR_TRAIN_REWARD_WEIGHT": "1.0",
        "DENOISR_TRAIN_PLY_WEIGHT": "0.1",
        "DENOISR_TRAIN_ILLEGAL_PENALTY_WEIGHT": "0.01",
        "DENOISR_TRAIN_HARMONY_DREAM": "1",
        "DENOISR_TRAIN_HARMONY_EMA_DECAY": "0.99",
        "DENOISR_TRAIN_CURRICULUM_INITIAL_FRACTION": "0.25",
        "DENOISR_TRAIN_CURRICULUM_GROWTH": "1.02",
        "DENOISR_WORKERS": "64",
        "DENOISR_TQDM": "0",
        "DENOISR_PHASE1_GATE": "0.50",
        "DENOISR_PHASE2_GATE": "5.0",
        "DENOISR_GROK_TRACKING": "1",
        "DENOISR_GROK_ERANK_FREQ": "1000",
        "DENOISR_GROK_SPECTRAL_FREQ": "5000",
        "DENOISR_GROK_ONSET_THRESHOLD": "0.95",
        "DENOISR_GROKFAST": "1",
        "DENOISR_GROKFAST_ALPHA": "0.98",
        "DENOISR_GROKFAST_LAMB": "2.0",
        "DENOISR_EMA_DECAY": "0.999",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


class TestGrokConfig:
    def test_default_grok_tracking_on(self) -> None:
        cfg = TrainingConfig()
        assert cfg.grok_tracking is True

    def test_default_grokfast_on(self) -> None:
        cfg = TrainingConfig()
        assert cfg.grokfast is True

    def test_grok_fields_exist(self) -> None:
        cfg = TrainingConfig(
            grok_tracking=True,
            grok_erank_freq=500,
            grok_spectral_freq=2000,
            grok_onset_threshold=0.93,
            grokfast=True,
            grokfast_alpha=0.95,
            grokfast_lamb=3.0,
        )
        assert cfg.grok_tracking is True
        assert cfg.grok_erank_freq == 500
        assert cfg.grokfast_lamb == 3.0

    def test_cli_flags_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_training_env(monkeypatch)
        parser = argparse.ArgumentParser()
        add_training_args(parser)
        args = parser.parse_args(["--grok-tracking", "--grokfast"])
        assert args.grok_tracking is True
        assert args.grokfast is True

    def test_training_config_from_args_includes_grok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_required_training_env(monkeypatch)
        parser = argparse.ArgumentParser()
        add_training_args(parser)
        args = parser.parse_args(["--grok-tracking", "--grok-erank-freq", "500"])
        cfg = training_config_from_args(args)
        assert cfg.grok_tracking is True
        assert cfg.grok_erank_freq == 500
