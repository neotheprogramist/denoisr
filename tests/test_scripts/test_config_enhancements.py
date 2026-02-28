"""Tests for new training enhancement config fields."""

from denoisr.scripts.config import ModelConfig, TrainingConfig


class TestEnhancementConfigDefaults:
    def test_model_config_has_dropout(self) -> None:
        cfg = ModelConfig()
        assert cfg.dropout == 0.0

    def test_model_config_has_drop_path_rate(self) -> None:
        cfg = ModelConfig()
        assert cfg.drop_path_rate == 0.0

    def test_training_config_has_use_onecycle(self) -> None:
        cfg = TrainingConfig()
        assert cfg.use_onecycle is False

    def test_training_config_has_onecycle_pct_start(self) -> None:
        cfg = TrainingConfig()
        assert cfg.onecycle_pct_start == 0.3

    def test_training_config_has_gradient_accumulation_steps(self) -> None:
        cfg = TrainingConfig()
        assert cfg.gradient_accumulation_steps == 1

    def test_training_config_has_label_smoothing(self) -> None:
        cfg = TrainingConfig()
        assert cfg.label_smoothing == 0.0

    def test_training_config_has_value_noise_prob(self) -> None:
        cfg = TrainingConfig()
        assert cfg.value_noise_prob == 0.0

    def test_training_config_has_value_noise_scale(self) -> None:
        cfg = TrainingConfig()
        assert cfg.value_noise_scale == 0.02

    def test_training_config_has_policy_temp_augment_prob(self) -> None:
        cfg = TrainingConfig()
        assert cfg.policy_temp_augment_prob == 0.0
