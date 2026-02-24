import argparse

from denoisr.scripts.config import TrainingConfig, add_training_args, training_config_from_args


class TestGrokConfig:
    def test_default_grok_tracking_on(self) -> None:
        cfg = TrainingConfig()
        assert cfg.grok_tracking is True

    def test_default_grokfast_on(self) -> None:
        cfg = TrainingConfig()
        assert cfg.grokfast is True

    def test_default_compile_mode_on(self) -> None:
        cfg = TrainingConfig()
        assert cfg.compile_mode == "on"

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

    def test_cli_flags_registered(self) -> None:
        parser = argparse.ArgumentParser()
        add_training_args(parser)
        args = parser.parse_args(["--grok-tracking", "--grokfast"])
        assert args.grok_tracking is True
        assert args.grokfast is True

    def test_training_config_from_args_includes_grok(self) -> None:
        parser = argparse.ArgumentParser()
        add_training_args(parser)
        args = parser.parse_args(["--grok-tracking", "--grok-erank-freq", "500"])
        cfg = training_config_from_args(args)
        assert cfg.grok_tracking is True
        assert cfg.grok_erank_freq == 500

    def test_training_config_from_args_includes_compile_mode(self) -> None:
        parser = argparse.ArgumentParser()
        add_training_args(parser)
        args = parser.parse_args(["--compile", "off"])
        cfg = training_config_from_args(args)
        assert cfg.compile_mode == "off"
