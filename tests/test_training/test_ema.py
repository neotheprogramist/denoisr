import torch
from torch import nn

from denoisr.training.ema import ModelEMA


class TestModelEMA:
    def test_ema_update_moves_toward_params(self) -> None:
        """EMA shadow weights converge toward model weights."""
        model = nn.Linear(4, 4, bias=False)
        ema = ModelEMA({"model": model}, decay=0.9)

        # Change model weights
        with torch.no_grad():
            model.weight.fill_(10.0)

        # After many updates, EMA should approach 10.0
        for _ in range(100):
            ema.update()

        shadow = ema.state_dicts()["model"]["weight"]
        assert torch.allclose(shadow, torch.full_like(shadow, 10.0), atol=0.1)

    def test_ema_apply_context_manager(self) -> None:
        """apply() swaps in EMA weights, restores originals on exit."""
        model = nn.Linear(4, 4, bias=False)
        ema = ModelEMA({"model": model}, decay=0.9)

        # Change model
        with torch.no_grad():
            model.weight.fill_(100.0)
        ema.update()

        # Inside context: EMA weights (mix of original and 100)
        with ema.apply():
            assert not torch.equal(
                model.weight.data, torch.full_like(model.weight, 100.0)
            )

        # After context: back to 100.0
        assert torch.allclose(model.weight.data, torch.full_like(model.weight, 100.0))

    def test_ema_multiple_modules(self) -> None:
        """EMA works with multiple modules."""
        enc = nn.Linear(4, 4)
        head = nn.Linear(4, 2)
        ema = ModelEMA({"encoder": enc, "head": head}, decay=0.999)

        ema.update()
        state = ema.state_dicts()
        assert "encoder" in state
        assert "head" in state

    def test_ema_state_dict_round_trip(self) -> None:
        """state_dicts() output can be loaded back."""
        model = nn.Linear(4, 4, bias=False)
        ema = ModelEMA({"model": model}, decay=0.9)

        with torch.no_grad():
            model.weight.fill_(5.0)
        for _ in range(50):
            ema.update()

        saved = ema.state_dicts()

        # Create fresh EMA and load
        ema2 = ModelEMA({"model": model}, decay=0.9)
        ema2.load_state_dicts(saved)

        assert torch.equal(
            ema.state_dicts()["model"]["weight"],
            ema2.state_dicts()["model"]["weight"],
        )

    def test_ema_decay_zero_copies_immediately(self) -> None:
        """With decay=0, EMA immediately matches current weights."""
        model = nn.Linear(4, 4, bias=False)
        ema = ModelEMA({"model": model}, decay=0.0)

        with torch.no_grad():
            model.weight.fill_(42.0)
        ema.update()

        assert torch.allclose(
            ema.state_dicts()["model"]["weight"],
            torch.full_like(model.weight, 42.0),
        )

    def test_ema_decay_one_ignores_updates(self) -> None:
        """With decay=1.0, shadow weights never change."""
        model = nn.Linear(4, 4, bias=False)
        initial_weight = model.weight.data.clone()
        ema = ModelEMA({"model": model}, decay=1.0)

        with torch.no_grad():
            model.weight.fill_(999.0)
        for _ in range(50):
            ema.update()

        assert torch.allclose(ema.state_dicts()["model"]["weight"], initial_weight)

    def test_ema_steps_increment(self) -> None:
        """Internal step counter increments on each update."""
        model = nn.Linear(4, 4)
        ema = ModelEMA({"model": model}, decay=0.999)

        assert ema._steps == 0
        ema.update()
        assert ema._steps == 1
        ema.update()
        assert ema._steps == 2

    def test_ema_decay_property(self) -> None:
        """The decay property returns the configured value."""
        model = nn.Linear(4, 4)
        ema = ModelEMA({"model": model}, decay=0.995)
        assert ema.decay == 0.995

    def test_ema_handles_batchnorm(self) -> None:
        """EMA correctly handles BatchNorm's non-float num_batches_tracked."""
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        ema = ModelEMA({"model": model}, decay=0.9)

        # Run forward pass to update batchnorm stats
        x = torch.randn(8, 4)
        model(x)

        # Should not error on non-float tensors
        ema.update()
        state = ema.state_dicts()
        assert "model" in state

    def test_ema_load_ignores_unknown_keys(self) -> None:
        """load_state_dicts silently ignores keys not in shadow."""
        model = nn.Linear(4, 4, bias=False)
        ema = ModelEMA({"model": model}, decay=0.9)

        # Load with extra key that doesn't exist
        ema.load_state_dicts(
            {"model": model.state_dict(), "nonexistent": model.state_dict()}
        )
        # Should not error; only "model" key is loaded
        assert "model" in ema.state_dicts()
