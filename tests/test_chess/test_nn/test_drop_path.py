"""Tests for DropPath (stochastic depth) module."""

import torch

from denoisr_chess.nn.drop_path import DropPath


class TestDropPath:
    def test_zero_rate_is_identity(self) -> None:
        dp = DropPath(0.0)
        x = torch.randn(4, 64, 128)
        out = dp(x)
        assert torch.equal(out, x)

    def test_one_rate_drops_everything_in_training(self) -> None:
        dp = DropPath(1.0)
        dp.train()
        x = torch.randn(4, 64, 128)
        out = dp(x)
        assert torch.all(out == 0)

    def test_inference_mode_is_identity(self) -> None:
        dp = DropPath(0.5)
        dp.train(False)
        x = torch.randn(4, 64, 128)
        out = dp(x)
        assert torch.equal(out, x)

    def test_training_mode_scales_output(self) -> None:
        torch.manual_seed(42)
        dp = DropPath(0.5)
        dp.train()
        x = torch.ones(100, 64, 128)
        out = dp(x)
        kept = (out[:, 0, 0] != 0).sum().item()
        assert 30 < kept < 70

    def test_gradient_flows(self) -> None:
        dp = DropPath(0.3)
        dp.train()
        x = torch.randn(4, 64, 128, requires_grad=True)
        out = dp(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_output_shape_preserved(self) -> None:
        dp = DropPath(0.2)
        x = torch.randn(8, 64, 256)
        assert dp(x).shape == x.shape
