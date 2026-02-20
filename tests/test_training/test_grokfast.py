import torch
from torch import nn

from denoisr.training.grokfast import GrokfastFilter


class TestGrokfastFilter:
    def test_first_apply_initializes_ema(self) -> None:
        model = nn.Linear(10, 5)
        model.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        gf.apply(model)
        assert len(gf.grads) > 0

    def test_amplifies_gradients(self) -> None:
        model = nn.Linear(10, 5, bias=False)
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        gf.apply(model)
        model.zero_grad()
        loss = model(x).sum()
        loss.backward()
        grad_before = model.weight.grad.clone()
        gf.apply(model)
        grad_after = model.weight.grad
        assert grad_after.norm() > grad_before.norm()

    def test_lamb_zero_means_no_amplification(self) -> None:
        model = nn.Linear(10, 5, bias=False)
        gf = GrokfastFilter(alpha=0.98, lamb=0.0)
        for _ in range(2):
            model.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            grad_before = model.weight.grad.clone()
            gf.apply(model)
        torch.testing.assert_close(model.weight.grad, grad_before)

    def test_skips_params_without_grad(self) -> None:
        model = nn.Linear(10, 5)
        model.weight.requires_grad_(False)
        model.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        gf.apply(model)  # Should not error
        assert "weight" not in str(list(gf.grads.keys()))
