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

    def test_nonfinite_gradients_do_not_poison_ema(self) -> None:
        model = nn.Linear(10, 5, bias=False)
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        gf.apply(model)

        model.weight.grad.fill_(float("inf"))
        gf.apply(model)
        assert "weight" not in gf.grads

        model.zero_grad()
        loss = model(x).sum()
        loss.backward()
        gf.apply(model)
        assert torch.isfinite(model.weight.grad).all()

    def test_reset_clears_ema_buffers(self) -> None:
        model = nn.Linear(10, 5)
        model.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        gf.apply(model)
        assert len(gf.grads) > 0
        gf.reset()
        assert gf.grads == {}

    def test_key_prefix_namespaces_ema_buffers(self) -> None:
        class _NormModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        encoder = _NormModule()
        value_head = _NormModule()
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)

        x = torch.randn(2, 4)
        encoder.zero_grad()
        value_head.zero_grad()
        encoder(x).sum().backward()
        value_head((x * 2.0)).sum().backward()

        gf.apply(encoder, key_prefix="encoder")
        gf.apply(value_head, key_prefix="value_head")

        assert "encoder.norm.weight" in gf.grads
        assert "value_head.norm.weight" in gf.grads
        before = gf.grads["value_head.norm.weight"].clone()
        gf.grads["encoder.norm.weight"].add_(1.0)
        torch.testing.assert_close(gf.grads["value_head.norm.weight"], before)
