import pytest
import torch

from denoisr.nn.policy_head import ChessPolicyHead

from conftest import SMALL_D_S


class TestChessPolicyHead:
    @pytest.fixture
    def head(self, device: torch.device) -> ChessPolicyHead:
        return ChessPolicyHead(d_s=SMALL_D_S).to(device)

    def test_output_shape(
        self, head: ChessPolicyHead, small_latent: torch.Tensor
    ) -> None:
        out = head(small_latent)
        assert out.shape == (2, 64, 64)

    def test_softmax_sums_to_one(
        self, head: ChessPolicyHead, small_latent: torch.Tensor
    ) -> None:
        logits = head(small_latent)
        probs = torch.softmax(logits.reshape(-1, 64 * 64), dim=-1)
        assert torch.allclose(
            probs.sum(dim=-1),
            torch.ones(probs.shape[0], device=probs.device),
        )

    def test_gradient_flows(
        self, head: ChessPolicyHead, small_latent: torch.Tensor
    ) -> None:
        out = head(small_latent)
        out.sum().backward()
        for p in head.parameters():
            assert p.grad is not None

    def test_no_nan(self, head: ChessPolicyHead, small_latent: torch.Tensor) -> None:
        out = head(small_latent)
        assert not torch.isnan(out).any()
