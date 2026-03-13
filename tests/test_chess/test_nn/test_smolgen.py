import pytest
import torch

from denoisr_chess.nn.smolgen import SmolgenBias

from conftest import SMALL_D_S, SMALL_NUM_HEADS


class TestSmolgenBias:
    @pytest.fixture
    def smolgen(self, device: torch.device) -> SmolgenBias:
        return SmolgenBias(d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS).to(device)

    def test_output_shape(
        self, smolgen: SmolgenBias, small_latent: torch.Tensor
    ) -> None:
        out = smolgen(small_latent)
        assert out.shape == (2, SMALL_NUM_HEADS, 64, 64)

    def test_content_dependent(
        self, smolgen: SmolgenBias, device: torch.device
    ) -> None:
        smolgen.eval()
        x1 = torch.randn(1, 64, SMALL_D_S, device=device)
        x2 = torch.randn(1, 64, SMALL_D_S, device=device)
        assert not torch.allclose(smolgen(x1), smolgen(x2))

    def test_gradient_flows(
        self, smolgen: SmolgenBias, small_latent: torch.Tensor
    ) -> None:
        out = smolgen(small_latent)
        out.sum().backward()
        for p in smolgen.parameters():
            assert p.grad is not None

    def test_no_nan(self, smolgen: SmolgenBias, small_latent: torch.Tensor) -> None:
        out = smolgen(small_latent)
        assert not torch.isnan(out).any()
