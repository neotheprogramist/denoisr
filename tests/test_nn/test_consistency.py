import pytest
import torch

from denoisr.nn.consistency import ChessConsistencyProjector

from conftest import SMALL_D_S


class TestChessConsistencyProjector:
    @pytest.fixture
    def proj(self, device: torch.device) -> ChessConsistencyProjector:
        return ChessConsistencyProjector(d_s=SMALL_D_S, proj_dim=32).to(device)

    def test_output_shape(
        self, proj: ChessConsistencyProjector, small_latent: torch.Tensor
    ) -> None:
        out = proj(small_latent)
        assert out.shape == (2, 32)

    def test_cosine_similarity_defined(
        self, proj: ChessConsistencyProjector, device: torch.device
    ) -> None:
        x1 = torch.randn(2, 64, SMALL_D_S, device=device)
        x2 = torch.randn(2, 64, SMALL_D_S, device=device)
        p1 = proj(x1)
        p2 = proj(x2)
        cos_sim = torch.nn.functional.cosine_similarity(p1, p2)
        assert cos_sim.shape == (2,)
        assert ((cos_sim >= -1.0) & (cos_sim <= 1.0)).all()

    def test_gradient_flows(
        self, proj: ChessConsistencyProjector, small_latent: torch.Tensor
    ) -> None:
        out = proj(small_latent)
        out.sum().backward()
        for p in proj.parameters():
            assert p.grad is not None

    def test_stop_gradient_target(
        self, proj: ChessConsistencyProjector, device: torch.device
    ) -> None:
        x_pred = torch.randn(1, 64, SMALL_D_S, device=device, requires_grad=True)
        x_target = torch.randn(1, 64, SMALL_D_S, device=device, requires_grad=True)
        p_pred = proj(x_pred)
        with torch.no_grad():
            p_target = proj(x_target)
        loss = -torch.nn.functional.cosine_similarity(p_pred, p_target).mean()
        loss.backward()
        assert x_pred.grad is not None
        assert x_target.grad is None
