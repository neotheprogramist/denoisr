import pytest
import torch

from denoisr_chess.nn.value_head import ChessValueHead

from conftest import SMALL_D_S


class TestChessValueHead:
    @pytest.fixture
    def head(self, device: torch.device) -> ChessValueHead:
        return ChessValueHead(d_s=SMALL_D_S).to(device)

    def test_forward_output_shape(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl_logits, ply = head(small_latent)
        assert wdl_logits.shape == (2, 3)

    def test_ply_output_shape(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        _, ply = head(small_latent)
        assert ply.shape == (2, 1)

    def test_forward_returns_finite_logits(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl_logits, _ = head(small_latent)
        assert torch.isfinite(wdl_logits).all()

    def test_forward_logits_not_constrained_to_unit(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        """Logits from forward() are raw -- not necessarily in [0, 1]."""
        wdl_logits, _ = head(small_latent)
        assert not ((wdl_logits >= 0).all() and (wdl_logits <= 1).all())

    def test_infer_wdl_sums_to_one(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, _ = head.infer(small_latent)
        sums = wdl.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_infer_wdl_in_zero_one(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, _ = head.infer(small_latent)
        assert (wdl >= 0).all()
        assert (wdl <= 1).all()

    def test_ply_non_negative(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        _, ply = head(small_latent)
        assert (ply >= 0).all()

    def test_gradient_flows(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl_logits, ply = head(small_latent)
        (wdl_logits.sum() + ply.sum()).backward()
        for p in head.parameters():
            assert p.grad is not None

    def test_has_attention_pooling(self, head: ChessValueHead) -> None:
        """Value head uses learned attention pooling query."""
        assert hasattr(head, "pool_query")
        assert head.pool_query.shape == (1, 1, SMALL_D_S)
