import pytest
import torch

from denoisr.nn.value_head import ChessValueHead

from conftest import SMALL_D_S


class TestChessValueHead:
    @pytest.fixture
    def head(self, device: torch.device) -> ChessValueHead:
        return ChessValueHead(d_s=SMALL_D_S).to(device)

    def test_wdl_output_shape(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, ply = head(small_latent)
        assert wdl.shape == (2, 3)

    def test_ply_output_shape(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, ply = head(small_latent)
        assert ply.shape == (2, 1)

    def test_wdl_sums_to_one(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, _ = head(small_latent)
        sums = wdl.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_wdl_in_zero_one(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, _ = head(small_latent)
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
        wdl, ply = head(small_latent)
        (wdl.sum() + ply.sum()).backward()
        for p in head.parameters():
            assert p.grad is not None
