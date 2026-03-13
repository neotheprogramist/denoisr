import pytest
import torch

from denoisr_chess.nn.encoder import ChessEncoder

from conftest import SMALL_D_S


class TestChessEncoder:
    @pytest.fixture
    def encoder(self, device: torch.device) -> ChessEncoder:
        return ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)

    def test_output_shape(
        self, encoder: ChessEncoder, small_board_tensor: torch.Tensor
    ) -> None:
        out = encoder(small_board_tensor)
        assert out.shape == (2, 64, SMALL_D_S)

    def test_single_batch(self, encoder: ChessEncoder, device: torch.device) -> None:
        x = torch.randn(1, 12, 8, 8, device=device)
        out = encoder(x)
        assert out.shape == (1, 64, SMALL_D_S)

    def test_gradient_flows(
        self, encoder: ChessEncoder, small_board_tensor: torch.Tensor
    ) -> None:
        out = encoder(small_board_tensor)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            assert p.grad is not None
            assert not torch.all(p.grad == 0)

    def test_deterministic(
        self, encoder: ChessEncoder, small_board_tensor: torch.Tensor
    ) -> None:
        encoder.eval()
        out1 = encoder(small_board_tensor)
        out2 = encoder(small_board_tensor)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(
        self, encoder: ChessEncoder, device: torch.device
    ) -> None:
        x1 = torch.randn(1, 12, 8, 8, device=device)
        x2 = torch.randn(1, 12, 8, 8, device=device)
        encoder.eval()
        assert not torch.allclose(encoder(x1), encoder(x2))

    def test_no_nan(
        self, encoder: ChessEncoder, small_board_tensor: torch.Tensor
    ) -> None:
        out = encoder(small_board_tensor)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_122_plane_input(self, device: torch.device) -> None:
        """Encoder should accept 122-plane input from ExtendedBoardEncoder."""
        enc = ChessEncoder(num_planes=122, d_s=SMALL_D_S).to(device)
        x = torch.randn(2, 122, 8, 8, device=device)
        out = enc(x)
        assert out.shape == (2, 64, SMALL_D_S)
