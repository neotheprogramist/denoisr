import chess
import torch

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.nn.encoder import ChessEncoder
from denoisr.training.augmentation import flip_board, flip_policy, flip_value

from conftest import SMALL_D_S


class TestBoardFlip:
    def test_flip_is_involution(self) -> None:
        """Flipping twice returns the original."""
        board = torch.randn(122, 8, 8)
        assert torch.allclose(flip_board(flip_board(board, 122), 122), board)

    def test_flip_swaps_colors(self) -> None:
        """White pawns (plane 0) become black pawns (plane 6) after flip."""
        encoder = ExtendedBoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board).data
        flipped = flip_board(tensor, 122)
        # White pawns on rank 1 should now be black pawns on rank 6
        assert flipped[6, 6, :].sum() == 8.0  # 8 pawns

    def test_flip_mirrors_ranks(self) -> None:
        """Rank 0 becomes rank 7 after flip."""
        encoder = ExtendedBoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board).data
        flipped = flip_board(tensor, 122)
        # White rook was at (plane=3, rank=0, file=0), after flip
        # it becomes black rook at (plane=9, rank=7, file=0)
        assert flipped[9, 7, 0] == 1.0


class TestPolicyFlip:
    def test_flip_is_involution(self) -> None:
        """Flipping twice returns the original."""
        policy = torch.randn(64, 64)
        assert torch.allclose(flip_policy(flip_policy(policy)), policy)

    def test_flip_mirrors_squares(self) -> None:
        """Move from e2(12) to e4(28) becomes e7(52) to e5(36)."""
        policy = torch.zeros(64, 64)
        policy[12, 28] = 1.0  # e2-e4
        flipped = flip_policy(policy)
        # e2 = rank1*8+file4 = 12, flipped rank = 7-1=6, sq = 6*8+4 = 52
        # e4 = rank3*8+file4 = 28, flipped rank = 7-3=4, sq = 4*8+4 = 36
        assert flipped[52, 36] == 1.0


class TestValueFlip:
    def test_flip_swaps_win_loss(self) -> None:
        """Win and loss swap, draw unchanged."""
        win, draw, loss = 0.6, 0.1, 0.3
        fw, fd, fl = flip_value(win, draw, loss)
        assert fw == loss
        assert fd == draw
        assert fl == win


class TestExtendedBoardFlip:
    """Tests for 122-plane flip_board path (metadata + tactical plane swaps)."""

    def test_122_plane_involution(self) -> None:
        """Flipping twice returns the original for 122-plane tensors."""
        board = torch.randn(122, 8, 8)
        assert torch.allclose(flip_board(flip_board(board, 122), 122), board)

    def test_side_to_move_inverted(self) -> None:
        """Side-to-move plane (96+7=103) should be inverted after flip."""
        encoder = ExtendedBoardEncoder()
        board = chess.Board()  # White to move
        tensor = encoder.encode(board).data
        assert tensor[103].sum() > 0  # side-to-move = 1 for white
        flipped = flip_board(tensor, 122)
        assert flipped[103].sum() == 0.0  # should be 0 for black

    def test_castling_swapped(self) -> None:
        """White castling planes (96,97) should swap with black (98,99)."""
        encoder = ExtendedBoardEncoder()
        board = chess.Board()  # All castling rights present
        tensor = encoder.encode(board).data
        w_king = tensor[96].clone()
        w_queen = tensor[97].clone()
        b_king = tensor[98].clone()
        b_queen = tensor[99].clone()
        flipped = flip_board(tensor, 122)
        assert torch.allclose(flipped[96], b_king)
        assert torch.allclose(flipped[97], b_queen)
        assert torch.allclose(flipped[98], w_king)
        assert torch.allclose(flipped[99], w_queen)

    def test_tactical_planes_swapped(self) -> None:
        """Tactical plane pairs (110-121) should swap white/black on color flip."""
        encoder = ExtendedBoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board).data
        # Store originals for all 6 pairs
        originals = [
            (tensor[110 + i].clone(), tensor[111 + i].clone()) for i in range(0, 12, 2)
        ]
        flipped = flip_board(tensor, 122)
        for pair_idx, (orig_w, orig_b) in enumerate(originals):
            base = 110 + pair_idx * 2
            # After flip, rank-mirrored white plane should be in black slot and vice versa
            assert torch.allclose(flipped[base], orig_b.flip(0))
            assert torch.allclose(flipped[base + 1], orig_w.flip(0))


class TestRoundTrip:
    """Integration test: ExtendedBoardEncoder output through ChessEncoder."""

    def test_extended_encoder_through_chess_encoder(self) -> None:
        """Real ExtendedBoardEncoder output passes through 122-plane ChessEncoder."""
        board_enc = ExtendedBoardEncoder()
        nn_enc = ChessEncoder(num_planes=122, d_s=SMALL_D_S)

        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")

        tensor = board_enc.encode(board).data  # [122, 8, 8]
        batch = tensor.unsqueeze(0)  # [1, 122, 8, 8]

        with torch.no_grad():
            out = nn_enc(batch)
        assert out.shape == (1, 64, SMALL_D_S)
        assert not torch.isnan(out).any()
