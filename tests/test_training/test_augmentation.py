import chess
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.training.augmentation import flip_board, flip_policy, flip_value


class TestBoardFlip:
    def test_flip_is_involution(self) -> None:
        """Flipping twice returns the original."""
        board = torch.randn(12, 8, 8)
        assert torch.allclose(flip_board(flip_board(board, 12), 12), board)

    def test_flip_swaps_colors(self) -> None:
        """White pawns (plane 0) become black pawns (plane 6) after flip."""
        encoder = SimpleBoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board).data
        flipped = flip_board(tensor, 12)
        # White pawns on rank 1 should now be black pawns on rank 6
        assert flipped[6, 6, :].sum() == 8.0  # 8 pawns

    def test_flip_mirrors_ranks(self) -> None:
        """Rank 0 becomes rank 7 after flip."""
        encoder = SimpleBoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board).data
        flipped = flip_board(tensor, 12)
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
