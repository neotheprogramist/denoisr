import chess
import pytest
import torch
from hypothesis import given, settings

from denoisr.data.board_encoder import SimpleBoardEncoder

from conftest import random_boards


class TestSimpleBoardEncoder:
    @pytest.fixture
    def encoder(self) -> SimpleBoardEncoder:
        return SimpleBoardEncoder()

    def test_num_planes(self, encoder: SimpleBoardEncoder) -> None:
        assert encoder.num_planes == 12

    def test_starting_position_shape(
        self, encoder: SimpleBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        assert bt.data.shape == (12, 8, 8)

    def test_starting_position_white_pawns(
        self, encoder: SimpleBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        # Plane 0 = white pawns, rank 1 (index 1) = all pawns
        assert bt.data[0, 1, :].sum().item() == 8
        assert bt.data[0, 1, :].all()

    def test_starting_position_white_king(
        self, encoder: SimpleBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        # Plane 5 = white king, e1 = rank 0, file 4
        assert bt.data[5, 0, 4].item() == 1.0
        assert bt.data[5].sum().item() == 1.0

    def test_starting_position_total_pieces(
        self, encoder: SimpleBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        assert bt.data.sum().item() == 32  # 16 white + 16 black pieces

    def test_empty_board(self, encoder: SimpleBoardEncoder) -> None:
        board = chess.Board.empty()
        bt = encoder.encode(board)
        assert bt.data.sum().item() == 0.0

    def test_deterministic(self, encoder: SimpleBoardEncoder) -> None:
        board = chess.Board()
        bt1 = encoder.encode(board)
        bt2 = encoder.encode(board)
        assert torch.equal(bt1.data, bt2.data)

    @given(board=random_boards())
    @settings(max_examples=50)
    def test_piece_count_matches(
        self, board: chess.Board
    ) -> None:
        encoder = SimpleBoardEncoder()
        bt = encoder.encode(board)
        total_tensor_pieces = bt.data.sum().item()
        total_board_pieces = len(board.piece_map())
        assert total_tensor_pieces == total_board_pieces

    @given(board=random_boards())
    @settings(max_examples=50)
    def test_no_overlapping_pieces(
        self, board: chess.Board
    ) -> None:
        encoder = SimpleBoardEncoder()
        bt = encoder.encode(board)
        # Sum across all 12 planes — no square should have more than 1
        per_square = bt.data.sum(dim=0)
        assert per_square.max().item() <= 1.0
