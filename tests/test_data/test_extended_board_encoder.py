import chess
import pytest
import torch
from hypothesis import given, settings

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.types import BoardTensor

from conftest import random_boards


class TestExtendedBoardEncoder:
    @pytest.fixture
    def encoder(self) -> ExtendedBoardEncoder:
        return ExtendedBoardEncoder()

    def test_num_planes_exceeds_simple(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        assert encoder.num_planes > 12

    def test_starting_position_shape(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        assert bt.data.shape == (encoder.num_planes, 8, 8)

    def test_piece_planes_match_simple_encoder(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board = chess.Board()
        bt = encoder.encode(board)
        # First 12 planes should be piece placement
        assert bt.data[:12].sum().item() == 32

    def test_castling_rights_change(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board_with = chess.Board()
        board_without = chess.Board()
        board_without.set_castling_fen("-")
        bt_with = encoder.encode(board_with)
        bt_without = encoder.encode(board_without)
        assert not torch.equal(bt_with.data, bt_without.data)

    def test_en_passant_encoded(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board = chess.Board()
        board.push_uci("e2e4")  # creates en passant target
        bt = encoder.encode(board)
        # Should differ from starting position encoding
        bt_start = encoder.encode(chess.Board())
        assert not torch.equal(bt.data, bt_start.data)

    def test_side_to_move_differs(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board_white = chess.Board()
        board_black = chess.Board()
        board_black.push_uci("e2e4")
        bt_w = encoder.encode(board_white)
        bt_b = encoder.encode(board_black)
        assert not torch.equal(bt_w.data, bt_b.data)

    def test_deterministic(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board = chess.Board()
        bt1 = encoder.encode(board)
        bt2 = encoder.encode(board)
        assert torch.equal(bt1.data, bt2.data)

    @given(board=random_boards())
    @settings(max_examples=30)
    def test_no_nan_or_inf(
        self, board: chess.Board
    ) -> None:
        encoder = ExtendedBoardEncoder()
        bt = encoder.encode(board)
        assert not torch.isnan(bt.data).any()
        assert not torch.isinf(bt.data).any()

    @given(board=random_boards())
    @settings(max_examples=30)
    def test_values_bounded(
        self, board: chess.Board
    ) -> None:
        encoder = ExtendedBoardEncoder()
        bt = encoder.encode(board)
        assert bt.data.min() >= 0.0
        assert bt.data.max() <= 1.0
