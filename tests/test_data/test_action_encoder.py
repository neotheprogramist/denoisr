import chess
import pytest
from hypothesis import given, settings

from denoisr.data.action_encoder import SimpleActionEncoder
from denoisr.types import Action

from conftest import random_boards


class TestSimpleActionEncoder:
    @pytest.fixture
    def encoder(self) -> SimpleActionEncoder:
        return SimpleActionEncoder()

    def test_encode_e2e4(self, encoder: SimpleActionEncoder) -> None:
        move = chess.Move.from_uci("e2e4")
        action = encoder.encode_move(move)
        assert action.from_square == chess.E2
        assert action.to_square == chess.E4
        assert action.promotion is None

    def test_encode_promotion(self, encoder: SimpleActionEncoder) -> None:
        move = chess.Move.from_uci("a7a8q")
        action = encoder.encode_move(move)
        assert action.promotion == chess.QUEEN

    def test_decode_e2e4(self, encoder: SimpleActionEncoder) -> None:
        board = chess.Board()
        action = Action(chess.E2, chess.E4)
        move = encoder.decode_action(action, board)
        assert move == chess.Move.from_uci("e2e4")

    def test_round_trip_starting_position(self, encoder: SimpleActionEncoder) -> None:
        board = chess.Board()
        for move in board.legal_moves:
            action = encoder.encode_move(move)
            decoded = encoder.decode_action(action, board)
            assert decoded == move

    def test_action_to_index_range(self, encoder: SimpleActionEncoder) -> None:
        action = Action(0, 63)
        idx = encoder.action_to_index(action)
        assert 0 <= idx < 64 * 64

    def test_index_round_trip(self, encoder: SimpleActionEncoder) -> None:
        board = chess.Board()
        for move in board.legal_moves:
            action = encoder.encode_move(move)
            idx = encoder.action_to_index(action)
            recovered = encoder.index_to_action(idx, board)
            decoded = encoder.decode_action(recovered, board)
            assert decoded == move

    @given(board=random_boards())
    @settings(max_examples=30)
    def test_all_legal_moves_round_trip(self, board: chess.Board) -> None:
        encoder = SimpleActionEncoder()
        for move in board.legal_moves:
            action = encoder.encode_move(move)
            idx = encoder.action_to_index(action)
            assert 0 <= idx < 64 * 64
            recovered = encoder.index_to_action(idx, board)
            decoded = encoder.decode_action(recovered, board)
            # For underpromotions, we default to queen — skip exact match
            if move.promotion and move.promotion != chess.QUEEN:
                continue
            assert decoded == move
