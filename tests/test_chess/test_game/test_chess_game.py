import chess
import pytest
import torch
from hypothesis import given, settings

from denoisr_chess.game.chess_game import ChessGame
from denoisr_chess.types import Action

from conftest import random_boards


class TestChessGame:
    @pytest.fixture
    def game(self) -> ChessGame:
        return ChessGame()

    def test_init_board_is_starting_position(self, game: ChessGame) -> None:
        board = game.get_init_board()
        assert board.fen() == chess.STARTING_FEN

    def test_board_size(self, game: ChessGame) -> None:
        assert game.get_board_size() == (8, 8)

    def test_action_size(self, game: ChessGame) -> None:
        assert game.get_action_size() == 64 * 64

    def test_next_state_applies_move(self, game: ChessGame) -> None:
        board = game.get_init_board()
        action = Action(from_square=12, to_square=28)  # e2e4
        new_board = game.get_next_state(board, action)
        assert new_board.piece_at(28) == chess.Piece(chess.PAWN, chess.WHITE)
        assert new_board.piece_at(12) is None

    def test_next_state_does_not_mutate_original(self, game: ChessGame) -> None:
        board = game.get_init_board()
        original_fen = board.fen()
        game.get_next_state(board, Action(12, 28))
        assert board.fen() == original_fen

    def test_valid_moves_starting_position(self, game: ChessGame) -> None:
        board = game.get_init_board()
        mask = game.get_valid_moves(board)
        assert mask.data.shape == (64, 64)
        assert mask.data.dtype == torch.bool
        # No promotions possible from starting position, so counts match
        num_legal = len(list(board.legal_moves))
        assert mask.data.sum().item() == num_legal

    def test_game_ended_none_at_start(self, game: ChessGame) -> None:
        assert game.get_game_ended(game.get_init_board()) is None

    def test_game_ended_scholars_mate(self, game: ChessGame) -> None:
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"):
            board.push_uci(uci)
        assert game.get_game_ended(board) == 1.0

    def test_game_ended_fools_mate(self, game: ChessGame) -> None:
        board = chess.Board()
        for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
            board.push_uci(uci)
        assert game.get_game_ended(board) == -1.0

    def test_canonical_form_white_unchanged(self, game: ChessGame) -> None:
        board = game.get_init_board()
        canonical = game.get_canonical_form(board)
        assert canonical.fen() == board.fen()

    def test_canonical_form_black_mirrored(self, game: ChessGame) -> None:
        board = chess.Board()
        board.push_uci("e2e4")  # now black to move
        canonical = game.get_canonical_form(board)
        assert canonical.turn == chess.WHITE

    def test_symmetries_returns_identity(self, game: ChessGame) -> None:
        board = game.get_init_board()
        policy = torch.randn(64, 64)
        syms = game.get_symmetries(board, policy)
        assert len(syms) == 1
        assert syms[0][0].fen() == board.fen()
        assert torch.equal(syms[0][1], policy)

    @given(board=random_boards())
    @settings(max_examples=50)
    def test_valid_moves_count_matches_python_chess(self, board: chess.Board) -> None:
        game = ChessGame()
        if game.get_game_ended(board) is not None:
            return
        mask = game.get_valid_moves(board)
        # Count unique (from, to) pairs — promotions share the same cell
        expected = len({(m.from_square, m.to_square) for m in board.legal_moves})
        assert mask.data.sum().item() == expected
