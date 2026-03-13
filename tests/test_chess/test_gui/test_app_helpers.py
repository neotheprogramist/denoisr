"""Pure helper tests for GUI application logic."""

from __future__ import annotations

import chess

from denoisr_chess.gui.app import summarize_captures


def test_summarize_captures_initial_position() -> None:
    board = chess.Board()
    captured_by_white, captured_by_black = summarize_captures(board)
    assert captured_by_white == "-"
    assert captured_by_black == "-"


def test_summarize_captures_after_simple_trade() -> None:
    board = chess.Board()
    for uci in ("e2e4", "d7d5", "e4d5", "d8d5", "b1c3"):
        board.push_uci(uci)

    captured_by_white, captured_by_black = summarize_captures(board)
    assert captured_by_white == "♟x1"
    assert captured_by_black == "♟x1"


def test_summarize_captures_multiple_piece_types() -> None:
    board = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")
    board.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.B2, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.C2, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.H7, chess.Piece(chess.PAWN, chess.BLACK))
    board.set_piece_at(chess.G7, chess.Piece(chess.ROOK, chess.BLACK))
    captured_by_white, captured_by_black = summarize_captures(board)
    assert captured_by_white == "♛x1 ♜x1 ♝x2 ♞x2 ♟x7"
    assert captured_by_black == "♛x1 ♜x2 ♝x1 ♞x1 ♟x7"
