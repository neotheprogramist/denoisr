# tests/test_gui/test_board_widget.py
import chess

from denoisr.gui.board_widget import (
    PIECE_SYMBOLS,
    file_rank_to_pixel,
    pixel_to_file_rank,
    resolve_click_move,
    SOLID_PIECE_SYMBOLS,
    square_to_file_rank,
)


class TestCoordinateMath:
    def test_a1_white_orientation(self) -> None:
        """a1 is bottom-left when white is at bottom (flipped=False)."""
        fr = square_to_file_rank(chess.A1, flipped=False)
        assert fr == (0, 7)  # file=0 (left), rank=7 (bottom row)

    def test_h8_white_orientation(self) -> None:
        fr = square_to_file_rank(chess.H8, flipped=False)
        assert fr == (7, 0)  # file=7 (right), rank=0 (top row)

    def test_a1_black_orientation(self) -> None:
        """a1 is top-right when black is at bottom (flipped=True)."""
        fr = square_to_file_rank(chess.A1, flipped=True)
        assert fr == (7, 0)

    def test_file_rank_to_pixel_center(self) -> None:
        sq_size = 60
        x, y = file_rank_to_pixel(0, 0, sq_size)
        assert x == 30  # center of first column
        assert y == 30  # center of first row

    def test_pixel_to_file_rank_roundtrip(self) -> None:
        sq_size = 60
        for f in range(8):
            for r in range(8):
                cx, cy = file_rank_to_pixel(f, r, sq_size)
                rf, rr = pixel_to_file_rank(cx, cy, sq_size)
                assert (rf, rr) == (f, r)

    def test_pixel_outside_board_returns_none(self) -> None:
        result = pixel_to_file_rank(-5, 10, 60)
        assert result is None
        result = pixel_to_file_rank(500, 10, 60)
        assert result is None


class TestPieceSymbols:
    def test_white_king(self) -> None:
        assert PIECE_SYMBOLS[(chess.KING, chess.WHITE)] == "\u2654"

    def test_black_pawn(self) -> None:
        assert PIECE_SYMBOLS[(chess.PAWN, chess.BLACK)] == "\u265f"

    def test_all_pieces_present(self) -> None:
        for color in (chess.WHITE, chess.BLACK):
            for piece_type in range(1, 7):
                assert (piece_type, color) in PIECE_SYMBOLS

    def test_solid_symbols_cover_all_piece_types(self) -> None:
        for piece_type in range(1, 7):
            assert piece_type in SOLID_PIECE_SYMBOLS


class TestResolveClickMove:
    def test_non_promotion_move_resolves_directly(self) -> None:
        board = chess.Board()
        move = resolve_click_move(board, chess.G1, chess.F3)
        assert move == chess.Move.from_uci("g1f3")

    def test_illegal_target_does_not_request_promotion(self) -> None:
        board = chess.Board("1N6/P7/8/8/8/8/8/k6K w - - 0 1")
        asked = {"value": False}

        def chooser() -> int | None:
            asked["value"] = True
            return chess.QUEEN

        move = resolve_click_move(board, chess.A7, chess.B8, chooser)
        assert move is None
        assert asked["value"] is False

    def test_promotion_move_uses_dialog_choice(self) -> None:
        board = chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1")
        move = resolve_click_move(
            board,
            chess.A7,
            chess.A8,
            lambda: chess.ROOK,
        )
        assert move == chess.Move.from_uci("a7a8r")
