"""Tkinter Canvas chess board widget with click-click interaction."""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    from collections.abc import Callable

# Colors
LIGHT_SQUARE = "#F0D9B5"
DARK_SQUARE = "#B58863"
SELECTED_COLOR = "#6495ED"
LEGAL_MOVE_COLOR = "#66BB6A"
LAST_MOVE_COLOR = "#FFFF66"

# Unicode chess pieces
PIECE_SYMBOLS: dict[tuple[int, bool], str] = {
    (chess.KING, chess.WHITE): "\u2654",
    (chess.QUEEN, chess.WHITE): "\u2655",
    (chess.ROOK, chess.WHITE): "\u2656",
    (chess.BISHOP, chess.WHITE): "\u2657",
    (chess.KNIGHT, chess.WHITE): "\u2658",
    (chess.PAWN, chess.WHITE): "\u2659",
    (chess.KING, chess.BLACK): "\u265a",
    (chess.QUEEN, chess.BLACK): "\u265b",
    (chess.ROOK, chess.BLACK): "\u265c",
    (chess.BISHOP, chess.BLACK): "\u265d",
    (chess.KNIGHT, chess.BLACK): "\u265e",
    (chess.PAWN, chess.BLACK): "\u265f",
}


def square_to_file_rank(square: int, flipped: bool) -> tuple[int, int]:
    """Convert chess square (0-63) to (file, rank) in display coordinates.

    file=0 is left column, rank=0 is top row.
    When flipped=False (white at bottom): a1=(0,7), h8=(7,0).
    When flipped=True (black at bottom): a1=(7,0), h8=(0,7).
    """
    f = chess.square_file(square)
    r = chess.square_rank(square)
    if flipped:
        return (7 - f, r)
    return (f, 7 - r)


def file_rank_to_pixel(file: int, rank: int, square_size: int) -> tuple[int, int]:
    """Convert display (file, rank) to pixel center coordinates."""
    x = file * square_size + square_size // 2
    y = rank * square_size + square_size // 2
    return (x, y)


def pixel_to_file_rank(x: int, y: int, square_size: int) -> tuple[int, int] | None:
    """Convert pixel coordinates to display (file, rank). None if outside."""
    f = x // square_size
    r = y // square_size
    if 0 <= f < 8 and 0 <= r < 8:
        return (f, r)
    return None


def _file_rank_to_square(file: int, rank: int, flipped: bool) -> int:
    """Convert display (file, rank) back to chess square index."""
    if flipped:
        return chess.square(7 - file, rank)
    return chess.square(file, 7 - rank)


class BoardWidget(tk.Canvas):
    """Interactive chess board rendered on a Tkinter Canvas."""

    def __init__(self, parent: tk.Widget, square_size: int = 60) -> None:
        self._sq = square_size
        size = square_size * 8
        super().__init__(parent, width=size, height=size, highlightthickness=0)

        self._board = chess.Board()
        self._flipped = False
        self._interactive = False
        self._on_move_cb: Callable[[chess.Move], None] | None = None
        self._selected_square: int | None = None
        self._last_move: chess.Move | None = None

        self.bind("<Button-1>", self._on_click)
        self._draw()

    def set_board(self, board: chess.Board) -> None:
        """Update the displayed position."""
        self._board = board.copy()
        self._selected_square = None
        self._draw()

    def set_interactive(self, enabled: bool) -> None:
        """Enable/disable human move input."""
        self._interactive = enabled
        if not enabled:
            self._selected_square = None
            self._draw()

    def set_on_move(self, callback: Callable[[chess.Move], None]) -> None:
        """Register callback when human makes a move."""
        self._on_move_cb = callback

    def flip(self) -> None:
        """Flip board orientation."""
        self._flipped = not self._flipped
        self._draw()

    def highlight_last_move(self, move: chess.Move) -> None:
        """Highlight the last move played."""
        self._last_move = move
        self._draw()

    def _on_click(self, event: tk.Event) -> None:
        if not self._interactive:
            return

        fr = pixel_to_file_rank(event.x, event.y, self._sq)
        if fr is None:
            return

        clicked_sq = _file_rank_to_square(fr[0], fr[1], self._flipped)

        if self._selected_square is None:
            # Select a piece
            piece = self._board.piece_at(clicked_sq)
            if piece is not None and piece.color == self._board.turn:
                self._selected_square = clicked_sq
                self._draw()
        else:
            # Try to make a move
            move = chess.Move(self._selected_square, clicked_sq)

            # Check for pawn promotion
            piece = self._board.piece_at(self._selected_square)
            if piece is not None and piece.piece_type == chess.PAWN:
                dest_rank = chess.square_rank(clicked_sq)
                if dest_rank in (0, 7):
                    promo = self._ask_promotion()
                    if promo is not None:
                        move = chess.Move(
                            self._selected_square,
                            clicked_sq,
                            promotion=promo,
                        )

            if move in self._board.legal_moves:
                self._selected_square = None
                self._last_move = move
                if self._on_move_cb is not None:
                    self._on_move_cb(move)
            else:
                # Deselect or reselect
                new_piece = self._board.piece_at(clicked_sq)
                if new_piece is not None and new_piece.color == self._board.turn:
                    self._selected_square = clicked_sq
                else:
                    self._selected_square = None
            self._draw()

    def _ask_promotion(self) -> int | None:
        """Show a promotion dialog. Returns piece type or None."""
        dialog = tk.Toplevel(self)
        dialog.title("Promote to")
        dialog.resizable(False, False)
        dialog.grab_set()

        result: list[int | None] = [None]

        for piece_type, label in [
            (chess.QUEEN, "Q"),
            (chess.ROOK, "R"),
            (chess.BISHOP, "B"),
            (chess.KNIGHT, "N"),
        ]:
            pt = piece_type

            def choose(p: int = pt) -> None:
                result[0] = p
                dialog.destroy()

            tk.Button(dialog, text=label, width=4, command=choose).pack(
                side=tk.LEFT, padx=2, pady=4
            )

        dialog.wait_window()
        return result[0]

    def _draw(self) -> None:
        """Redraw the entire board."""
        self.delete("all")
        sq = self._sq

        # Draw squares
        for f in range(8):
            for r in range(8):
                x0 = f * sq
                y0 = r * sq
                is_light = (f + r) % 2 == 0
                color = LIGHT_SQUARE if is_light else DARK_SQUARE

                # Last move highlight
                if self._last_move is not None:
                    sq_here = _file_rank_to_square(f, r, self._flipped)
                    if sq_here in (
                        self._last_move.from_square,
                        self._last_move.to_square,
                    ):
                        color = LAST_MOVE_COLOR

                self.create_rectangle(x0, y0, x0 + sq, y0 + sq, fill=color, outline="")

        # Selected square highlight
        if self._selected_square is not None:
            sf, sr = square_to_file_rank(self._selected_square, self._flipped)
            self.create_rectangle(
                sf * sq,
                sr * sq,
                sf * sq + sq,
                sr * sq + sq,
                outline=SELECTED_COLOR,
                width=3,
            )

            # Legal move indicators
            for move in self._board.legal_moves:
                if move.from_square == self._selected_square:
                    tf, tr = square_to_file_rank(move.to_square, self._flipped)
                    cx, cy = file_rank_to_pixel(tf, tr, sq)
                    radius = sq // 6
                    self.create_oval(
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                        fill=LEGAL_MOVE_COLOR,
                        outline="",
                        stipple="gray50",
                    )

        # Draw pieces
        for square in chess.SQUARES:
            piece = self._board.piece_at(square)
            if piece is None:
                continue
            symbol = PIECE_SYMBOLS.get((piece.piece_type, piece.color))
            if symbol is None:
                continue
            df, dr = square_to_file_rank(square, self._flipped)
            cx, cy = file_rank_to_pixel(df, dr, sq)
            self.create_text(
                cx,
                cy,
                text=symbol,
                font=("Arial", sq // 2),
                anchor="center",
            )

        # File/rank labels
        label_font = ("Arial", sq // 6)
        for i in range(8):
            # File labels (a-h) at bottom
            file_idx = (7 - i) if self._flipped else i
            self.create_text(
                i * sq + sq - 4,
                8 * sq - 4,
                text=chr(ord("a") + file_idx),
                font=label_font,
                anchor="se",
                fill="#666666",
            )
            # Rank labels (1-8) on left
            rank_idx = (i + 1) if self._flipped else (8 - i)
            self.create_text(
                4,
                i * sq + 4,
                text=str(rank_idx),
                font=label_font,
                anchor="nw",
                fill="#666666",
            )
