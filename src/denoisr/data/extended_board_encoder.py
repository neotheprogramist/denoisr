import chess
import torch

from denoisr.types import BoardTensor

_PLANE_INDEX = {
    (pt, color): (pt - 1) + (0 if color == chess.WHITE else 6)
    for pt in chess.PIECE_TYPES
    for color in chess.COLORS
}

_HISTORY_DEPTH = 7
_PIECE_PLANES = 12
_HISTORY_PLANES = _PIECE_PLANES * _HISTORY_DEPTH  # 84
_META_PLANES = 14  # castling(4) + ep(1) + rule50(1) + repetition(1) + stm(1) + material(2) + check(2) + opp_bishops(2)
_TOTAL_PLANES = _PIECE_PLANES + _HISTORY_PLANES + _META_PLANES  # 110


def _encode_pieces(board: chess.Board, planes: torch.Tensor, offset: int) -> None:
    for sq, piece in board.piece_map().items():
        plane = offset + _PLANE_INDEX[(piece.piece_type, piece.color)]
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        planes[plane, rank, file] = 1.0


class ExtendedBoardEncoder:
    @property
    def num_planes(self) -> int:
        return _TOTAL_PLANES

    def encode(self, board: chess.Board) -> BoardTensor:
        data = torch.zeros(_TOTAL_PLANES, 8, 8, dtype=torch.float32)

        # Current position piece planes (0..11)
        _encode_pieces(board, data, 0)

        # History planes (12..95): up to 7 past positions
        history_board = board.copy()
        for h in range(_HISTORY_DEPTH):
            if not history_board.move_stack:
                break
            history_board.pop()
            offset = _PIECE_PLANES + h * _PIECE_PLANES
            _encode_pieces(history_board, data, offset)

        meta_start = _PIECE_PLANES + _HISTORY_PLANES

        # Castling rights (4 planes, broadcast)
        for i, right in enumerate([
            chess.BB_H1,  # white kingside
            chess.BB_A1,  # white queenside
            chess.BB_H8,  # black kingside
            chess.BB_A8,  # black queenside
        ]):
            has_right = bool(board.castling_rights & right)
            if has_right:
                data[meta_start + i] = 1.0

        # En passant (1 plane)
        if board.ep_square is not None:
            rank = chess.square_rank(board.ep_square)
            file = chess.square_file(board.ep_square)
            data[meta_start + 4, rank, file] = 1.0

        # Rule-50 counter (1 plane, normalized)
        data[meta_start + 5] = min(board.halfmove_clock / 100.0, 1.0)

        # Repetition count (1 plane, normalized: 0, 0.5, 1.0)
        if board.is_repetition(3):
            data[meta_start + 6] = 1.0
        elif board.is_repetition(2):
            data[meta_start + 6] = 0.5

        # Side to move (1 plane)
        if board.turn == chess.WHITE:
            data[meta_start + 7] = 1.0

        # Material counts (2 planes, normalized by max possible)
        for ci, color in enumerate(chess.COLORS):
            material = sum(
                len(board.pieces(pt, color)) * v
                for pt, v in zip(chess.PIECE_TYPES, [1, 3, 3, 5, 9, 0])
            )
            data[meta_start + 8 + ci] = min(material / 39.0, 1.0)

        # Pieces giving check (2 planes)
        for ci, color in enumerate(chess.COLORS):
            king_sq = board.king(not color)
            if king_sq is not None:
                attackers = board.attackers(color, king_sq)
                for sq in attackers:
                    rank = chess.square_rank(sq)
                    file = chess.square_file(sq)
                    data[meta_start + 10 + ci, rank, file] = 1.0

        # Opposite-colored bishops (2 planes, broadcast)
        white_bishops = board.pieces(chess.BISHOP, chess.WHITE)
        black_bishops = board.pieces(chess.BISHOP, chess.BLACK)
        if white_bishops and black_bishops:
            w_light = any(
                (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 0
                for sq in white_bishops
            )
            w_dark = any(
                (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 1
                for sq in white_bishops
            )
            b_light = any(
                (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 0
                for sq in black_bishops
            )
            b_dark = any(
                (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 1
                for sq in black_bishops
            )
            if (w_light and b_dark and not w_dark and not b_light) or \
               (w_dark and b_light and not w_light and not b_dark):
                data[meta_start + 12] = 1.0
                data[meta_start + 13] = 1.0

        return BoardTensor(data)
