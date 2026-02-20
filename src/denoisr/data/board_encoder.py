import chess
import torch

from denoisr.types import BoardTensor

_PLANE_INDEX = {
    (pt, color): (pt - 1) + (0 if color == chess.WHITE else 6)
    for pt in chess.PIECE_TYPES
    for color in chess.COLORS
}


class SimpleBoardEncoder:
    @property
    def num_planes(self) -> int:
        return 12

    def encode(self, board: chess.Board) -> BoardTensor:
        data = torch.zeros(12, 8, 8, dtype=torch.float32)
        for sq, piece in board.piece_map().items():
            plane = _PLANE_INDEX[(piece.piece_type, piece.color)]
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            data[plane, rank, file] = 1.0
        return BoardTensor(data)
