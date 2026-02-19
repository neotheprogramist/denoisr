import chess

from denoisr.types import Action


class SimpleActionEncoder:
    def encode_move(self, move: chess.Move) -> Action:
        return Action(move.from_square, move.to_square, move.promotion)

    def decode_action(self, action: Action, board: chess.Board) -> chess.Move:
        return chess.Move(
            action.from_square, action.to_square, action.promotion
        )

    def action_to_index(self, action: Action) -> int:
        return action.from_square * 64 + action.to_square

    def index_to_action(self, index: int, board: chess.Board) -> Action:
        from_sq = index // 64
        to_sq = index % 64
        promotion = self._infer_promotion(from_sq, to_sq, board)
        return Action(from_sq, to_sq, promotion)

    def _infer_promotion(
        self, from_sq: int, to_sq: int, board: chess.Board
    ) -> int | None:
        piece = board.piece_at(from_sq)
        if piece is None or piece.piece_type != chess.PAWN:
            return None
        to_rank = chess.square_rank(to_sq)
        if piece.color == chess.WHITE and to_rank == 7:
            return chess.QUEEN
        if piece.color == chess.BLACK and to_rank == 0:
            return chess.QUEEN
        return None
