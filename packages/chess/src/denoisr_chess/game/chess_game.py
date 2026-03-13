import chess
import torch

from denoisr_chess.types import Action, LegalMask


class ChessGame:
    def get_init_board(self) -> chess.Board:
        return chess.Board()

    def get_board_size(self) -> tuple[int, int]:
        return (8, 8)

    def get_action_size(self) -> int:
        return 64 * 64

    def get_next_state(self, board: chess.Board, action: Action) -> chess.Board:
        new_board = board.copy()
        move = chess.Move(action.from_square, action.to_square, action.promotion)
        new_board.push(move)
        return new_board

    def get_valid_moves(self, board: chess.Board) -> LegalMask:
        mask = torch.zeros(64, 64, dtype=torch.bool)
        for move in board.legal_moves:
            mask[move.from_square, move.to_square] = True
        return LegalMask(mask)

    def get_game_ended(self, board: chess.Board) -> float | None:
        if not board.is_game_over():
            return None
        result = board.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0

    def get_canonical_form(self, board: chess.Board) -> chess.Board:
        if board.turn == chess.BLACK:
            return board.mirror()
        return board.copy()

    def get_symmetries(
        self, board: chess.Board, policy: torch.Tensor
    ) -> list[tuple[chess.Board, torch.Tensor]]:
        return [(board.copy(), policy.clone())]
