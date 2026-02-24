from typing import Callable

import chess
import torch
from torch import Tensor

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.game.chess_game import ChessGame
from denoisr.training.mcts import MCTS, MCTSConfig
from denoisr.types import (
    GameRecord,
    PolicyTarget,
    TrainingExample,
    ValueTarget,
)


class ReanalyseActor:
    """MuZero Reanalyse: re-run MCTS on old trajectories with the current network.

    Given a GameRecord from the replay buffer, replays each position
    through MCTS with the latest model weights to generate improved
    policy targets. Value targets use the original game result.
    """

    def __init__(
        self,
        policy_value_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
        world_model_fn: Callable[[Tensor, int, int], tuple[Tensor, float]],
        encode_fn: Callable[[Tensor], Tensor],
        diffusion_policy_fn: Callable[[Tensor, Tensor], Tensor] | None,
        game: ChessGame,
        board_encoder: ExtendedBoardEncoder,
        num_simulations: int = 100,
    ) -> None:
        self._game = game
        self._board_encoder = board_encoder
        self._encode = encode_fn
        self._diffusion_policy_fn = diffusion_policy_fn
        self._mcts = MCTS(
            policy_value_fn=policy_value_fn,
            world_model_fn=world_model_fn,
            # Reanalyse should be deterministic/improvement-focused:
            # disable root Dirichlet noise.
            config=MCTSConfig(
                num_simulations=num_simulations,
                dirichlet_epsilon=0.0,
            ),
            legal_mask_fn=lambda b: self._game.get_valid_moves(b).data,
            transition_fn=self._transition_board,
        )

    def _transition_board(
        self, board: chess.Board, from_sq: int, to_sq: int
    ) -> chess.Board:
        promotion = None
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0
            ):
                promotion = chess.QUEEN
        move = chess.Move(from_sq, to_sq, promotion)
        next_board = board.copy()
        next_board.push(move)
        return next_board

    def reanalyse(
        self, record: GameRecord, alpha: float = 0.0
    ) -> list[TrainingExample]:
        board = chess.Board()
        examples: list[TrainingExample] = []
        alpha = max(0.0, min(1.0, alpha))

        for action in record.actions:
            board_tensor = self._board_encoder.encode(board)
            latent = self._encode(board_tensor.data.unsqueeze(0)).squeeze(0)
            legal_mask = self._game.get_valid_moves(board).data.to(
                device=latent.device, dtype=torch.bool
            )

            to_play = 1 if board.turn == chess.WHITE else -1
            visit_dist = self._mcts.search(
                latent,
                legal_mask,
                root_to_play=to_play,
                root_board=board,
            )
            if self._diffusion_policy_fn is not None and alpha > 0.0:
                diffusion_policy = self._diffusion_policy_fn(latent, legal_mask)
                visit_dist = (1.0 - alpha) * visit_dist + alpha * diffusion_policy
                visit_dist = visit_dist * legal_mask.float()
                total = visit_dist.sum()
                if total > 0:
                    visit_dist = visit_dist / total
            policy = PolicyTarget(visit_dist)

            if record.result == 1.0:
                value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
            elif record.result == -1.0:
                value = ValueTarget(win=0.0, draw=0.0, loss=1.0)
            else:
                value = ValueTarget(win=0.0, draw=1.0, loss=0.0)

            examples.append(
                TrainingExample(board=board_tensor, policy=policy, value=value)
            )
            move = chess.Move(action.from_square, action.to_square, action.promotion)
            board.push(move)

        return examples
