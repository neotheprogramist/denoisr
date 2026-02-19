from typing import Callable

import chess
from torch import Tensor

from denoisr.data.board_encoder import SimpleBoardEncoder
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
        game: ChessGame,
        board_encoder: SimpleBoardEncoder,
        num_simulations: int = 100,
    ) -> None:
        self._game = game
        self._board_encoder = board_encoder
        self._encode = encode_fn
        self._mcts = MCTS(
            policy_value_fn=policy_value_fn,
            world_model_fn=world_model_fn,
            config=MCTSConfig(num_simulations=num_simulations),
        )

    def reanalyse(self, record: GameRecord) -> list[TrainingExample]:
        board = chess.Board()
        examples: list[TrainingExample] = []

        for action in record.actions:
            board_tensor = self._board_encoder.encode(board)
            latent = self._encode(board_tensor.data.unsqueeze(0)).squeeze(0)
            legal_mask = self._game.get_valid_moves(board).data

            visit_dist = self._mcts.search(latent, legal_mask)
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
            move = chess.Move(
                action.from_square, action.to_square, action.promotion
            )
            board.push(move)

        return examples
