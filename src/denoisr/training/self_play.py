from dataclasses import dataclass
from typing import Callable

import chess
import torch
from torch import Tensor

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.game.chess_game import ChessGame
from denoisr.training.mcts import MCTS, MCTSConfig
from denoisr.types import Action, GameRecord


@dataclass(frozen=True)
class TemperatureSchedule:
    """Temperature scheduling for self-play exploration/exploitation.

    Within a game: temperature = base for first `explore_moves` moves, then 0.
    Across generations: base *= generation_decay per generation.
    """

    base: float = 1.0
    explore_moves: int = 30
    generation_decay: float = 0.97

    def get_temperature(
        self, move_number: int, generation: int = 0
    ) -> float:
        base = self.base * (self.generation_decay**generation)
        return base if move_number < self.explore_moves else 0.0


@dataclass(frozen=True)
class SelfPlayConfig:
    num_simulations: int = 100
    max_moves: int = 300
    temperature: float = 1.0
    c_puct: float = 1.4
    temp_schedule: TemperatureSchedule | None = None


class SelfPlayActor:
    """Runs self-play games using MCTS in latent space."""

    def __init__(
        self,
        policy_value_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
        world_model_fn: Callable[[Tensor, int, int], tuple[Tensor, float]],
        encode_fn: Callable[[Tensor], Tensor],
        game: ChessGame,
        board_encoder: SimpleBoardEncoder,
        config: SelfPlayConfig,
    ) -> None:
        self._game = game
        self._board_encoder = board_encoder
        self._encode = encode_fn
        self._config = config
        self._mcts = MCTS(
            policy_value_fn=policy_value_fn,
            world_model_fn=world_model_fn,
            config=MCTSConfig(
                num_simulations=config.num_simulations,
                c_puct=config.c_puct,
                temperature=config.temperature,
            ),
        )

    def play_game(self, generation: int = 0) -> GameRecord:
        board = self._game.get_init_board()
        actions: list[Action] = []

        for move_num in range(self._config.max_moves):
            result = self._game.get_game_ended(board)
            if result is not None:
                return GameRecord(actions=tuple(actions), result=result)

            if self._config.temp_schedule is not None:
                temp = self._config.temp_schedule.get_temperature(
                    move_num, generation
                )
                self._mcts._config = MCTSConfig(
                    num_simulations=self._config.num_simulations,
                    c_puct=self._config.c_puct,
                    temperature=temp,
                )

            board_tensor = self._board_encoder.encode(board).data
            latent = self._encode(board_tensor.unsqueeze(0)).squeeze(0)
            legal_mask = self._game.get_valid_moves(board).data

            visit_dist = self._mcts.search(latent, legal_mask)

            flat_dist = visit_dist.reshape(-1)
            if flat_dist.sum() == 0:
                flat_dist = legal_mask.float().reshape(-1)
                flat_dist = flat_dist / flat_dist.sum()

            idx = torch.multinomial(flat_dist, 1).item()
            from_sq = idx // 64
            to_sq = idx % 64

            promotion = None
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(to_sq)
                if (piece.color == chess.WHITE and to_rank == 7) or (
                    piece.color == chess.BLACK and to_rank == 0
                ):
                    promotion = chess.QUEEN

            action = Action(from_sq, to_sq, promotion)
            actions.append(action)
            board = self._game.get_next_state(board, action)

        return GameRecord(actions=tuple(actions), result=0.0)
