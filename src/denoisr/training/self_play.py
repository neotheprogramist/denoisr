from dataclasses import dataclass
from typing import Callable

import chess
import torch
from torch import Tensor

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
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

    def get_temperature(self, move_number: int, generation: int = 0) -> float:
        base = self.base * (self.generation_decay**generation)
        return base if move_number < self.explore_moves else 0.0


@dataclass(frozen=True)
class SelfPlayConfig:
    num_simulations: int = 100
    max_moves: int = 300
    temperature: float = 1.0
    c_puct: float = 1.4
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temp_schedule: TemperatureSchedule | None = None


class SelfPlayActor:
    """Runs self-play games using MCTS in latent space."""

    def __init__(
        self,
        policy_value_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
        world_model_fn: Callable[[Tensor, int, int], tuple[Tensor, float]],
        encode_fn: Callable[[Tensor], Tensor],
        diffusion_policy_fn: Callable[[Tensor, Tensor], Tensor] | None,
        game: ChessGame,
        board_encoder: ExtendedBoardEncoder,
        config: SelfPlayConfig,
    ) -> None:
        self._game = game
        self._board_encoder = board_encoder
        self._encode = encode_fn
        self._diffusion_policy_fn = diffusion_policy_fn
        self._config = config
        self._mcts = MCTS(
            policy_value_fn=policy_value_fn,
            world_model_fn=world_model_fn,
            config=MCTSConfig(
                num_simulations=config.num_simulations,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon,
                temperature=config.temperature,
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
        return self._game.get_next_state(board, Action(from_sq, to_sq, promotion))

    def play_game(self, generation: int = 0, alpha: float = 0.0) -> GameRecord:
        board = self._game.get_init_board()
        actions: list[Action] = []
        alpha = max(0.0, min(1.0, alpha))

        for move_num in range(self._config.max_moves):
            result = self._game.get_game_ended(board)
            if result is not None:
                return GameRecord(actions=tuple(actions), result=result)

            if self._config.temp_schedule is not None:
                temp = self._config.temp_schedule.get_temperature(move_num, generation)
                self._mcts._config = MCTSConfig(
                    num_simulations=self._config.num_simulations,
                    c_puct=self._config.c_puct,
                    dirichlet_alpha=self._config.dirichlet_alpha,
                    dirichlet_epsilon=self._config.dirichlet_epsilon,
                    temperature=temp,
                )

            board_tensor = self._board_encoder.encode(board).data
            latent = self._encode(board_tensor.unsqueeze(0)).squeeze(0)
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

            flat_dist = visit_dist.reshape(-1)
            if flat_dist.sum() == 0:
                raise RuntimeError(
                    "MCTS produced zero visit distribution. "
                    "This indicates a bug in MCTS search or legal mask generation. "
                    f"Board FEN: {board.fen()}, move_num: {move_num}"
                )

            idx = int(torch.multinomial(flat_dist, 1).item())
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
