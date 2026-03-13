import chess
import pytest
import torch

from denoisr_chess.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr_chess.game.chess_game import ChessGame
from denoisr_chess.training.reanalyse import ReanalyseActor
from denoisr_chess.types import Action, GameRecord, TrainingExample

from conftest import SMALL_D_S


class _DummyModel:
    def __init__(self, d_s: int) -> None:
        self.d_s = d_s

    def predict(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.rand(64, 64), torch.tensor([0.33, 0.34, 0.33])

    def predict_next(
        self, state: torch.Tensor, f: int, t: int
    ) -> tuple[torch.Tensor, float]:
        return torch.randn(64, self.d_s), 0.0

    def encode(self, board_tensor: torch.Tensor) -> torch.Tensor:
        return torch.randn(64, self.d_s)


class TestReanalyseActor:
    @pytest.fixture
    def actor(self) -> ReanalyseActor:
        model = _DummyModel(SMALL_D_S)
        return ReanalyseActor(
            policy_value_fn=model.predict,
            world_model_fn=model.predict_next,
            encode_fn=model.encode,
            diffusion_policy_fn=None,
            game=ChessGame(),
            board_encoder=ExtendedBoardEncoder(),
            num_simulations=10,
        )

    def test_reanalyse_produces_examples(self, actor: ReanalyseActor) -> None:
        move = chess.Move.from_uci("e2e4")
        record = GameRecord(
            actions=(Action(move.from_square, move.to_square),),
            result=1.0,
        )
        examples = actor.reanalyse(record)
        assert len(examples) == 1
        assert isinstance(examples[0], TrainingExample)

    def test_policy_targets_are_distributions(self, actor: ReanalyseActor) -> None:
        moves = ["e2e4", "e7e5", "g1f3"]
        record = GameRecord(
            actions=tuple(
                Action(
                    chess.Move.from_uci(m).from_square,
                    chess.Move.from_uci(m).to_square,
                )
                for m in moves
            ),
            result=0.0,
        )
        examples = actor.reanalyse(record)
        for ex in examples:
            total = ex.policy.data.sum().item()
            assert abs(total - 1.0) < 0.01
