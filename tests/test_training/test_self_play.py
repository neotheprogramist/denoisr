import pytest
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.game.chess_game import ChessGame
from denoisr.training.self_play import (
    SelfPlayActor,
    SelfPlayConfig,
    TemperatureSchedule,
)

from conftest import SMALL_D_S


class _DummyModel:
    def __init__(self, d_s: int) -> None:
        self.d_s = d_s

    def predict(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.rand(64, 64), torch.tensor([0.33, 0.34, 0.33])

    def predict_next(
        self, state: torch.Tensor, f: int, t: int
    ) -> tuple[torch.Tensor, float]:
        return torch.randn(64, self.d_s), 0.0

    def encode(self, board_tensor: torch.Tensor) -> torch.Tensor:
        return torch.randn(64, self.d_s)


class TestSelfPlayActor:
    @pytest.fixture
    def actor(self) -> SelfPlayActor:
        model = _DummyModel(SMALL_D_S)
        return SelfPlayActor(
            policy_value_fn=model.predict,
            world_model_fn=model.predict_next,
            encode_fn=model.encode,
            game=ChessGame(),
            board_encoder=SimpleBoardEncoder(),
            config=SelfPlayConfig(
                num_simulations=10, max_moves=50, temperature=1.0
            ),
        )

    def test_play_game_returns_record(self, actor: SelfPlayActor) -> None:
        record = actor.play_game()
        assert len(record.actions) > 0
        assert record.result in (1.0, 0.0, -1.0)

    def test_game_terminates(self, actor: SelfPlayActor) -> None:
        record = actor.play_game()
        assert len(record.actions) <= 50

    def test_all_actions_valid(self, actor: SelfPlayActor) -> None:
        record = actor.play_game()
        for action in record.actions:
            assert 0 <= action.from_square < 64
            assert 0 <= action.to_square < 64


class TestTemperatureSchedule:
    def test_explore_phase_returns_base(self) -> None:
        ts = TemperatureSchedule(base=1.0, explore_moves=30)
        assert ts.get_temperature(0) == 1.0
        assert ts.get_temperature(29) == 1.0

    def test_exploit_phase_returns_zero(self) -> None:
        ts = TemperatureSchedule(base=1.0, explore_moves=30)
        assert ts.get_temperature(30) == 0.0
        assert ts.get_temperature(100) == 0.0

    def test_generation_decay(self) -> None:
        ts = TemperatureSchedule(
            base=1.0, explore_moves=30, generation_decay=0.5
        )
        assert ts.get_temperature(0, generation=0) == 1.0
        assert ts.get_temperature(0, generation=1) == pytest.approx(0.5)
        assert ts.get_temperature(0, generation=2) == pytest.approx(0.25)
