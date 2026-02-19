import pytest
import torch

from denoisr.types.action import Action
from denoisr.types.board import BoardTensor
from denoisr.types.training import (
    GameRecord,
    PolicyTarget,
    TrainingExample,
    ValueTarget,
)


class TestPolicyTarget:
    def test_valid_shape(self) -> None:
        pt = PolicyTarget(torch.zeros(64, 64))
        assert pt.data.shape == (64, 64)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            PolicyTarget(torch.zeros(10, 10))


class TestValueTarget:
    def test_valid_wdl(self) -> None:
        vt = ValueTarget(win=0.7, draw=0.2, loss=0.1)
        assert abs(vt.win + vt.draw + vt.loss - 1.0) < 1e-5

    def test_rejects_bad_sum(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            ValueTarget(win=0.5, draw=0.5, loss=0.5)


class TestTrainingExample:
    def test_holds_components(self) -> None:
        board = BoardTensor(torch.zeros(12, 8, 8))
        policy = PolicyTarget(torch.zeros(64, 64))
        value = ValueTarget(1.0, 0.0, 0.0)
        ex = TrainingExample(board=board, policy=policy, value=value)
        assert ex.board is board


class TestGameRecord:
    def test_valid(self) -> None:
        actions = (Action(12, 28), Action(52, 36))
        gr = GameRecord(actions=actions, result=1.0)
        assert len(gr.actions) == 2

    def test_result_values(self) -> None:
        for r in (1.0, 0.0, -1.0):
            GameRecord(actions=(), result=r)
