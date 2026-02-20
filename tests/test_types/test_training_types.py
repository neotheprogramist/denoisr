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


class TestTrainingExampleMetadata:
    def test_training_example_with_metadata(self) -> None:
        board = BoardTensor(torch.randn(12, 8, 8))
        policy = PolicyTarget(torch.zeros(64, 64))
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        ex = TrainingExample(
            board=board, policy=policy, value=value,
            game_id=42, eco_code="B90", piece_count=24,
        )
        assert ex.game_id == 42
        assert ex.eco_code == "B90"
        assert ex.piece_count == 24

    def test_training_example_metadata_defaults_to_none(self) -> None:
        board = BoardTensor(torch.randn(12, 8, 8))
        policy = PolicyTarget(torch.zeros(64, 64))
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        ex = TrainingExample(board=board, policy=policy, value=value)
        assert ex.game_id is None
        assert ex.eco_code is None
        assert ex.piece_count is None


class TestStackUnstackMetadata:
    def test_stack_includes_metadata_tensors(self) -> None:
        from denoisr.scripts.generate_data import stack_examples

        ex = TrainingExample(
            board=BoardTensor(torch.randn(12, 8, 8)),
            policy=PolicyTarget(torch.zeros(64, 64)),
            value=ValueTarget(win=1.0, draw=0.0, loss=0.0),
            game_id=7, eco_code="C50", piece_count=30,
        )
        stacked = stack_examples([ex])
        assert "game_ids" in stacked
        assert "piece_counts" in stacked
        assert stacked["game_ids"][0].item() == 7
        assert stacked["piece_counts"][0].item() == 30

    def test_unstack_recovers_metadata(self) -> None:
        from denoisr.scripts.generate_data import stack_examples, unstack_examples

        ex = TrainingExample(
            board=BoardTensor(torch.randn(12, 8, 8)),
            policy=PolicyTarget(torch.zeros(64, 64)),
            value=ValueTarget(win=1.0, draw=0.0, loss=0.0),
            game_id=7, eco_code="C50", piece_count=30,
        )
        stacked = stack_examples([ex])
        recovered = unstack_examples(stacked)
        assert recovered[0].game_id == 7
        assert recovered[0].eco_code == "C50"
        assert recovered[0].piece_count == 30

    def test_unstack_backward_compat_no_metadata(self) -> None:
        from denoisr.scripts.generate_data import unstack_examples

        data = {
            "boards": torch.randn(2, 12, 8, 8),
            "policies": torch.zeros(2, 64, 64),
            "values": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
        }
        examples = unstack_examples(data)
        assert len(examples) == 2
        assert examples[0].game_id is None
        assert examples[0].eco_code is None
        assert examples[0].piece_count is None


class TestGameRecord:
    def test_valid(self) -> None:
        actions = (Action(12, 28), Action(52, 36))
        gr = GameRecord(actions=actions, result=1.0)
        assert len(gr.actions) == 2

    def test_result_values(self) -> None:
        for r in (1.0, 0.0, -1.0):
            GameRecord(actions=(), result=r)

    def test_eco_code(self) -> None:
        gr = GameRecord(actions=(), result=1.0, eco_code="B90")
        assert gr.eco_code == "B90"

    def test_eco_code_defaults_to_none(self) -> None:
        gr = GameRecord(actions=(), result=1.0)
        assert gr.eco_code is None
