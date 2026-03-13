import pytest

from denoisr_chess.training.replay_buffer import PriorityReplayBuffer
from denoisr_chess.types import Action, GameRecord


def _make_record(n_moves: int = 3, result: float = 1.0) -> GameRecord:
    actions = tuple(Action(i, i + 1) for i in range(n_moves))
    return GameRecord(actions=actions, result=result)


class TestPriorityReplayBuffer:
    def test_empty_length(self) -> None:
        buf = PriorityReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_add_with_priority(self) -> None:
        buf = PriorityReplayBuffer(capacity=100)
        buf.add(_make_record(), priority=5.0)
        assert len(buf) == 1

    def test_high_priority_sampled_more(self) -> None:
        buf = PriorityReplayBuffer(capacity=100)
        low = _make_record(result=-1.0)
        high = _make_record(result=1.0)
        buf.add(low, priority=0.01)
        buf.add(high, priority=100.0)
        results = [buf.sample(1)[0].result for _ in range(100)]
        high_count = sum(1 for r in results if r == 1.0)
        assert high_count > 80

    def test_update_priorities(self) -> None:
        buf = PriorityReplayBuffer(capacity=100)
        buf.add(_make_record(result=0.0), priority=1.0)
        buf.add(_make_record(result=1.0), priority=1.0)
        buf.update_priorities([0], [100.0])
        results = [buf.sample(1)[0].result for _ in range(100)]
        first_count = sum(1 for r in results if r == 0.0)
        assert first_count > 80

    def test_capacity_evicts(self) -> None:
        buf = PriorityReplayBuffer(capacity=3)
        for i in range(5):
            buf.add(_make_record(result=float(i)), priority=1.0)
        assert len(buf) == 3

    def test_sample_from_empty_raises(self) -> None:
        buf = PriorityReplayBuffer(capacity=100)
        with pytest.raises(ValueError, match="empty"):
            buf.sample(batch_size=1)
