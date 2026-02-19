import random
from collections import deque

from denoisr.types import GameRecord


class SimpleReplayBuffer:
    """Uniform-sampling replay buffer with fixed capacity (Phase 1).

    When capacity is exceeded, the oldest entries are evicted (FIFO).
    Sampling is uniform with replacement.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[GameRecord] = deque(maxlen=capacity)

    def add(self, record: GameRecord, priority: float = 1.0) -> None:
        self._buffer.append(record)

    def sample(self, batch_size: int) -> list[GameRecord]:
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")
        return random.choices(list(self._buffer), k=batch_size)

    def update_priorities(
        self, indices: list[int], priorities: list[float]
    ) -> None:
        pass  # no-op for uniform buffer

    def __len__(self) -> int:
        return len(self._buffer)


class PriorityReplayBuffer:
    """Priority-based replay buffer (EfficientZero V2 style, Phase 3).

    Samples proportionally to priority^alpha. Higher priority items
    (larger TD error / loss) are sampled more frequently.
    Supports priority updates after training on sampled batches.
    """

    def __init__(
        self, capacity: int, alpha: float = 0.6
    ) -> None:
        self._capacity = capacity
        self._alpha = alpha
        self._records: list[GameRecord] = []
        self._priorities: list[float] = []

    def add(self, record: GameRecord, priority: float = 1.0) -> None:
        if len(self._records) >= self._capacity:
            min_idx = min(
                range(len(self._priorities)),
                key=lambda i: self._priorities[i],
            )
            self._records.pop(min_idx)
            self._priorities.pop(min_idx)
        self._records.append(record)
        self._priorities.append(priority)

    def sample(self, batch_size: int) -> list[GameRecord]:
        if not self._records:
            raise ValueError("Cannot sample from empty buffer")
        weights = [p**self._alpha for p in self._priorities]
        return random.choices(self._records, weights=weights, k=batch_size)

    def update_priorities(
        self, indices: list[int], priorities: list[float]
    ) -> None:
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = prio

    def __len__(self) -> int:
        return len(self._records)
