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

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        pass  # no-op for uniform buffer

    def __len__(self) -> int:
        return len(self._buffer)


class PriorityReplayBuffer:
    """Priority-based replay buffer (EfficientZero V2 style, Phase 3).

    Samples proportionally to priority^alpha. Higher priority items
    (larger TD error / loss) are sampled more frequently.
    Supports priority updates after training on sampled batches.
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self._capacity = capacity
        self._alpha = alpha
        self._records: list[GameRecord] = []
        self._priorities: list[float] = []
        self._next_insert = 0
        self._weights_cache: list[float] | None = None

    def _invalidate_cache(self) -> None:
        self._weights_cache = None

    def add(self, record: GameRecord, priority: float = 1.0) -> None:
        safe_priority = max(float(priority), 1e-8)
        if len(self._records) < self._capacity:
            self._records.append(record)
            self._priorities.append(safe_priority)
            if len(self._records) == self._capacity:
                self._next_insert = 0
        else:
            self._records[self._next_insert] = record
            self._priorities[self._next_insert] = safe_priority
            self._next_insert = (self._next_insert + 1) % self._capacity
        self._invalidate_cache()

    def sample(self, batch_size: int) -> list[GameRecord]:
        if not self._records:
            raise ValueError("Cannot sample from empty buffer")
        if self._weights_cache is None:
            self._weights_cache = [max(p, 1e-8) ** self._alpha for p in self._priorities]
        return random.choices(self._records, weights=self._weights_cache, k=batch_size)

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        changed = False
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = max(float(prio), 1e-8)
                changed = True
        if changed:
            self._invalidate_cache()

    def __len__(self) -> int:
        return len(self._records)
