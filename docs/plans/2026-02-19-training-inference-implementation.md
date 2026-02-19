# Training & Inference Implementation Plan (Tiers 6–7)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the training infrastructure (loss computation, replay buffer, MCTS, supervised trainer, diffusion trainer, self-play) and inference layer (chess engine, UCI protocol).

**Architecture:** Three-phase training pipeline. Phase 1: supervised from Lichess+Stockfish. Phase 2: world model + diffusion bootstrapping. Phase 3: RL self-play with MCTS to diffusion transition. UCI wrapper for GUI integration.

**Tech Stack:** PyTorch, python-chess, all `denoisr.nn` and `denoisr.data` modules

**Prerequisite:** Tiers 1–5 must have 100% passing tests.

---

### Task 0: Training Protocols

**Files:**

- Create: `src/denoisr/training/protocols.py`

**Step 1: Write protocols**

`src/denoisr/training/protocols.py`:

```python
from collections.abc import Iterator
from typing import Protocol

import torch

from denoisr.types import GameRecord, TrainingExample


class LossComputer(Protocol):
    def compute(
        self,
        pred_policy: torch.Tensor,
        pred_value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        **auxiliary_losses: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (total_loss, {loss_name: scalar_value}).

        auxiliary_losses may include: consistency_loss, diffusion_loss,
        reward_loss, ply_loss (added in Phase 2/3).
        """
        ...


class ReplayBuffer(Protocol):
    def add(self, record: GameRecord, priority: float = 1.0) -> None: ...
    def sample(self, batch_size: int) -> list[GameRecord]: ...
    def update_priorities(self, indices: list[int], priorities: list[float]) -> None: ...
    def __len__(self) -> int: ...


class MCTSPolicy(Protocol):
    def search(
        self,
        root_state: torch.Tensor,
        num_simulations: int,
    ) -> torch.Tensor:
        """Returns visit count distribution [64, 64]."""
        ...
```

**Step 2: Commit**

```bash
git add src/denoisr/training/protocols.py
git commit -m "feat: add training protocols (T6 setup)"
```

---

### Task 1: Loss Computer (6-Term + HarmonyDream)

**Spec reference:** "6-term combined loss: policy, value, consistency, diffusion, reward, regularization" and "HarmonyDream dynamic loss coefficient adjustment."

**Regularization note:** The spec lists λ_reg ||θ||^2 as the 6th loss term. Our implementation handles this via `weight_decay` in AdamW (which is equivalent to L2 regularization). This frees the 6th explicit loss slot for **ply prediction** (game-length auxiliary signal from the WDLP value head), which provides stronger gradient signal than a redundant regularization term.

**Files:**

- Create: `src/denoisr/training/loss.py`
- Test: `tests/test_training/test_loss.py`

**Step 1: Write failing tests**

`tests/test_training/test_loss.py`:

```python
import pytest
import torch

from denoisr.training.loss import ChessLossComputer


class TestChessLossComputer:
    @pytest.fixture
    def loss_fn(self) -> ChessLossComputer:
        return ChessLossComputer()

    def test_total_loss_is_scalar(self, loss_fn: ChessLossComputer) -> None:
        pred_policy = torch.randn(4, 64, 64)
        pred_value = torch.softmax(torch.randn(4, 3), dim=-1)
        target_policy = torch.zeros(4, 64, 64)
        target_policy[:, 12, 28] = 1.0
        target_value = torch.tensor(
            [[1.0, 0.0, 0.0]] * 4, dtype=torch.float32
        )
        total, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        assert total.ndim == 0
        assert total.item() >= 0

    def test_breakdown_has_policy_and_value(
        self, loss_fn: ChessLossComputer
    ) -> None:
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 1] = 1.0
        target_value = torch.tensor([[0.0, 1.0, 0.0]] * 2)
        _, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        assert "policy" in breakdown
        assert "value" in breakdown
        assert all(v >= 0 for v in breakdown.values())

    def test_auxiliary_losses_included_in_total(
        self, loss_fn: ChessLossComputer
    ) -> None:
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

        # Without auxiliary
        total_base, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        # With auxiliary losses (consistency, diffusion, reward, ply)
        total_aux, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value,
            consistency_loss=torch.tensor(0.5),
            diffusion_loss=torch.tensor(0.3),
            reward_loss=torch.tensor(0.1),
            ply_loss=torch.tensor(0.2),
        )
        assert total_aux.item() > total_base.item()
        assert "consistency" in breakdown
        assert "diffusion" in breakdown
        assert "reward" in breakdown
        assert "ply" in breakdown

    def test_all_6_terms_in_full_breakdown(self) -> None:
        loss_fn = ChessLossComputer()
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)
        _, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value,
            consistency_loss=torch.tensor(0.5),
            diffusion_loss=torch.tensor(0.3),
            reward_loss=torch.tensor(0.1),
            ply_loss=torch.tensor(0.2),
        )
        expected_keys = {"policy", "value", "consistency", "diffusion", "reward", "ply", "total"}
        assert expected_keys == set(breakdown.keys())

    def test_perfect_prediction_low_loss(
        self, loss_fn: ChessLossComputer
    ) -> None:
        target_policy = torch.zeros(1, 64, 64)
        target_policy[0, 12, 28] = 1.0
        pred_policy = torch.full((1, 64, 64), -10.0)
        pred_policy[0, 12, 28] = 10.0

        target_value = torch.tensor([[1.0, 0.0, 0.0]])
        pred_value = torch.tensor([[0.95, 0.04, 0.01]])

        total, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        assert total.item() < 1.0

    def test_loss_is_finite(self, loss_fn: ChessLossComputer) -> None:
        pred_policy = torch.randn(4, 64, 64)
        pred_value = torch.softmax(torch.randn(4, 3), dim=-1)
        target_policy = torch.zeros(4, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[0.5, 0.3, 0.2]] * 4)
        total, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        assert not torch.isnan(total)
        assert not torch.isinf(total)

    def test_gradient_flows(self, loss_fn: ChessLossComputer) -> None:
        pred_policy = torch.randn(2, 64, 64, requires_grad=True)
        pred_value = torch.softmax(
            torch.randn(2, 3, requires_grad=True), dim=-1
        )
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)
        total, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        total.backward()
        assert pred_policy.grad is not None

    def test_harmony_dream_adjusts_coefficients(self) -> None:
        loss_fn = ChessLossComputer(use_harmony_dream=True)
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

        # Run several steps so HarmonyDream adapts
        for _ in range(5):
            loss_fn.compute(
                pred_policy, pred_value, target_policy, target_value,
                consistency_loss=torch.tensor(10.0),  # much larger
                diffusion_loss=torch.tensor(0.01),     # much smaller
            )
        # After adaptation, coefficients should have shifted
        coeffs = loss_fn.get_coefficients()
        assert "consistency" in coeffs
        assert "diffusion" in coeffs
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_training/test_loss.py -v`

**Step 3: Implement**

`src/denoisr/training/loss.py`:

```python
import torch
from torch import Tensor
from torch.nn import functional as F


class ChessLossComputer:
    """6-term loss computer with optional HarmonyDream balancing.

    Core losses (always active):
    1. Policy: cross-entropy between predicted logits and target distribution
    2. Value: cross-entropy between predicted WDL and target WDL

    Auxiliary losses (Phase 2/3, passed via kwargs):
    3. Consistency: SimSiam negative cosine similarity
    4. Diffusion: MSE between predicted and actual noise
    5. Reward: MSE between predicted and actual reward
    6. Ply: Huber loss on game-length prediction

    HarmonyDream (optional): tracks EMA of per-loss gradient norms and
    adjusts coefficients inversely proportional to balance contributions.
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        consistency_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        reward_weight: float = 1.0,
        ply_weight: float = 0.1,
        use_harmony_dream: bool = False,
        harmony_ema_decay: float = 0.99,
    ) -> None:
        self._base_weights = {
            "policy": policy_weight,
            "value": value_weight,
            "consistency": consistency_weight,
            "diffusion": diffusion_weight,
            "reward": reward_weight,
            "ply": ply_weight,
        }
        self._use_harmony = use_harmony_dream
        self._ema_decay = harmony_ema_decay
        self._ema_norms: dict[str, float] = {}
        self._coefficients: dict[str, float] = dict(self._base_weights)

    def compute(
        self,
        pred_policy: Tensor,
        pred_value: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
        **auxiliary_losses: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        B = pred_policy.shape[0]
        pred_flat = pred_policy.reshape(B, -1)
        target_flat = target_policy.reshape(B, -1)
        log_probs = F.log_softmax(pred_flat, dim=-1)
        policy_loss = -(target_flat * log_probs).sum(dim=-1).mean()

        pred_log = torch.log(pred_value.clamp(min=1e-8))
        value_loss = -(target_value * pred_log).sum(dim=-1).mean()

        losses = {"policy": policy_loss, "value": value_loss}

        # Add auxiliary losses
        for name in ("consistency", "diffusion", "reward", "ply"):
            key = f"{name}_loss"
            if key in auxiliary_losses:
                losses[name] = auxiliary_losses[key]

        # HarmonyDream coefficient update
        if self._use_harmony:
            self._update_harmony(losses)

        # Weighted sum
        total = sum(
            self._coefficients.get(name, self._base_weights.get(name, 1.0)) * loss
            for name, loss in losses.items()
        )

        breakdown = {name: loss.item() for name, loss in losses.items()}
        breakdown["total"] = total.item()
        return total, breakdown

    def _update_harmony(self, losses: dict[str, Tensor]) -> None:
        for name, loss in losses.items():
            norm = loss.detach().abs().item()
            if name not in self._ema_norms:
                self._ema_norms[name] = norm
            else:
                self._ema_norms[name] = (
                    self._ema_decay * self._ema_norms[name]
                    + (1 - self._ema_decay) * norm
                )

        if self._ema_norms:
            mean_norm = sum(self._ema_norms.values()) / len(self._ema_norms)
            if mean_norm > 0:
                for name in self._ema_norms:
                    ratio = mean_norm / max(self._ema_norms[name], 1e-8)
                    self._coefficients[name] = (
                        self._base_weights.get(name, 1.0) * ratio
                    )

    def get_coefficients(self) -> dict[str, float]:
        return dict(self._coefficients)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_loss.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/training/loss.py tests/test_training/test_loss.py
git commit -m "feat: add policy+value loss computer (T6)"
```

---

### Task 2: Replay Buffer (Simple + Priority)

**Spec reference:** Simple FIFO buffer for Phase 1, EfficientZero V2 priority-based buffer for Phase 3.

**Files:**

- Create: `src/denoisr/training/replay_buffer.py`
- Test: `tests/test_training/test_replay_buffer.py`

**Step 1: Write failing tests**

`tests/test_training/test_replay_buffer.py`:

```python
import pytest

from denoisr.training.replay_buffer import PriorityReplayBuffer, SimpleReplayBuffer
from denoisr.types import Action, GameRecord


def _make_record(n_moves: int = 3, result: float = 1.0) -> GameRecord:
    actions = tuple(Action(i, i + 1) for i in range(n_moves))
    return GameRecord(actions=actions, result=result)


class TestSimpleReplayBuffer:
    def test_empty_length(self) -> None:
        buf = SimpleReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_add_and_length(self) -> None:
        buf = SimpleReplayBuffer(capacity=100)
        buf.add(_make_record())
        assert len(buf) == 1

    def test_sample_returns_correct_count(self) -> None:
        buf = SimpleReplayBuffer(capacity=100)
        for _ in range(10):
            buf.add(_make_record())
        samples = buf.sample(batch_size=5)
        assert len(samples) == 5

    def test_sample_returns_game_records(self) -> None:
        buf = SimpleReplayBuffer(capacity=100)
        buf.add(_make_record())
        samples = buf.sample(batch_size=1)
        assert isinstance(samples[0], GameRecord)

    def test_capacity_evicts_oldest(self) -> None:
        buf = SimpleReplayBuffer(capacity=3)
        for i in range(5):
            buf.add(_make_record(result=float(i)))
        assert len(buf) == 3

    def test_sample_from_empty_raises(self) -> None:
        buf = SimpleReplayBuffer(capacity=100)
        with pytest.raises(ValueError, match="empty"):
            buf.sample(batch_size=1)

    def test_sample_more_than_available(self) -> None:
        buf = SimpleReplayBuffer(capacity=100)
        buf.add(_make_record())
        buf.add(_make_record())
        samples = buf.sample(batch_size=5)
        assert len(samples) == 5


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
        # Sample many times, high-priority should dominate
        results = [buf.sample(1)[0].result for _ in range(100)]
        high_count = sum(1 for r in results if r == 1.0)
        assert high_count > 80  # should be ~99%

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
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_training/test_replay_buffer.py -v`

**Step 3: Implement**

`src/denoisr/training/replay_buffer.py`:

```python
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
            # Evict lowest priority
            min_idx = min(range(len(self._priorities)), key=lambda i: self._priorities[i])
            self._records.pop(min_idx)
            self._priorities.pop(min_idx)
        self._records.append(record)
        self._priorities.append(priority)

    def sample(self, batch_size: int) -> list[GameRecord]:
        if not self._records:
            raise ValueError("Cannot sample from empty buffer")
        weights = [p ** self._alpha for p in self._priorities]
        return random.choices(self._records, weights=weights, k=batch_size)

    def update_priorities(
        self, indices: list[int], priorities: list[float]
    ) -> None:
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = prio

    def __len__(self) -> int:
        return len(self._records)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_replay_buffer.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/training/replay_buffer.py tests/test_training/test_replay_buffer.py
git commit -m "feat: add replay buffer with FIFO eviction (T6)"
```

---

### Task 3: MCTS

**Files:**

- Create: `src/denoisr/training/mcts.py`
- Test: `tests/test_training/test_mcts.py`

**Step 1: Write failing tests**

`tests/test_training/test_mcts.py`:

```python
import math

import pytest
import torch

from denoisr.training.mcts import MCTS, MCTSConfig

from tests.conftest import SMALL_D_S


class _MockPolicyValue:
    def __init__(self, d_s: int) -> None:
        self.d_s = d_s

    def predict(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.ones(64, 64) / (64 * 64)
        value = torch.tensor([0.33, 0.34, 0.33])
        return policy, value


class _MockWorldModel:
    def __init__(self, d_s: int) -> None:
        self.d_s = d_s

    def predict_next(
        self, state: torch.Tensor, f: int, t: int
    ) -> tuple[torch.Tensor, float]:
        return torch.randn(64, self.d_s), 0.0


class TestMCTS:
    @pytest.fixture
    def config(self) -> MCTSConfig:
        return MCTSConfig(
            num_simulations=50,
            c_puct=1.4,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )

    @pytest.fixture
    def mcts(self, config: MCTSConfig) -> MCTS:
        pv = _MockPolicyValue(SMALL_D_S)
        wm = _MockWorldModel(SMALL_D_S)
        return MCTS(
            policy_value_fn=pv.predict,
            world_model_fn=wm.predict_next,
            config=config,
        )

    def test_search_returns_valid_distribution(
        self, mcts: MCTS
    ) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True
        legal_mask[1, 18] = True

        visit_dist = mcts.search(state, legal_mask)
        assert visit_dist.shape == (64, 64)
        assert visit_dist.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_search_only_legal_moves(self, mcts: MCTS) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True

        visit_dist = mcts.search(state, legal_mask)
        illegal_mass = visit_dist[~legal_mask].sum().item()
        assert illegal_mass == pytest.approx(0.0, abs=1e-7)

    def test_more_sims_concentrates_distribution(self) -> None:
        pv = _MockPolicyValue(SMALL_D_S)
        wm = _MockWorldModel(SMALL_D_S)
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True

        config_few = MCTSConfig(num_simulations=10, c_puct=1.4)
        config_many = MCTSConfig(num_simulations=200, c_puct=1.4)
        mcts_few = MCTS(pv.predict, wm.predict_next, config_few)
        mcts_many = MCTS(pv.predict, wm.predict_next, config_many)

        dist_few = mcts_few.search(state, legal_mask)
        dist_many = mcts_many.search(state, legal_mask)

        entropy_few = -(
            dist_few[dist_few > 0] * dist_few[dist_few > 0].log()
        ).sum()
        entropy_many = -(
            dist_many[dist_many > 0] * dist_many[dist_many > 0].log()
        ).sum()
        assert entropy_many <= entropy_few + 0.5

    def test_visit_counts_nonnegative(self, mcts: MCTS) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        visit_dist = mcts.search(state, legal_mask)
        assert (visit_dist >= 0).all()
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_training/test_mcts.py -v`

**Step 3: Implement**

`src/denoisr/training/mcts.py`:

```python
import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.4
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0


class _Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "state")

    def __init__(self, prior: float, state: Tensor | None = None) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[tuple[int, int], "_Node"] = {}
        self.state = state

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(parent_visits)
            / (1 + self.visit_count)
        )
        return self.q_value + exploration


PolicyValueFn = Callable[[Tensor], tuple[Tensor, Tensor]]
WorldModelFn = Callable[[Tensor, int, int], tuple[Tensor, float]]


class MCTS:
    """Monte Carlo Tree Search in latent space.

    Uses a policy-value function for leaf evaluation and a world model
    for state transitions. Returns a visit-count distribution over
    legal moves after num_simulations rollouts.
    """

    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        world_model_fn: WorldModelFn,
        config: MCTSConfig,
    ) -> None:
        self._pv = policy_value_fn
        self._wm = world_model_fn
        self._config = config

    def search(self, root_state: Tensor, legal_mask: Tensor) -> Tensor:
        root = _Node(prior=0.0, state=root_state)

        policy, _ = self._pv(root_state)
        policy = policy * legal_mask.float()
        total = policy.sum()
        if total > 0:
            policy = policy / total

        legal_indices = legal_mask.nonzero(as_tuple=False)
        if len(legal_indices) > 0 and self._config.dirichlet_epsilon > 0:
            noise = torch.distributions.Dirichlet(
                torch.full(
                    (len(legal_indices),), self._config.dirichlet_alpha
                )
            ).sample()
            eps = self._config.dirichlet_epsilon
            for idx, (f, t) in enumerate(legal_indices.tolist()):
                orig = policy[f, t].item()
                policy[f, t] = (1 - eps) * orig + eps * noise[idx].item()

        for f, t in legal_indices.tolist():
            root.children[(f, t)] = _Node(prior=policy[f, t].item())

        for _ in range(self._config.num_simulations):
            self._simulate(root)

        visit_dist = torch.zeros(64, 64)
        for (f, t), child in root.children.items():
            visit_dist[f, t] = child.visit_count

        total_visits = visit_dist.sum()
        if total_visits > 0:
            if self._config.temperature == 0:
                best = visit_dist.argmax()
                visit_dist = torch.zeros(64, 64)
                visit_dist.view(-1)[best] = 1.0
            else:
                visit_dist = visit_dist / total_visits

        return visit_dist

    def _simulate(self, root: _Node) -> float:
        node = root
        path: list[_Node] = [node]
        action_taken: tuple[int, int] | None = None

        while node.children and node.visit_count > 0:
            best_action = max(
                node.children.keys(),
                key=lambda a: node.children[a].ucb_score(
                    node.visit_count, self._config.c_puct
                ),
            )
            action_taken = best_action
            node = node.children[best_action]
            path.append(node)

        if node.state is None and action_taken is not None:
            parent = path[-2]
            f, t = action_taken
            state, reward = self._wm(parent.state, f, t)
            node.state = state
        else:
            reward = 0.0

        if node.state is not None:
            policy, value = self._pv(node.state)
            leaf_value = (value[0] - value[2]).item() + reward

            if not node.children:
                for fi in range(64):
                    for ti in range(64):
                        p = policy[fi, ti].item()
                        if p > 1e-8:
                            node.children[(fi, ti)] = _Node(prior=p)
        else:
            leaf_value = 0.0

        for n in path:
            n.visit_count += 1
            n.value_sum += leaf_value

        return leaf_value
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_mcts.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/training/mcts.py tests/test_training/test_mcts.py
git commit -m "feat: add latent-space MCTS with UCB and Dirichlet noise (T6)"
```

---

### Task 4: Supervised Trainer

**Files:**

- Create: `src/denoisr/training/supervised_trainer.py`
- Test: `tests/test_training/test_supervised_trainer.py`

**Step 1: Write failing tests**

`tests/test_training/test_supervised_trainer.py`:

```python
import pathlib

import pytest
import torch

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

from tests.conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


def _make_batch(n: int = 8) -> list[TrainingExample]:
    examples = []
    for _ in range(n):
        board = BoardTensor(torch.randn(12, 8, 8))
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0
        policy = PolicyTarget(policy_data)
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        examples.append(
            TrainingExample(board=board, policy=policy, value=value)
        )
    return examples


class TestSupervisedTrainer:
    @pytest.fixture
    def trainer(self, device: torch.device) -> SupervisedTrainer:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        loss_fn = ChessLossComputer()
        return SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
        )

    def test_train_step_returns_loss(
        self, trainer: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        assert "policy" in breakdown

    def test_loss_decreases_over_steps(
        self, trainer: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        losses = []
        for _ in range(20):
            loss, _ = trainer.train_step(batch)
            losses.append(loss)
        assert losses[-1] < losses[0]

    def test_save_and_load_checkpoint(
        self, trainer: SupervisedTrainer, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(path)
        assert path.exists()
        trainer.load_checkpoint(path)
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v`

**Step 3: Implement**

`src/denoisr/training/supervised_trainer.py`:

```python
from pathlib import Path

import torch
from torch import nn

from denoisr.training.loss import ChessLossComputer
from denoisr.types import TrainingExample


class SupervisedTrainer:
    """Supervised training loop for Phase 1.

    Takes batches of TrainingExamples (board tensor + policy/value targets)
    and updates the encoder, backbone, policy head, and value head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        loss_fn: ChessLossComputer,
        lr: float = 1e-4,
        device: torch.device | None = None,
    ) -> None:
        self.encoder = encoder
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head
        self.loss_fn = loss_fn
        self.device = device or torch.device("cpu")

        params = (
            list(encoder.parameters())
            + list(backbone.parameters())
            + list(policy_head.parameters())
            + list(value_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def train_step(
        self, batch: list[TrainingExample]
    ) -> tuple[float, dict[str, float]]:
        boards = torch.stack([ex.board.data for ex in batch]).to(self.device)
        target_policies = torch.stack([ex.policy.data for ex in batch]).to(
            self.device
        )
        target_values = torch.tensor(
            [[ex.value.win, ex.value.draw, ex.value.loss] for ex in batch],
            dtype=torch.float32,
            device=self.device,
        )

        self.encoder.train()
        self.backbone.train()
        self.policy_head.train()
        self.value_head.train()

        latent = self.encoder(boards)
        features = self.backbone(latent)
        pred_policy = self.policy_head(features)
        pred_value, pred_ply = self.value_head(features)

        total_loss, breakdown = self.loss_fn.compute(
            pred_policy, pred_value, target_policies, target_values
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), breakdown

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "backbone": self.backbone.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.backbone.load_state_dict(checkpoint["backbone"])
        self.policy_head.load_state_dict(checkpoint["policy_head"])
        self.value_head.load_state_dict(checkpoint["value_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/training/supervised_trainer.py tests/test_training/test_supervised_trainer.py
git commit -m "feat: add supervised trainer with checkpoint support (T6)"
```

---

### Task 5: Diffusion Trainer

**Intuition — learning what real games look like:** The diffusion trainer teaches the model by repeatedly corrupting real trajectory snippets from self-play with noise and asking it to reconstruct the originals. Concretely: take a trajectory `[pos₁] → move₁ → [pos₂] → move₂ → ...`, add Gaussian noise at a random timestep, and train the model to predict the clean trajectory. The curriculum progresses from few denoising steps (coarse strategic patterns) to many steps (fine tactical details). Over thousands of training games, the model internalizes what "plausible winning futures" look like — games where forks were played, where pieces were developed, where checkmates were delivered. See the component design doc's "How the Diffusion Model Learns to Score Moves Through Self-Play" section for the full explanation.

**Files:**

- Create: `src/denoisr/training/diffusion_trainer.py`
- Test: `tests/test_training/test_diffusion_trainer.py`

**Step 1: Write failing tests**

`tests/test_training/test_diffusion_trainer.py`:

```python
import pytest
import torch

from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.training.diffusion_trainer import DiffusionTrainer

from tests.conftest import (
    SMALL_D_S,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


class TestDiffusionTrainer:
    @pytest.fixture
    def trainer(self, device: torch.device) -> DiffusionTrainer:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        diffusion = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        schedule = CosineNoiseSchedule(num_timesteps=SMALL_NUM_TIMESTEPS)
        return DiffusionTrainer(
            encoder=encoder,
            diffusion=diffusion,
            schedule=schedule,
            lr=1e-3,
            device=device,
        )

    def test_train_step_returns_loss(
        self, trainer: DiffusionTrainer, device: torch.device
    ) -> None:
        trajectory = torch.randn(2, 5, 12, 8, 8, device=device)
        loss = trainer.train_step(trajectory)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_is_finite(
        self, trainer: DiffusionTrainer, device: torch.device
    ) -> None:
        trajectory = torch.randn(2, 3, 12, 8, 8, device=device)
        loss = trainer.train_step(trajectory)
        assert not (loss != loss)  # NaN check

    def test_loss_decreases(
        self, trainer: DiffusionTrainer, device: torch.device
    ) -> None:
        trajectory = torch.randn(2, 4, 12, 8, 8, device=device)
        losses = [trainer.train_step(trajectory) for _ in range(30)]
        assert losses[-1] < losses[0]
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_training/test_diffusion_trainer.py -v`

**Step 3: Implement**

`src/denoisr/training/diffusion_trainer.py`:

```python
import torch
from torch import Tensor, nn

from denoisr.nn.diffusion import CosineNoiseSchedule


class DiffusionTrainer:
    """Trains the diffusion module to denoise future latent trajectories.

    Given a trajectory of board tensors, encodes them into latent space,
    corrupts future states with DDPM noise, and trains the diffusion
    model to predict the noise. The current state serves as the condition.
    """

    def __init__(
        self,
        encoder: nn.Module,
        diffusion: nn.Module,
        schedule: CosineNoiseSchedule,
        lr: float = 1e-4,
        device: torch.device | None = None,
    ) -> None:
        self.encoder = encoder
        self.diffusion = diffusion
        self.schedule = schedule
        self.device = device or torch.device("cpu")

        params = list(diffusion.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def train_step(self, trajectories: Tensor) -> float:
        """Train on a batch of board tensor trajectories.

        trajectories: [B, T, C, 8, 8] where T is consecutive board states.
        Uses position 0 as condition, a random later position as target.
        """
        B, T, C, H, W = trajectories.shape

        self.encoder.eval()
        self.diffusion.train()

        with torch.no_grad():
            flat = trajectories.reshape(B * T, C, H, W)
            latent_flat = self.encoder(flat)
            latent = latent_flat.reshape(B, T, 64, -1)

        cond = latent[:, 0]

        target_idx = torch.randint(1, T, (B,), device=self.device)
        target = torch.stack(
            [latent[b, target_idx[b]] for b in range(B)]
        )

        t = torch.randint(
            0, self.schedule.num_timesteps, (B,), device=self.device
        )
        noise = torch.randn_like(target)
        noisy_target = self.schedule.q_sample(target, t, noise)

        predicted_noise = self.diffusion(noisy_target, t, cond)

        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_diffusion_trainer.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/training/diffusion_trainer.py tests/test_training/test_diffusion_trainer.py
git commit -m "feat: add DDPM diffusion trainer (T6)"
```

---

### Task 6: Self-Play Actor

**Files:**

- Create: `src/denoisr/training/self_play.py`
- Test: `tests/test_training/test_self_play.py`

**Step 1: Write failing tests**

`tests/test_training/test_self_play.py`:

```python
import pytest
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.game.chess_game import ChessGame
from denoisr.training.self_play import SelfPlayActor, SelfPlayConfig

from tests.conftest import SMALL_D_S


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
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_training/test_self_play.py -v`

**Step 3: Implement**

`src/denoisr/training/self_play.py`:

```python
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
    Spec: early exploration → late exploitation, decaying over training.
    """
    base: float = 1.0
    explore_moves: int = 30
    generation_decay: float = 0.97

    def get_temperature(self, move_number: int, generation: int = 0) -> float:
        base = self.base * (self.generation_decay ** generation)
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

    def play_game(self) -> GameRecord:
        board = self._game.get_init_board()
        actions: list[Action] = []

        for _ in range(self._config.max_moves):
            result = self._game.get_game_ended(board)
            if result is not None:
                return GameRecord(actions=tuple(actions), result=result)

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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_self_play.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/training/self_play.py tests/test_training/test_self_play.py
git commit -m "feat: add self-play actor with MCTS (T6)"
```

---

### Task 7: Chess Engine (Inference)

**Files:**

- Create: `src/denoisr/inference/engine.py`
- Test: `tests/test_inference/test_engine.py`

**Step 1: Write failing tests**

`tests/test_inference/test_engine.py`:

```python
import chess
import pytest
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.inference.engine import ChessEngine
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead

from tests.conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


class TestChessEngine:
    @pytest.fixture
    def engine(self, device: torch.device) -> ChessEngine:
        return ChessEngine(
            encoder=ChessEncoder(12, SMALL_D_S).to(device),
            backbone=ChessPolicyBackbone(
                SMALL_D_S,
                SMALL_NUM_HEADS,
                SMALL_NUM_LAYERS,
                SMALL_FFN_DIM,
            ).to(device),
            policy_head=ChessPolicyHead(SMALL_D_S).to(device),
            value_head=ChessValueHead(SMALL_D_S).to(device),
            board_encoder=SimpleBoardEncoder(),
            device=device,
        )

    def test_select_move_returns_legal_move(
        self, engine: ChessEngine
    ) -> None:
        board = chess.Board()
        move = engine.select_move(board)
        assert move in board.legal_moves

    def test_select_move_various_positions(
        self, engine: ChessEngine
    ) -> None:
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3"):
            board.push_uci(uci)
        move = engine.select_move(board)
        assert move in board.legal_moves

    def test_evaluate_returns_wdl(self, engine: ChessEngine) -> None:
        board = chess.Board()
        wdl = engine.evaluate(board)
        assert len(wdl) == 3
        assert abs(sum(wdl) - 1.0) < 1e-5
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_inference/test_engine.py -v`

**Step 3: Implement**

`src/denoisr/inference/engine.py`:

```python
import chess
import torch
from torch import nn

from denoisr.data.board_encoder import SimpleBoardEncoder


class ChessEngine:
    """Combines encoder + backbone + heads to select chess moves.

    Single-pass inference (no diffusion or MCTS). The simplest
    inference mode. Diffusion-enhanced and MCTS-enhanced modes
    can be added by extending this class.
    """

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        board_encoder: SimpleBoardEncoder,
        device: torch.device | None = None,
    ) -> None:
        self._encoder = encoder
        self._backbone = backbone
        self._policy_head = policy_head
        self._value_head = value_head
        self._board_encoder = board_encoder
        self._device = device or torch.device("cpu")

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        self._encoder.eval()
        self._backbone.eval()
        self._policy_head.eval()

        board_tensor = self._board_encoder.encode(board).data
        x = board_tensor.unsqueeze(0).to(self._device)

        latent = self._encoder(x)
        features = self._backbone(latent)
        logits = self._policy_head(features).squeeze(0)

        legal_mask = torch.full((64, 64), float("-inf"))
        for move in board.legal_moves:
            legal_mask[move.from_square, move.to_square] = 0.0
        legal_mask = legal_mask.to(self._device)

        masked_logits = logits + legal_mask
        probs = torch.softmax(masked_logits.reshape(-1), dim=0)
        idx = torch.multinomial(probs, 1).item()

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

        return chess.Move(from_sq, to_sq, promotion)

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[float, float, float]:
        self._encoder.eval()
        self._backbone.eval()
        self._value_head.eval()

        board_tensor = self._board_encoder.encode(board).data
        x = board_tensor.unsqueeze(0).to(self._device)

        latent = self._encoder(x)
        features = self._backbone(latent)
        wdl, _ = self._value_head(features)
        wdl = wdl.squeeze(0)

        return (wdl[0].item(), wdl[1].item(), wdl[2].item())
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_inference/test_engine.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/inference/engine.py tests/test_inference/test_engine.py
git commit -m "feat: add chess engine for single-pass inference (T7)"
```

---

### Task 8: UCI Protocol Wrapper

**Files:**

- Create: `src/denoisr/inference/uci.py`
- Test: `tests/test_inference/test_uci.py`

**Step 1: Write failing tests**

`tests/test_inference/test_uci.py`:

```python
from denoisr.inference.uci import format_bestmove, parse_go, parse_position


class TestParsePosition:
    def test_startpos(self) -> None:
        fen, moves = parse_position("position startpos")
        assert fen is None
        assert moves == []

    def test_startpos_with_moves(self) -> None:
        fen, moves = parse_position("position startpos moves e2e4 e7e5")
        assert fen is None
        assert moves == ["e2e4", "e7e5"]

    def test_fen(self) -> None:
        cmd = "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        fen, moves = parse_position(cmd)
        assert fen is not None
        assert "rnbqkbnr" in fen
        assert moves == []

    def test_fen_with_moves(self) -> None:
        cmd = "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 moves e7e5"
        fen, moves = parse_position(cmd)
        assert fen is not None
        assert moves == ["e7e5"]


class TestParseGo:
    def test_movetime(self) -> None:
        params = parse_go("go movetime 1000")
        assert params["movetime"] == 1000

    def test_depth(self) -> None:
        params = parse_go("go depth 10")
        assert params["depth"] == 10

    def test_infinite(self) -> None:
        params = parse_go("go infinite")
        assert params.get("infinite") is True

    def test_wtime_btime(self) -> None:
        params = parse_go("go wtime 60000 btime 60000 winc 1000 binc 1000")
        assert params["wtime"] == 60000
        assert params["btime"] == 60000


class TestFormatBestmove:
    def test_simple_move(self) -> None:
        assert format_bestmove("e2e4") == "bestmove e2e4"

    def test_promotion(self) -> None:
        assert format_bestmove("a7a8q") == "bestmove a7a8q"
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_inference/test_uci.py -v`

**Step 3: Implement**

`src/denoisr/inference/uci.py`:

```python
"""UCI (Universal Chess Interface) protocol parser and formatter.

Handles parsing UCI commands from stdin and formatting responses.
The actual engine logic is in engine.py.
"""


def parse_position(command: str) -> tuple[str | None, list[str]]:
    """Parse a UCI 'position' command.

    Returns (fen_or_none, list_of_uci_moves).
    """
    parts = command.split()
    fen = None
    moves: list[str] = []

    if "startpos" in parts:
        if "moves" in parts:
            moves_idx = parts.index("moves")
            moves = parts[moves_idx + 1 :]
    elif "fen" in parts:
        fen_idx = parts.index("fen")
        if "moves" in parts:
            moves_idx = parts.index("moves")
            fen = " ".join(parts[fen_idx + 1 : moves_idx])
            moves = parts[moves_idx + 1 :]
        else:
            fen = " ".join(parts[fen_idx + 1 :])

    return fen, moves


def parse_go(command: str) -> dict[str, int | bool]:
    """Parse a UCI 'go' command into parameters."""
    parts = command.split()
    params: dict[str, int | bool] = {}

    int_keys = {
        "movetime",
        "depth",
        "nodes",
        "wtime",
        "btime",
        "winc",
        "binc",
        "movestogo",
    }

    i = 1
    while i < len(parts):
        token = parts[i]
        if token == "infinite":
            params["infinite"] = True
        elif token in int_keys and i + 1 < len(parts):
            params[token] = int(parts[i + 1])
            i += 1
        i += 1

    return params


def format_bestmove(uci_move: str) -> str:
    """Format a UCI bestmove response."""
    return f"bestmove {uci_move}"


def run_uci_loop(
    engine_select_move_fn: object,
) -> None:
    """Main UCI loop reading from stdin.

    Connect to any UCI-compatible GUI (CuteChess, Arena, etc.).
    engine_select_move_fn: callable(chess.Board) -> chess.Move.
    """
    import sys

    import chess

    board = chess.Board()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line == "uci":
            print("id name denoisr")
            print("id author denoisr-team")
            print("uciok")

        elif line == "isready":
            print("readyok")

        elif line.startswith("position"):
            fen, moves = parse_position(line)
            board = chess.Board(fen) if fen else chess.Board()
            for uci in moves:
                board.push_uci(uci)

        elif line.startswith("go"):
            move = engine_select_move_fn(board)
            print(format_bestmove(move.uci()))

        elif line == "quit":
            break

        sys.stdout.flush()
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_inference/test_uci.py -v`
Expected: All pass.

**Step 5: Run ALL tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests across T1-T7 pass.

**Step 6: Commit**

```bash
git add src/denoisr/inference/uci.py tests/test_inference/test_uci.py
git commit -m "feat: add UCI protocol wrapper (T7)"
```

---

### Task 9: Diffusion-Enhanced Inference Engine

**Spec reference:** "Encoder → diffusion imagines future → enriched policy prediction" and "anytime search: adjustable denoising steps."

**Rationale:** The spec's core innovation is using diffusion to imagine future trajectories, enriching policy prediction. The base `ChessEngine` (Task 7) does single-pass inference. This task adds the DiffuSearch-style diffusion-enhanced mode with adjustable compute.

**Intuition — how denoising becomes move scoring:** At inference time, the engine starts with the real current position and pure noise for the future: `[Real position] → ???? → ???? → ????`. It then iteratively denoises (e.g. 20 steps), progressively resolving the future trajectory until a complete continuation emerges: `[Real pos] → Nf3 → [pos₂] → d5 → [pos₃] → Bg5 → [pos₄]`. The first move of this denoised trajectory is the engine's chosen move. Crucially, **the model never assigns an explicit score to any move** — the scoring is implicit in _which trajectory_ the denoising process converges to. More denoising steps means more refined futures (anytime search), so `num_denoising_steps` directly controls the strength/speed tradeoff. See the component design doc's "How the Diffusion Model Learns to Score Moves Through Self-Play" section for the full explanation.

**Files:**

- Create: `src/denoisr/inference/diffusion_engine.py`
- Test: `tests/test_inference/test_diffusion_engine.py`

**Step 1: Write failing tests**

`tests/test_inference/test_diffusion_engine.py`:

```python
import chess
import pytest
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.inference.diffusion_engine import DiffusionChessEngine
from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead

from tests.conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


class TestDiffusionChessEngine:
    @pytest.fixture
    def engine(self, device: torch.device) -> DiffusionChessEngine:
        return DiffusionChessEngine(
            encoder=ChessEncoder(12, SMALL_D_S).to(device),
            backbone=ChessPolicyBackbone(
                SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM
            ).to(device),
            policy_head=ChessPolicyHead(SMALL_D_S).to(device),
            value_head=ChessValueHead(SMALL_D_S).to(device),
            diffusion=ChessDiffusionModule(
                SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_NUM_TIMESTEPS
            ).to(device),
            schedule=CosineNoiseSchedule(SMALL_NUM_TIMESTEPS),
            board_encoder=SimpleBoardEncoder(),
            device=device,
            num_denoising_steps=5,
        )

    def test_select_move_returns_legal(
        self, engine: DiffusionChessEngine
    ) -> None:
        board = chess.Board()
        move = engine.select_move(board)
        assert move in board.legal_moves

    def test_evaluate_returns_wdl(
        self, engine: DiffusionChessEngine
    ) -> None:
        board = chess.Board()
        wdl = engine.evaluate(board)
        assert len(wdl) == 3
        assert abs(sum(wdl) - 1.0) < 1e-5

    def test_anytime_property(self, device: torch.device) -> None:
        """Different denoising step counts produce different (potentially better) results."""
        args = dict(
            encoder=ChessEncoder(12, SMALL_D_S).to(device),
            backbone=ChessPolicyBackbone(
                SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM
            ).to(device),
            policy_head=ChessPolicyHead(SMALL_D_S).to(device),
            value_head=ChessValueHead(SMALL_D_S).to(device),
            diffusion=ChessDiffusionModule(
                SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_NUM_TIMESTEPS
            ).to(device),
            schedule=CosineNoiseSchedule(SMALL_NUM_TIMESTEPS),
            board_encoder=SimpleBoardEncoder(),
            device=device,
        )
        engine_1 = DiffusionChessEngine(**args, num_denoising_steps=1)
        engine_10 = DiffusionChessEngine(**args, num_denoising_steps=10)
        board = chess.Board()
        # Both should return legal moves
        assert engine_1.select_move(board) in board.legal_moves
        assert engine_10.select_move(board) in board.legal_moves
```

**Step 2: Implement**

`src/denoisr/inference/diffusion_engine.py`:

```python
import chess
import torch
from torch import nn

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.nn.diffusion import CosineNoiseSchedule


class DiffusionChessEngine:
    """DiffuSearch-style chess engine with diffusion-enhanced inference.

    Combines encoder + diffusion imagination + policy backbone:
    1. Encode current board to latent state
    2. Run N denoising steps to imagine future trajectory
    3. Fuse current latent with denoised future
    4. Run policy backbone + head on fused representation

    The num_denoising_steps parameter gives anytime search:
    more steps = stronger but slower inference.
    """

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        diffusion: nn.Module,
        schedule: CosineNoiseSchedule,
        board_encoder: SimpleBoardEncoder,
        device: torch.device | None = None,
        num_denoising_steps: int = 10,
    ) -> None:
        self._encoder = encoder
        self._backbone = backbone
        self._policy_head = policy_head
        self._value_head = value_head
        self._diffusion = diffusion
        self._schedule = schedule
        self._board_encoder = board_encoder
        self._device = device or torch.device("cpu")
        self._num_steps = num_denoising_steps

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        self._set_eval()
        latent = self._encode_board(board)
        enriched = self._diffusion_imagine(latent)
        features = self._backbone(enriched)
        logits = self._policy_head(features).squeeze(0)

        legal_mask = torch.full((64, 64), float("-inf"))
        for move in board.legal_moves:
            legal_mask[move.from_square, move.to_square] = 0.0
        legal_mask = legal_mask.to(self._device)

        probs = torch.softmax((logits + legal_mask).reshape(-1), dim=0)
        idx = torch.multinomial(probs, 1).item()
        from_sq, to_sq = idx // 64, idx % 64

        promotion = None
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion)

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[float, float, float]:
        self._set_eval()
        latent = self._encode_board(board)
        enriched = self._diffusion_imagine(latent)
        features = self._backbone(enriched)
        wdl, _ = self._value_head(features)
        wdl = wdl.squeeze(0)
        return (wdl[0].item(), wdl[1].item(), wdl[2].item())

    def _encode_board(self, board: chess.Board) -> torch.Tensor:
        board_tensor = self._board_encoder.encode(board).data
        return self._encoder(board_tensor.unsqueeze(0).to(self._device))

    def _diffusion_imagine(self, latent: torch.Tensor) -> torch.Tensor:
        """Run iterative denoising to imagine future trajectories."""
        x = torch.randn_like(latent)
        step_size = max(1, self._schedule.num_timesteps // self._num_steps)

        for i in range(self._num_steps):
            t_val = max(0, self._schedule.num_timesteps - 1 - i * step_size)
            t = torch.tensor([t_val], device=self._device)
            noise_pred = self._diffusion(x, t, latent)
            ab = self._schedule.alpha_bar.to(self._device)[t_val]
            x = (x - (1 - ab).sqrt() * noise_pred) / ab.sqrt()

        # Fuse: average current latent with denoised future
        return (latent + x) / 2

    def _set_eval(self) -> None:
        self._encoder.eval()
        self._backbone.eval()
        self._policy_head.eval()
        self._value_head.eval()
        self._diffusion.eval()
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_inference/test_diffusion_engine.py -v`

**Step 4: Commit**

```bash
git add src/denoisr/inference/diffusion_engine.py tests/test_inference/test_diffusion_engine.py
git commit -m "feat: add diffusion-enhanced chess engine with anytime search (T7)"
```

---

### Task 10: MuZero Reanalyse Actor

**Spec reference:** "Process 2: Reanalyse — re-run MCTS on old trajectories with the latest network."

**Rationale:** Reanalyse improves sample efficiency by replaying old game trajectories through the current (improved) network's MCTS, generating higher-quality policy targets without playing new games. This is one of MuZero's key efficiency innovations.

**Files:**

- Create: `src/denoisr/training/reanalyse.py`
- Test: `tests/test_training/test_reanalyse.py`

**Step 1: Write failing tests**

`tests/test_training/test_reanalyse.py`:

```python
import chess
import pytest
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.game.chess_game import ChessGame
from denoisr.training.reanalyse import ReanalyseActor
from denoisr.types import Action, GameRecord, TrainingExample

from tests.conftest import SMALL_D_S


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


class TestReanalyseActor:
    @pytest.fixture
    def actor(self) -> ReanalyseActor:
        model = _DummyModel(SMALL_D_S)
        return ReanalyseActor(
            policy_value_fn=model.predict,
            world_model_fn=model.predict_next,
            encode_fn=model.encode,
            game=ChessGame(),
            board_encoder=SimpleBoardEncoder(),
            num_simulations=10,
        )

    def test_reanalyse_produces_examples(
        self, actor: ReanalyseActor
    ) -> None:
        move = chess.Move.from_uci("e2e4")
        record = GameRecord(
            actions=(Action(move.from_square, move.to_square),),
            result=1.0,
        )
        examples = actor.reanalyse(record)
        assert len(examples) == 1
        assert isinstance(examples[0], TrainingExample)

    def test_policy_targets_are_distributions(
        self, actor: ReanalyseActor
    ) -> None:
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
```

**Step 2: Implement**

`src/denoisr/training/reanalyse.py`:

```python
from typing import Callable

import chess
import torch
from torch import Tensor

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.game.chess_game import ChessGame
from denoisr.training.mcts import MCTS, MCTSConfig
from denoisr.types import (
    Action,
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
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_training/test_reanalyse.py -v`

**Step 4: Commit**

```bash
git add src/denoisr/training/reanalyse.py tests/test_training/test_reanalyse.py
git commit -m "feat: add MuZero reanalyse actor (T6)"
```

---

### Task 11: Phase Transition Orchestrator

**Spec reference:** Phase gates (30% top-1, +5pp diffusion), α mixing (MCTS→diffusion, α: 0→1).

**Intuition — the MCTS bootstrap makes diffusion practical:** In early training, self-play games are terrible (random play → random trajectories → no useful signal for the diffusion model). This is why Phase 3 uses a two-step approach: **Phase 3a (MCTS bootstrap)** — traditional MCTS generates self-play games. MCTS provides decent move quality even with a weak neural network, so the diffusion model gets meaningful winning/losing patterns from the start. **Phase 3b (Diffusion transition)** — once the diffusion model has absorbed enough patterns, it gradually takes over move selection via α mixing (linearly 0→1). Now its _own_ games produce the training data, creating a virtuous cycle: better diffusion → better games → better training data → better diffusion. See the component design doc's "How the Diffusion Model Learns to Score Moves Through Self-Play" section for the full explanation.

**Files:**

- Create: `src/denoisr/training/phase_orchestrator.py`
- Test: `tests/test_training/test_phase_orchestrator.py`

**Step 1: Write failing tests**

`tests/test_training/test_phase_orchestrator.py`:

```python
import pytest
import torch

from denoisr.training.phase_orchestrator import PhaseOrchestrator, PhaseConfig


class TestPhaseOrchestrator:
    @pytest.fixture
    def orchestrator(self) -> PhaseOrchestrator:
        return PhaseOrchestrator(PhaseConfig())

    def test_starts_at_phase_1(
        self, orchestrator: PhaseOrchestrator
    ) -> None:
        assert orchestrator.current_phase == 1

    def test_phase_1_to_2_gate(
        self, orchestrator: PhaseOrchestrator
    ) -> None:
        assert not orchestrator.check_gate(
            {"top1_accuracy": 0.25}
        )
        assert orchestrator.check_gate(
            {"top1_accuracy": 0.35}
        )
        assert orchestrator.current_phase == 2

    def test_phase_2_to_3_gate(self) -> None:
        o = PhaseOrchestrator(PhaseConfig())
        o.check_gate({"top1_accuracy": 0.35})
        assert o.current_phase == 2
        assert not o.check_gate(
            {"diffusion_improvement_pp": 3.0}
        )
        assert o.check_gate(
            {"diffusion_improvement_pp": 6.0}
        )
        assert o.current_phase == 3

    def test_alpha_mixing(self) -> None:
        o = PhaseOrchestrator(PhaseConfig(alpha_generations=10))
        o.check_gate({"top1_accuracy": 0.35})
        o.check_gate({"diffusion_improvement_pp": 6.0})
        assert o.current_phase == 3
        assert o.get_alpha(generation=0) == 0.0
        assert o.get_alpha(generation=5) == pytest.approx(0.5)
        assert o.get_alpha(generation=10) == pytest.approx(1.0)
        assert o.get_alpha(generation=20) == pytest.approx(1.0)

    def test_mixed_policy(self) -> None:
        o = PhaseOrchestrator(PhaseConfig())
        mcts_policy = torch.zeros(64, 64)
        mcts_policy[12, 28] = 1.0
        diff_policy = torch.zeros(64, 64)
        diff_policy[12, 20] = 1.0
        mixed = o.mix_policies(mcts_policy, diff_policy, alpha=0.5)
        assert mixed[12, 28].item() == pytest.approx(0.5)
        assert mixed[12, 20].item() == pytest.approx(0.5)
```

**Step 2: Implement**

`src/denoisr/training/phase_orchestrator.py`:

```python
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for phase transition gates and α mixing.

    Gates:
    - Phase 1→2: top1_accuracy > phase1_gate (default 0.30)
    - Phase 2→3: diffusion_improvement_pp > phase2_gate (default 5.0)

    α mixing (Phase 3):
    - α linearly increases from 0 to 1 over alpha_generations
    - final_policy = (1-α) * mcts_policy + α * diffusion_policy
    """
    phase1_gate: float = 0.30
    phase2_gate: float = 5.0
    alpha_generations: int = 50


class PhaseOrchestrator:
    """Manages training phase transitions and MCTS→diffusion α mixing."""

    def __init__(self, config: PhaseConfig) -> None:
        self._config = config
        self._phase = 1

    @property
    def current_phase(self) -> int:
        return self._phase

    def check_gate(self, metrics: dict[str, float]) -> bool:
        if self._phase == 1:
            if metrics.get("top1_accuracy", 0) > self._config.phase1_gate:
                self._phase = 2
                return True
        elif self._phase == 2:
            if metrics.get("diffusion_improvement_pp", 0) > self._config.phase2_gate:
                self._phase = 3
                return True
        return False

    def get_alpha(self, generation: int) -> float:
        if self._phase < 3:
            return 0.0
        return min(1.0, generation / max(1, self._config.alpha_generations))

    def mix_policies(
        self, mcts_policy: Tensor, diffusion_policy: Tensor, alpha: float
    ) -> Tensor:
        return (1 - alpha) * mcts_policy + alpha * diffusion_policy
```

**Step 3: Run tests, commit**

```bash
uv run pytest tests/test_training/test_phase_orchestrator.py -v
git add src/denoisr/training/phase_orchestrator.py tests/test_training/test_phase_orchestrator.py
git commit -m "feat: add phase orchestrator with gates and alpha mixing (T6)"
```

---

### Task 12: Benchmarking Harness (cutechess-cli + ordo)

**Spec reference:** "cutechess-cli with SPRT statistical testing for Elo measurement." The companion `ordo` tool computes Elo differences from PGN result files — use it alongside cutechess-cli for multi-engine round-robin tournaments.

**Files:**

- Create: `src/denoisr/evaluation/__init__.py`
- Create: `src/denoisr/evaluation/benchmark.py`
- Test: `tests/test_evaluation/test_benchmark.py`

**Step 1: Write failing tests**

`tests/test_evaluation/test_benchmark.py`:

```python
import pytest

from denoisr.evaluation.benchmark import BenchmarkConfig, build_cutechess_command, parse_cutechess_output


class TestBuildCommand:
    def test_basic_command(self) -> None:
        config = BenchmarkConfig(
            engine_cmd="./denoisr",
            opponent_cmd="stockfish",
            games=100,
            time_control="10+0.1",
        )
        cmd = build_cutechess_command(config)
        assert "cutechess-cli" in cmd
        assert "-games 100" in cmd
        assert "-engine cmd=./denoisr" in cmd
        assert "-engine cmd=stockfish" in cmd

    def test_sprt_parameters(self) -> None:
        config = BenchmarkConfig(
            engine_cmd="./denoisr",
            opponent_cmd="stockfish",
            games=1000,
            time_control="10+0.1",
            sprt_elo0=0,
            sprt_elo1=50,
        )
        cmd = build_cutechess_command(config)
        assert "sprt" in cmd


class TestParseOutput:
    def test_parse_elo(self) -> None:
        output = "Elo difference: 42.3 +/- 15.1, LOS: 99.2 %, DrawRatio: 30.5 %"
        result = parse_cutechess_output(output)
        assert abs(result["elo_diff"] - 42.3) < 0.1
        assert abs(result["elo_error"] - 15.1) < 0.1

    def test_parse_sprt_accept(self) -> None:
        output = "SPRT: llr 2.97 (100.0%), lbound -2.94, ubound 2.94 - H1 was accepted"
        result = parse_cutechess_output(output)
        assert result["sprt_result"] == "H1"
```

**Step 2: Implement**

`src/denoisr/evaluation/benchmark.py`:

```python
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    engine_cmd: str
    opponent_cmd: str
    games: int = 100
    time_control: str = "10+0.1"
    sprt_elo0: int | None = None
    sprt_elo1: int | None = None
    concurrency: int = 1


def build_cutechess_command(config: BenchmarkConfig) -> str:
    parts = [
        "cutechess-cli",
        f"-engine cmd={config.engine_cmd} proto=uci",
        f"-engine cmd={config.opponent_cmd} proto=uci",
        f"-games {config.games}",
        f"-each tc={config.time_control}",
        f"-concurrency {config.concurrency}",
    ]
    if config.sprt_elo0 is not None and config.sprt_elo1 is not None:
        parts.append(
            f"-sprt elo0={config.sprt_elo0} elo1={config.sprt_elo1} alpha=0.05 beta=0.05"
        )
    return " ".join(parts)


def parse_cutechess_output(output: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}

    elo_match = re.search(
        r"Elo difference: ([-\d.]+) \+/- ([\d.]+)", output
    )
    if elo_match:
        result["elo_diff"] = float(elo_match.group(1))
        result["elo_error"] = float(elo_match.group(2))

    los_match = re.search(r"LOS: ([\d.]+)", output)
    if los_match:
        result["los"] = float(los_match.group(1))

    if "H1 was accepted" in output:
        result["sprt_result"] = "H1"
    elif "H0 was accepted" in output:
        result["sprt_result"] = "H0"

    return result
```

**Step 3: Run tests, commit**

```bash
uv run pytest tests/test_evaluation/test_benchmark.py -v
git add src/denoisr/evaluation/ tests/test_evaluation/
git commit -m "feat: add cutechess-cli benchmarking harness"
```

---

### Task 13: Gradient Flow Mitigations

**Spec reference:** "Stop-gradient boundaries, separate learning rates per loss term, curriculum over diffusion steps."

**Rationale:** Multi-objective training needs careful gradient management. This task modifies existing trainers to use parameter groups with separate learning rates and adds gradient clipping.

**Files:**

- Modify: `src/denoisr/training/supervised_trainer.py` (parameter groups)
- Modify: `src/denoisr/training/diffusion_trainer.py` (diffusion step curriculum)
- Test: extend existing trainer tests

**Changes to `SupervisedTrainer.__init__`:**

```python
# Replace single AdamW with parameter groups
param_groups = [
    {"params": list(encoder.parameters()), "lr": lr * 0.1},      # slower encoder
    {"params": list(backbone.parameters()), "lr": lr * 0.3},     # moderate backbone
    {"params": list(policy_head.parameters()), "lr": lr},         # full LR for heads
    {"params": list(value_head.parameters()), "lr": lr},
]
self.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
self.max_grad_norm = 1.0  # gradient clipping
```

**Add to `SupervisedTrainer.train_step` (after `loss.backward()`):**

```python
torch.nn.utils.clip_grad_norm_(
    [p for group in self.optimizer.param_groups for p in group["params"]],
    self.max_grad_norm,
)
```

**Changes to `DiffusionTrainer.__init__`:**

```python
# Diffusion step curriculum: start with fewer steps, increase over epochs
self._curriculum_max_steps = num_timesteps
self._current_max_steps = max(1, num_timesteps // 4)  # start at 25%
self._curriculum_growth = 1.02  # 2% increase per epoch
```

**Add `DiffusionTrainer.advance_curriculum()` method:**

```python
def advance_curriculum(self) -> None:
    """Call once per epoch to increase diffusion step difficulty."""
    self._current_max_steps = min(
        self._curriculum_max_steps,
        int(self._current_max_steps * self._curriculum_growth),
    )
```

**Tests to add:**

- `test_encoder_lr_lower_than_head_lr`: verify parameter groups have different LRs
- `test_gradients_are_clipped`: verify no gradient exceeds max norm after step
- `test_curriculum_increases_over_epochs`: verify `_current_max_steps` grows

---

## Milestones Summary

| Milestone        | Gate Condition                                  | Task                             |
| ---------------- | ----------------------------------------------- | -------------------------------- |
| Phase 1 complete | Policy accuracy >30% top-1 on 10k held-out      | Task 4 (supervised trainer) eval |
| Phase 2 complete | Diffusion accuracy > single-step by >5pp        | Task 9 (diffusion engine) eval   |
| Phase 3a stable  | Elo increases over generations                  | Task 12 (benchmarking)           |
| Phase 3b target  | Diffusion-only matches MCTS strength            | Task 11 (orchestrator) + Task 12 |
| AlphaVile bonus  | Extended features add >50 Elo vs simple encoder | Task 12 vs baseline              |

---

## Full Project Gate Check

```bash
uv run pytest tests/ -v --tb=short
```

**Expected:** All tests pass across all 7 tiers:

- **T1**: Domain types with shape invariants and immutability
- **T2**: Game interface, encoders (simple + extended AlphaVile), bijective round-trips
- **T3**: Data pipeline with PGN streaming and Stockfish oracle
- **T4**: Standalone nn modules (encoder, smolgen, Shaw PE, WDLP value head, policy head)
- **T5**: Composite nn modules (backbone with smolgen+Shaw, world model, diffusion)
- **T6**: Training infrastructure (6-term loss + HarmonyDream, MCTS, priority replay, supervised/diffusion trainers, self-play with temperature schedule, reanalyse, phase orchestrator, gradient mitigations)
- **T7**: Inference (single-pass engine, diffusion-enhanced engine with anytime search, UCI protocol, cutechess-cli benchmarking)

**Next steps:** Execute Phase 1 training with Lichess data + Stockfish targets. Measure policy accuracy against the 30% top-1 gate threshold. Use phase orchestrator to manage transitions.
