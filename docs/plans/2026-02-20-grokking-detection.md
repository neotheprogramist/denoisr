# Grokking Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Instrument Phase 1 supervised training with grokking detection metrics, structured holdout splits, adaptive evaluation frequency, console alerts, and optional Grokfast gradient acceleration.

**Architecture:** A standalone `GrokTracker` class registers forward hooks on backbone layers and computes Tier 1 (weight/gradient norms, loss gap) and Tier 2 (effective rank, spectral norms, HTSR alpha) metrics at configurable frequencies. A 4-state machine (BASELINE → ONSET → TRANSITIONING → GROKKED) drives adaptive evaluation frequency and console alerts. `GrokfastFilter` provides opt-in EMA gradient filtering for ~50× grokking speedup. `StratifiedHoldoutSplitter` creates game-level, opening-family, and piece-count holdout sets from enriched `TrainingExample` metadata.

**Tech Stack:** PyTorch (SVD, hooks, gradient access), python-chess (ECO extraction), existing TensorBoard logger, frozen dataclasses for config.

---

### Task 1: Extend TrainingExample with holdout metadata

**Files:**
- Modify: `src/denoisr/types/training.py:32-36`
- Modify: `src/denoisr/scripts/generate_data.py:56-76,137-167`
- Test: `tests/test_types/test_training_types.py`

**Step 1: Write failing test for new TrainingExample fields**

Add to `tests/test_types/test_training_types.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types/test_training_types.py::TestTrainingExampleMetadata -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'game_id'`

**Step 3: Add metadata fields to TrainingExample**

In `src/denoisr/types/training.py`, modify the `TrainingExample` dataclass (line 32):

```python
@dataclass(frozen=True)
class TrainingExample:
    board: BoardTensor
    policy: PolicyTarget
    value: ValueTarget
    game_id: int | None = None
    eco_code: str | None = None
    piece_count: int | None = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_types/test_training_types.py -v`
Expected: all PASS

**Step 5: Write failing test for stack/unstack roundtrip with metadata**

Add to `tests/test_types/test_training_types.py`:

```python
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
        """Old .pt files without metadata fields should still load."""
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
```

**Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_types/test_training_types.py::TestStackUnstackMetadata -v`
Expected: FAIL — `"game_ids" not in stacked`

**Step 7: Update stack_examples and unstack_examples**

In `src/denoisr/scripts/generate_data.py`, update `stack_examples` (line 137):

```python
def stack_examples(
    examples: list[TrainingExample],
) -> dict[str, torch.Tensor | list[str | None]]:
    boards = torch.stack([ex.board.data for ex in examples])
    policies = torch.stack([ex.policy.data for ex in examples])
    values = torch.tensor(
        [[ex.value.win, ex.value.draw, ex.value.loss] for ex in examples],
        dtype=torch.float32,
    )
    result: dict[str, torch.Tensor | list[str | None]] = {
        "boards": boards,
        "policies": policies,
        "values": values,
    }
    # Metadata (optional — only present if data generation populated it)
    if any(ex.game_id is not None for ex in examples):
        result["game_ids"] = torch.tensor(
            [ex.game_id if ex.game_id is not None else -1 for ex in examples],
            dtype=torch.int64,
        )
    if any(ex.eco_code is not None for ex in examples):
        result["eco_codes"] = [ex.eco_code for ex in examples]
    if any(ex.piece_count is not None for ex in examples):
        result["piece_counts"] = torch.tensor(
            [ex.piece_count if ex.piece_count is not None else -1 for ex in examples],
            dtype=torch.int32,
        )
    return result
```

Update `unstack_examples` (line 149):

```python
def unstack_examples(
    data: dict[str, torch.Tensor | list[str | None]],
) -> list[TrainingExample]:
    boards = data["boards"]
    policies = data["policies"]
    values = data["values"]
    game_ids = data.get("game_ids")
    eco_codes = data.get("eco_codes")
    piece_counts = data.get("piece_counts")
    n = boards.shape[0]
    return [
        TrainingExample(
            board=BoardTensor(boards[i]),
            policy=PolicyTarget(policies[i]),
            value=ValueTarget(
                win=values[i, 0].item(),
                draw=values[i, 1].item(),
                loss=values[i, 2].item(),
            ),
            game_id=(
                int(game_ids[i].item())
                if game_ids is not None and game_ids[i].item() >= 0
                else None
            ),
            eco_code=(
                eco_codes[i]
                if eco_codes is not None
                else None
            ),
            piece_count=(
                int(piece_counts[i].item())
                if piece_counts is not None and piece_counts[i].item() >= 0
                else None
            ),
        )
        for i in range(n)
    ]
```

**Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_types/test_training_types.py -v`
Expected: all PASS

**Step 9: Update data generation to populate metadata**

In `src/denoisr/scripts/generate_data.py`, modify `_extract_positions` (line 82) to track game IDs, ECO codes, and piece counts. The position list should carry metadata alongside move sequences.

Change the position type and extraction:

```python
@dataclass(frozen=True)
class _PositionMeta:
    moves: _MoveSeq
    game_id: int
    eco_code: str | None
    piece_count: int


def _extract_positions(pgn_path: Path, max_positions: int) -> list[_PositionMeta]:
    streamer = SimplePGNStreamer()
    positions: list[_PositionMeta] = []
    pbar = tqdm(total=max_positions, desc="Extracting positions", unit="pos", smoothing=0.3)
    game_id = 0

    for record in streamer.stream(pgn_path):
        eco_code = record.eco_code  # Need to add this to GameRecord
        moves_so_far: _MoveSeq = []
        board = chess.Board()
        for action in record.actions:
            moves_so_far.append((action.from_square, action.to_square, action.promotion))
            board.push(chess.Move(action.from_square, action.to_square, action.promotion))
            if len(positions) >= max_positions:
                break
            piece_count = bin(board.occupied).count("1")
            positions.append(_PositionMeta(
                moves=list(moves_so_far),
                game_id=game_id,
                eco_code=eco_code,
                piece_count=piece_count,
            ))
            pbar.update(1)

        game_id += 1
        if len(positions) >= max_positions:
            break

    pbar.close()
    return positions
```

This also requires adding `eco_code` to `GameRecord` and extracting it from PGN headers in the streamer. See step 10.

**Step 10: Add ECO code extraction to PGN streamer and GameRecord**

In `src/denoisr/types/training.py`, add `eco_code` to `GameRecord`:

```python
@dataclass(frozen=True)
class GameRecord:
    actions: tuple[Action, ...]
    result: float
    eco_code: str | None = None
```

In `src/denoisr/data/pgn_streamer.py`, extract ECO from PGN headers (line 34):

```python
def _parse_games(
    self, stream: TextIO
) -> Iterator[GameRecord]:
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        result_str = game.headers.get("Result", "*")
        result = _RESULT_MAP.get(result_str)
        if result is None:
            continue
        eco_code = game.headers.get("ECO")
        actions = tuple(
            Action(m.from_square, m.to_square, m.promotion)
            for m in game.mainline_moves()
        )
        yield GameRecord(actions=actions, result=result, eco_code=eco_code)
```

Update `generate_examples` to use `_PositionMeta` and pass metadata through to `TrainingExample`. Update `_evaluate_position` to accept and return metadata.

**Step 11: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: all PASS

**Step 12: Commit**

```bash
git add src/denoisr/types/training.py src/denoisr/scripts/generate_data.py \
    src/denoisr/data/pgn_streamer.py tests/test_types/test_training_types.py
git commit -m "feat: add holdout metadata to TrainingExample and data generation"
```

---

### Task 2: Implement StratifiedHoldoutSplitter

**Files:**
- Create: `src/denoisr/data/holdout_splitter.py`
- Test: `tests/test_data/test_holdout_splitter.py`

**Step 1: Write failing tests**

Create `tests/test_data/test_holdout_splitter.py`:

```python
import torch
import pytest

from denoisr.data.holdout_splitter import StratifiedHoldoutSplitter, HoldoutSplits
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget


def _make_example(
    game_id: int | None = None,
    eco_code: str | None = None,
    piece_count: int | None = None,
) -> TrainingExample:
    return TrainingExample(
        board=BoardTensor(torch.randn(12, 8, 8)),
        policy=PolicyTarget(torch.zeros(64, 64)),
        value=ValueTarget(win=1.0, draw=0.0, loss=0.0),
        game_id=game_id,
        eco_code=eco_code,
        piece_count=piece_count,
    )


class TestStratifiedHoldoutSplitter:
    def test_returns_holdout_splits_dataclass(self) -> None:
        examples = [_make_example(game_id=i % 5, eco_code="B90", piece_count=30) for i in range(100)]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        assert isinstance(splits, HoldoutSplits)
        assert len(splits.train) > 0
        assert len(splits.random) > 0

    def test_game_level_holdout_no_overlap(self) -> None:
        """No game_id should appear in both train and game-level holdout."""
        examples = [_make_example(game_id=i % 20, eco_code="B90", piece_count=30) for i in range(200)]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        train_game_ids = {ex.game_id for ex in splits.train if ex.game_id is not None}
        holdout_game_ids = {ex.game_id for ex in splits.game_level if ex.game_id is not None}
        assert train_game_ids.isdisjoint(holdout_game_ids)

    def test_opening_family_holdout_no_overlap(self) -> None:
        """No ECO family (first letter) should appear in both train and opening holdout."""
        eco_codes = ["B90", "B91", "B92", "C50", "C51", "D30", "D31", "E60", "E61", "A00"]
        examples = [_make_example(game_id=i, eco_code=eco_codes[i % len(eco_codes)], piece_count=30) for i in range(200)]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        if splits.opening_family:
            train_ecos = {ex.eco_code for ex in splits.train if ex.eco_code}
            holdout_ecos = {ex.eco_code for ex in splits.opening_family if ex.eco_code}
            # At least the held-out ECOs should not appear in train
            assert holdout_ecos - train_ecos  # some ECOs are holdout-only

    def test_piece_count_holdout_all_endgame(self) -> None:
        """Piece-count holdout should contain only positions <= threshold."""
        examples = [_make_example(game_id=i, piece_count=p) for i, p in enumerate([32, 24, 16, 8, 5, 4, 3, 6, 2, 30])]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        for ex in splits.piece_count:
            assert ex.piece_count is not None
            assert ex.piece_count <= 6

    def test_fallback_when_no_metadata(self) -> None:
        """Without metadata, only random holdout should be populated."""
        examples = [_make_example() for _ in range(100)]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        assert len(splits.random) > 0
        assert len(splits.game_level) == 0
        assert len(splits.opening_family) == 0
        assert len(splits.piece_count) == 0

    def test_train_does_not_contain_holdout_examples(self) -> None:
        """Train set should not contain any examples from any holdout set."""
        examples = [_make_example(game_id=i % 10, eco_code="B90", piece_count=30) for i in range(100)]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.2, endgame_threshold=6)
        splits = splitter.split(examples)
        all_holdout = set(id(ex) for ex in splits.random + splits.game_level + splits.opening_family + splits.piece_count)
        train_ids = set(id(ex) for ex in splits.train)
        assert all_holdout.isdisjoint(train_ids)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data/test_holdout_splitter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'denoisr.data.holdout_splitter'`

**Step 3: Implement StratifiedHoldoutSplitter**

Create `src/denoisr/data/holdout_splitter.py`:

```python
"""Stratified holdout splitting for grokking detection.

Creates multiple independent holdout sets from TrainingExample lists:
- Random: baseline random split
- Game-level: entire games held out (no positional continuity leakage)
- Opening-family: entire ECO families held out (tests cross-structure generalization)
- Piece-count: endgame positions held out (tests phase-of-game generalization)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from denoisr.types import TrainingExample


@dataclass(frozen=True)
class HoldoutSplits:
    train: list[TrainingExample]
    random: list[TrainingExample]
    game_level: list[TrainingExample]
    opening_family: list[TrainingExample]
    piece_count: list[TrainingExample]


class StratifiedHoldoutSplitter:
    def __init__(
        self,
        holdout_frac: float = 0.05,
        endgame_threshold: int = 6,
    ) -> None:
        self._holdout_frac = holdout_frac
        self._endgame_threshold = endgame_threshold

    def split(self, examples: list[TrainingExample]) -> HoldoutSplits:
        n = len(examples)
        holdout_n = max(1, int(n * self._holdout_frac))
        excluded: set[int] = set()  # indices of examples in any holdout

        # 1. Piece-count holdout: all endgame positions
        piece_count_holdout: list[TrainingExample] = []
        for i, ex in enumerate(examples):
            if ex.piece_count is not None and ex.piece_count <= self._endgame_threshold:
                piece_count_holdout.append(ex)
                excluded.add(i)

        # 2. Game-level holdout: entire games
        game_level_holdout: list[TrainingExample] = []
        game_ids = {ex.game_id for ex in examples if ex.game_id is not None}
        if game_ids:
            sorted_ids = sorted(game_ids)
            holdout_game_count = max(1, int(len(sorted_ids) * self._holdout_frac))
            holdout_game_ids = set(random.sample(sorted_ids, holdout_game_count))
            for i, ex in enumerate(examples):
                if ex.game_id in holdout_game_ids and i not in excluded:
                    game_level_holdout.append(ex)
                    excluded.add(i)

        # 3. Opening-family holdout: entire ECO letter groups
        opening_holdout: list[TrainingExample] = []
        eco_families = {ex.eco_code[0] for ex in examples if ex.eco_code}
        if eco_families:
            holdout_family_count = max(1, int(len(eco_families) * self._holdout_frac))
            holdout_family_count = min(holdout_family_count, len(eco_families))
            holdout_families = set(random.sample(sorted(eco_families), holdout_family_count))
            for i, ex in enumerate(examples):
                if (
                    ex.eco_code
                    and ex.eco_code[0] in holdout_families
                    and i not in excluded
                ):
                    opening_holdout.append(ex)
                    excluded.add(i)

        # 4. Random holdout: from remaining examples
        remaining_indices = [i for i in range(n) if i not in excluded]
        random_holdout_n = min(holdout_n, len(remaining_indices))
        random_holdout_indices = set(
            random.sample(remaining_indices, random_holdout_n)
        )
        random_holdout = [examples[i] for i in random_holdout_indices]
        excluded.update(random_holdout_indices)

        # 5. Train: everything not in any holdout
        train = [examples[i] for i in range(n) if i not in excluded]

        return HoldoutSplits(
            train=train,
            random=random_holdout,
            game_level=game_level_holdout,
            opening_family=opening_holdout,
            piece_count=piece_count_holdout,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data/test_holdout_splitter.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/denoisr/data/holdout_splitter.py tests/test_data/test_holdout_splitter.py
git commit -m "feat: add StratifiedHoldoutSplitter for grokking detection holdouts"
```

---

### Task 3: Implement GrokfastFilter

**Files:**
- Create: `src/denoisr/training/grokfast.py`
- Test: `tests/test_training/test_grokfast.py`

**Step 1: Write failing tests**

Create `tests/test_training/test_grokfast.py`:

```python
import torch
from torch import nn

from denoisr.training.grokfast import GrokfastFilter


class TestGrokfastFilter:
    def test_first_apply_initializes_ema(self) -> None:
        model = nn.Linear(10, 5)
        model.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        gf.apply(model)
        # After first apply, EMA should be initialized (dict populated)
        assert len(gf.grads) > 0

    def test_amplifies_gradients(self) -> None:
        model = nn.Linear(10, 5, bias=False)
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        # First step: initialize EMA
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        gf.apply(model)
        model.zero_grad()
        # Second step: should amplify
        loss = model(x).sum()
        loss.backward()
        grad_before = model.weight.grad.clone()
        gf.apply(model)
        grad_after = model.weight.grad
        # Amplified gradient should have larger norm
        assert grad_after.norm() > grad_before.norm()

    def test_lamb_zero_means_no_amplification(self) -> None:
        model = nn.Linear(10, 5, bias=False)
        gf = GrokfastFilter(alpha=0.98, lamb=0.0)
        # Two steps to get past initialization
        for _ in range(2):
            model.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            grad_before = model.weight.grad.clone()
            gf.apply(model)
        # With lamb=0, gradient should be unchanged
        torch.testing.assert_close(model.weight.grad, grad_before)

    def test_skips_params_without_grad(self) -> None:
        model = nn.Linear(10, 5)
        model.weight.requires_grad_(False)
        model.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        gf.apply(model)  # Should not error
        assert "weight" not in str(gf.grads.keys())
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_grokfast.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement GrokfastFilter**

Create `src/denoisr/training/grokfast.py`:

```python
"""Grokfast: EMA gradient filtering for accelerated grokking.

Amplifies slow-varying gradient components (the generalizing circuit's signal)
while leaving fast-varying components (memorization) alone. Achieves ~50x
speedup of grokking in published results.

Reference: Lee et al. (2024) "Grokfast: Accelerated Grokking by Amplifying
Slow Gradients"

Usage:
    gf = GrokfastFilter(alpha=0.98, lamb=2.0)
    # In training loop, after loss.backward() and scaler.unscale_():
    gf.apply(model)
    # Then clip_grad_norm_() and optimizer.step() as normal
"""

from __future__ import annotations

from torch import Tensor, nn


class GrokfastFilter:
    """EMA-based gradient filter that amplifies slow-varying components."""

    def __init__(self, alpha: float = 0.98, lamb: float = 2.0) -> None:
        self.alpha = alpha
        self.lamb = lamb
        self.grads: dict[str, Tensor] = {}

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if name not in self.grads:
                self.grads[name] = param.grad.data.detach().clone()
            else:
                self.grads[name].mul_(self.alpha).add_(
                    param.grad.data.detach(), alpha=1 - self.alpha
                )
                param.grad.data.add_(self.grads[name], alpha=self.lamb)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_grokfast.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/grokfast.py tests/test_training/test_grokfast.py
git commit -m "feat: add GrokfastFilter for EMA gradient acceleration"
```

---

### Task 4: Extend TrainingConfig with grokking/Grokfast flags

**Files:**
- Modify: `src/denoisr/scripts/config.py:83-242,366-441,481-502`
- Test: `tests/test_scripts/test_config_grok.py`

**Step 1: Write failing tests**

Create `tests/test_scripts/test_config_grok.py`:

```python
import argparse

from denoisr.scripts.config import TrainingConfig, add_training_args, training_config_from_args


class TestGrokConfig:
    def test_default_grok_tracking_off(self) -> None:
        cfg = TrainingConfig()
        assert cfg.grok_tracking is False

    def test_default_grokfast_off(self) -> None:
        cfg = TrainingConfig()
        assert cfg.grokfast is False

    def test_grok_fields_exist(self) -> None:
        cfg = TrainingConfig(
            grok_tracking=True,
            grok_erank_freq=500,
            grok_spectral_freq=2000,
            grok_onset_threshold=0.93,
            grokfast=True,
            grokfast_alpha=0.95,
            grokfast_lamb=3.0,
        )
        assert cfg.grok_tracking is True
        assert cfg.grok_erank_freq == 500
        assert cfg.grok_spectral_freq == 2000
        assert cfg.grok_onset_threshold == 0.93
        assert cfg.grokfast is True
        assert cfg.grokfast_alpha == 0.95
        assert cfg.grokfast_lamb == 3.0

    def test_cli_flags_registered(self) -> None:
        parser = argparse.ArgumentParser()
        add_training_args(parser)
        args = parser.parse_args(["--grok-tracking", "--grokfast"])
        assert args.grok_tracking is True
        assert args.grokfast is True

    def test_training_config_from_args_includes_grok(self) -> None:
        parser = argparse.ArgumentParser()
        add_training_args(parser)
        args = parser.parse_args(["--grok-tracking", "--grok-erank-freq", "500"])
        cfg = training_config_from_args(args)
        assert cfg.grok_tracking is True
        assert cfg.grok_erank_freq == 500
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scripts/test_config_grok.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'grok_tracking'`

**Step 3: Add grokking fields to TrainingConfig**

In `src/denoisr/scripts/config.py`, add to `TrainingConfig` (after line 242, before the `detect_device` function):

```python
    # -- Grokking detection ---------------------------------------------------

    # Enable grokking detection metrics (weight norms, effective rank,
    # spectral norms, HTSR alpha, structured holdout evaluation, adaptive
    # frequency, and console alerts).
    grok_tracking: bool = False

    # Effective rank computation frequency (global steps).
    # Lower = more data points but higher compute cost.
    grok_erank_freq: int = 1000

    # Spectral norm and HTSR alpha computation frequency (global steps).
    grok_spectral_freq: int = 5000

    # Weight norm ratio threshold for onset detection.
    # Onset detected when mean(last 50 steps) < threshold * mean(steps[-100:-50]).
    grok_onset_threshold: float = 0.95

    # -- Grokfast acceleration ------------------------------------------------

    # Enable Grokfast EMA gradient filtering (Lee et al. 2024).
    # Amplifies slow-varying gradient components for ~50x grokking speedup.
    grokfast: bool = False

    # EMA decay rate for Grokfast gradient filter.
    # Higher = smoother (more historical averaging). 0.98 is the paper default.
    grokfast_alpha: float = 0.98

    # Grokfast amplification factor. Higher = stronger boost to slow gradients.
    # 2.0 is the paper default.
    grokfast_lamb: float = 2.0
```

Add CLI flags in `add_training_args` (after `--phase2-gate` at line 441):

```python
    # Grokking detection
    g.add_argument(
        "--grok-tracking",
        action=argparse.BooleanOptionalAction, default=False,
        help="enable grokking detection metrics (default: off)",
    )
    g.add_argument(
        "--grok-erank-freq", type=int, default=1000,
        help="effective rank computation frequency in steps (default: 1000)",
    )
    g.add_argument(
        "--grok-spectral-freq", type=int, default=5000,
        help="spectral norm / HTSR alpha frequency in steps (default: 5000)",
    )
    g.add_argument(
        "--grok-onset-threshold", type=float, default=0.95,
        help="weight norm ratio for onset detection (default: 0.95)",
    )
    # Grokfast
    g.add_argument(
        "--grokfast",
        action=argparse.BooleanOptionalAction, default=False,
        help="enable Grokfast EMA gradient filtering (default: off)",
    )
    g.add_argument(
        "--grokfast-alpha", type=float, default=0.98,
        help="Grokfast EMA decay rate (default: 0.98)",
    )
    g.add_argument(
        "--grokfast-lamb", type=float, default=2.0,
        help="Grokfast amplification factor (default: 2.0)",
    )
```

Update `training_config_from_args` (line 481) to include new fields:

```python
def training_config_from_args(args: Namespace) -> TrainingConfig:
    return TrainingConfig(
        # ... existing fields ...
        phase2_gate=args.phase2_gate,
        grok_tracking=args.grok_tracking,
        grok_erank_freq=args.grok_erank_freq,
        grok_spectral_freq=args.grok_spectral_freq,
        grok_onset_threshold=args.grok_onset_threshold,
        grokfast=args.grokfast,
        grokfast_alpha=args.grokfast_alpha,
        grokfast_lamb=args.grokfast_lamb,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scripts/test_config_grok.py -v`
Expected: all PASS

**Step 5: Run all tests to check for regressions**

Run: `uv run pytest tests/ -x -q`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/denoisr/scripts/config.py tests/test_scripts/test_config_grok.py
git commit -m "feat: add grokking detection and Grokfast config flags"
```

---

### Task 5: Implement GrokTracker — metric computation core

**Files:**
- Create: `src/denoisr/training/grok_tracker.py`
- Test: `tests/test_training/test_grok_tracker.py`

This is the largest task. Split into two sub-parts: metric computation (this task) and state machine/alerts (Task 6).

**Step 1: Write failing tests for metric computation**

Create `tests/test_training/test_grok_tracker.py`:

```python
import torch
from torch import nn

from conftest import SMALL_D_S, SMALL_FFN_DIM, SMALL_NUM_HEADS, SMALL_NUM_LAYERS

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.grok_tracker import GrokTracker


def _build_small_model(
    device: torch.device,
) -> tuple[ChessEncoder, ChessPolicyBackbone, ChessPolicyHead, ChessValueHead]:
    encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
    backbone = ChessPolicyBackbone(
        d_s=SMALL_D_S,
        num_heads=SMALL_NUM_HEADS,
        num_layers=SMALL_NUM_LAYERS,
        ffn_dim=SMALL_FFN_DIM,
    ).to(device)
    policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
    value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
    return encoder, backbone, policy_head, value_head


class TestGrokTrackerMetrics:
    def test_compute_weight_norms(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        norms = tracker.compute_weight_norms()
        assert "total" in norms
        assert "encoder" in norms
        assert "backbone" in norms
        assert "policy_head" in norms
        assert "value_head" in norms
        assert all(v > 0 for v in norms.values())

    def test_compute_effective_rank(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        # Run a forward pass to populate activation hooks
        x = torch.randn(2, 12, 8, 8, device=device)
        latent = encoder(x)
        _ = backbone(latent)
        eranks = tracker.compute_effective_rank()
        assert len(eranks) == SMALL_NUM_LAYERS
        for layer_idx, erank in eranks.items():
            assert erank > 0
            assert erank <= SMALL_D_S  # Can't exceed embedding dimension

    def test_compute_spectral_norms(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        spectral = tracker.compute_spectral_norms()
        assert len(spectral) > 0
        assert all(v > 0 for v in spectral.values())

    def test_compute_htsr_alpha(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        alphas = tracker.compute_htsr_alpha()
        assert len(alphas) > 0
        for layer_idx, alpha in alphas.items():
            assert alpha > 0  # Power-law exponent must be positive

    def test_hooks_capture_activations(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        assert len(tracker._activations) == 0
        x = torch.randn(2, 12, 8, 8, device=device)
        latent = encoder(x)
        _ = backbone(latent)
        assert len(tracker._activations) == SMALL_NUM_LAYERS

    def test_hooks_are_removed_on_close(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        tracker.close()
        tracker._activations.clear()
        x = torch.randn(2, 12, 8, 8, device=device)
        latent = encoder(x)
        _ = backbone(latent)
        assert len(tracker._activations) == 0  # Hooks removed, no captures
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_grok_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement GrokTracker metric computation**

Create `src/denoisr/training/grok_tracker.py`:

```python
"""Grokking detection tracker for Phase 1 supervised training.

Registers forward hooks on backbone transformer layers to capture activations.
Computes Tier 1 (weight norms, loss gap) and Tier 2 (effective rank, spectral
norms, HTSR alpha) metrics at configurable frequencies. Contains a 4-state
machine for adaptive evaluation frequency and console alerts.

References:
- Power et al. (2022) — grokking in modular arithmetic
- Nanda et al. (2023) — mechanistic interpretability of grokking (ICLR Oral)
- Zunkovic & Ilievski (2024) — effective dimensionality (JMLR)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from enum import IntEnum, auto

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHook

from denoisr.nn.policy_backbone import ChessPolicyBackbone, TransformerBlock

logger = logging.getLogger(__name__)


class GrokState(IntEnum):
    BASELINE = 0
    ONSET_DETECTED = auto()
    TRANSITIONING = auto()
    GROKKED = auto()


class GrokTracker:
    """Grokking detection and metric computation for training instrumentation."""

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        erank_freq: int = 1000,
        spectral_freq: int = 5000,
        onset_threshold: float = 0.95,
    ) -> None:
        self._encoder = encoder
        self._backbone = backbone
        self._policy_head = policy_head
        self._value_head = value_head
        self._erank_freq = erank_freq
        self._spectral_freq = spectral_freq
        self._onset_threshold = onset_threshold

        # State machine
        self._state = GrokState.BASELINE
        self._weight_norm_history: list[float] = []
        self._holdout_accuracy_history: list[float] = []
        self._erank_history: list[float] = []
        self._train_saturation_step: int | None = None

        # Adaptive frequency multiplier
        self._freq_multiplier = 1

        # Forward hooks on backbone layers
        self._activations: dict[int, Tensor] = {}
        self._hooks: list[RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        if not isinstance(self._backbone, ChessPolicyBackbone):
            return
        for i, layer in enumerate(self._backbone.layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook_fn(module: nn.Module, input: object, output: Tensor) -> None:
            self._activations[layer_idx] = output.detach()
        return hook_fn

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # -- Tier 1: Weight norms (every step) -----------------------------------

    def compute_weight_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}
        for name, module in [
            ("encoder", self._encoder),
            ("backbone", self._backbone),
            ("policy_head", self._policy_head),
            ("value_head", self._value_head),
        ]:
            sq_sum = sum(
                p.data.norm(2).item() ** 2
                for p in module.parameters()
            )
            norms[name] = sq_sum ** 0.5
        norms["total"] = sum(v ** 2 for v in norms.values()) ** 0.5
        return norms

    # -- Tier 2: Effective rank (every N steps) ------------------------------

    def compute_effective_rank(self) -> dict[int, float]:
        eranks: dict[int, float] = {}
        for layer_idx, act in self._activations.items():
            # act shape: [B, 64, d_s]
            flat = act.reshape(-1, act.shape[-1]).float()
            flat = flat - flat.mean(dim=0)
            svs = torch.linalg.svdvals(flat)
            # Normalize to probability distribution
            p = svs / svs.sum()
            p = p[p > 1e-12]
            # Shannon entropy → effective rank
            entropy = -(p * p.log()).sum()
            eranks[layer_idx] = math.exp(entropy.item())
        return eranks

    # -- Tier 2: Spectral norms (every N steps) -----------------------------

    def compute_spectral_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}
        if not isinstance(self._backbone, ChessPolicyBackbone):
            return norms
        for i, layer in enumerate(self._backbone.layers):
            if not isinstance(layer, TransformerBlock):
                continue
            # Attention QKV weight
            w_qkv = layer.qkv.weight.data.float()
            norms[f"layer_{i}/attn"] = torch.linalg.svdvals(w_qkv)[0].item()
            # FFN first linear weight
            w_ffn = layer.ffn[0].weight.data.float()
            norms[f"layer_{i}/ffn"] = torch.linalg.svdvals(w_ffn)[0].item()
        return norms

    # -- Tier 2: HTSR alpha (every N steps) ----------------------------------

    def compute_htsr_alpha(self) -> dict[int, float]:
        alphas: dict[int, float] = {}
        if not isinstance(self._backbone, ChessPolicyBackbone):
            return alphas
        for i, layer in enumerate(self._backbone.layers):
            if not isinstance(layer, TransformerBlock):
                continue
            w = layer.qkv.weight.data.float()
            eigenvalues = torch.linalg.svdvals(w) ** 2
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) < 10:
                continue
            # Fit power-law tail (top 50% of eigenvalues)
            n = len(eigenvalues)
            tail = eigenvalues[: n // 2]
            log_rank = torch.log(torch.arange(1, len(tail) + 1, dtype=torch.float32))
            log_vals = torch.log(tail.cpu())
            # Linear regression: log_vals = -alpha * log_rank + const
            x = log_rank
            y = log_vals
            x_mean = x.mean()
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            alphas[i] = -slope.item()
        return alphas

    # -- Step and epoch hooks ------------------------------------------------

    def step(
        self,
        global_step: int,
        loss_breakdown: dict[str, float],
        grad_norm: float,
    ) -> dict[str, float]:
        """Called after every training step. Returns metrics dict for logging."""
        metrics: dict[str, float] = {}

        # Tier 1: weight norms (every step)
        norms = self.compute_weight_norms()
        metrics["grok/weight_norm_total"] = norms["total"]
        for name, val in norms.items():
            if name != "total":
                metrics[f"grok/weight_norm/{name}"] = val
        self._weight_norm_history.append(norms["total"])

        # Tier 2: effective rank (every N steps, adaptive)
        erank_freq = max(1, self._erank_freq // self._freq_multiplier)
        if global_step % erank_freq == 0 and self._activations:
            eranks = self.compute_effective_rank()
            for layer_idx, erank in eranks.items():
                metrics[f"grok/erank/layer_{layer_idx}"] = erank
            if eranks:
                mean_erank = sum(eranks.values()) / len(eranks)
                self._erank_history.append(mean_erank)

        # Tier 2: spectral norms + HTSR alpha (every N steps, adaptive)
        spectral_freq = max(1, self._spectral_freq // self._freq_multiplier)
        if global_step % spectral_freq == 0:
            spectral = self.compute_spectral_norms()
            for key, val in spectral.items():
                metrics[f"grok/spectral_norm/{key}"] = val
            alphas = self.compute_htsr_alpha()
            for layer_idx, alpha in alphas.items():
                metrics[f"grok/alpha/layer_{layer_idx}"] = alpha

        # State machine transition check
        self._check_step_transitions(global_step)
        metrics["grok/state"] = float(self._state)

        return metrics

    def epoch(
        self,
        epoch: int,
        train_loss: float,
        holdout_metrics: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """Called after every epoch.

        holdout_metrics: mapping of split_name -> (accuracy, loss)
        """
        metrics: dict[str, float] = {}
        for split_name, (acc, loss) in holdout_metrics.items():
            metrics[f"grok/holdout/{split_name}/accuracy"] = acc
            metrics[f"grok/holdout/{split_name}/loss"] = loss

        # Compute loss gap (train - best holdout)
        if holdout_metrics:
            best_holdout_loss = min(loss for _, loss in holdout_metrics.values())
            metrics["grok/loss_gap"] = train_loss - best_holdout_loss

        # Track accuracy for state machine
        if "random" in holdout_metrics:
            self._holdout_accuracy_history.append(holdout_metrics["random"][0])

        self._check_epoch_transitions(epoch, holdout_metrics)
        metrics["grok/state"] = float(self._state)

        return metrics

    # -- State machine -------------------------------------------------------

    def _check_step_transitions(self, global_step: int) -> None:
        if self._state >= GrokState.ONSET_DETECTED:
            return
        # Check weight norm sustained decrease
        history = self._weight_norm_history
        if len(history) >= 100:
            recent = sum(history[-50:]) / 50
            earlier = sum(history[-100:-50]) / 50
            if earlier > 0 and recent < self._onset_threshold * earlier:
                self._transition_to(GrokState.ONSET_DETECTED, global_step,
                    f"weight_norm decreased {(1 - recent/earlier)*100:.1f}% (50-step window)")

        # Check effective rank drop
        if len(self._erank_history) >= 10:
            recent_er = sum(self._erank_history[-5:]) / 5
            earlier_er = sum(self._erank_history[-10:-5]) / 5
            if earlier_er > 0 and recent_er < 0.9 * earlier_er:
                self._transition_to(GrokState.ONSET_DETECTED, global_step,
                    f"effective_rank decreased {(1 - recent_er/earlier_er)*100:.1f}%")

    def _check_epoch_transitions(
        self, epoch: int, holdout_metrics: dict[str, tuple[float, float]]
    ) -> None:
        acc_history = self._holdout_accuracy_history

        # ONSET -> TRANSITIONING
        if self._state == GrokState.ONSET_DETECTED and len(acc_history) >= 20:
            recent = sum(acc_history[-10:]) / 10
            earlier = sum(acc_history[-20:-10]) / 10
            if recent - earlier > 0.05:  # >5pp improvement
                self._transition_to(GrokState.TRANSITIONING, epoch,
                    f"holdout accuracy improved {(recent-earlier)*100:.1f}pp over 20 epochs")

        # TRANSITIONING -> GROKKED
        if self._state == GrokState.TRANSITIONING and acc_history:
            if acc_history[-1] > 0.25:
                grok_gap = epoch - (self._train_saturation_step or 0)
                self._transition_to(GrokState.GROKKED, epoch,
                    f"holdout accuracy {acc_history[-1]*100:.1f}% > 25% threshold, gap={grok_gap} epochs")

        # BASELINE -> ONSET via holdout accuracy jump
        if self._state == GrokState.BASELINE and len(acc_history) >= 10:
            recent = sum(acc_history[-5:]) / 5
            earlier = sum(acc_history[-10:-5]) / 5
            if recent - earlier > 0.02:  # >2pp improvement
                self._transition_to(GrokState.ONSET_DETECTED, epoch,
                    f"holdout accuracy improved {(recent-earlier)*100:.1f}pp over 10 epochs")

    def _transition_to(
        self, new_state: GrokState, step_or_epoch: int, trigger: str
    ) -> None:
        old_state = self._state
        self._state = new_state

        # Adaptive frequency
        if new_state == GrokState.ONSET_DETECTED:
            self._freq_multiplier = 5
        elif new_state == GrokState.TRANSITIONING:
            self._freq_multiplier = 10

        logger.warning(
            "GROKKING %s (step/epoch %d)",
            new_state.name, step_or_epoch,
        )
        logger.warning("  Trigger: %s", trigger)
        if self._freq_multiplier > 1:
            logger.warning(
                "  Action: eval frequency %dx (erank: %d steps, spectral: %d steps)",
                self._freq_multiplier,
                max(1, self._erank_freq // self._freq_multiplier),
                max(1, self._spectral_freq // self._freq_multiplier),
            )

    @property
    def state(self) -> GrokState:
        return self._state
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_grok_tracker.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/grok_tracker.py tests/test_training/test_grok_tracker.py
git commit -m "feat: add GrokTracker with metric computation and activation hooks"
```

---

### Task 6: GrokTracker state machine and alerting tests

**Files:**
- Modify: `tests/test_training/test_grok_tracker.py` (add state machine tests)

**Step 1: Write state machine tests**

Append to `tests/test_training/test_grok_tracker.py`:

```python
class TestGrokTrackerStateMachine:
    def test_starts_in_baseline(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
        )
        assert tracker.state == GrokState.BASELINE

    def test_onset_detected_on_weight_norm_decrease(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
            onset_threshold=0.95,
        )
        # Simulate weight norm history: 100 values, first 50 at 10.0, last 50 at 9.0
        tracker._weight_norm_history = [10.0] * 50 + [9.0] * 50
        tracker._check_step_transitions(global_step=100)
        assert tracker.state == GrokState.ONSET_DETECTED

    def test_no_onset_if_norm_stable(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
        )
        tracker._weight_norm_history = [10.0] * 100
        tracker._check_step_transitions(global_step=100)
        assert tracker.state == GrokState.BASELINE

    def test_onset_increases_frequency(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
        )
        assert tracker._freq_multiplier == 1
        tracker._transition_to(GrokState.ONSET_DETECTED, 100, "test")
        assert tracker._freq_multiplier == 5

    def test_transitioning_increases_frequency_further(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
        )
        tracker._transition_to(GrokState.ONSET_DETECTED, 100, "test")
        tracker._transition_to(GrokState.TRANSITIONING, 200, "test")
        assert tracker._freq_multiplier == 10

    def test_step_returns_grok_state(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
        )
        # Run forward pass so hooks fire
        x = torch.randn(2, 12, 8, 8, device=device)
        latent = encoder(x)
        _ = backbone(latent)
        metrics = tracker.step(0, {"policy": 1.0, "value": 0.5, "total": 1.5}, 0.5)
        assert "grok/state" in metrics
        assert metrics["grok/state"] == 0.0  # BASELINE

    def test_epoch_computes_loss_gap(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
        )
        metrics = tracker.epoch(
            epoch=0,
            train_loss=5.0,
            holdout_metrics={"random": (0.01, 6.0), "game_level": (0.02, 5.5)},
        )
        assert "grok/loss_gap" in metrics
        assert metrics["grok/loss_gap"] == pytest.approx(5.0 - 5.5)  # train - min(holdout losses)
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_training/test_grok_tracker.py -v`
Expected: all PASS

**Step 3: Commit**

```bash
git add tests/test_training/test_grok_tracker.py
git commit -m "test: add GrokTracker state machine and alerting tests"
```

---

### Task 7: Extend TrainingLogger for grokking metrics

**Files:**
- Modify: `src/denoisr/training/logger.py:21-133`
- Modify: `tests/test_training/test_logger.py`

**Step 1: Write failing test**

Append to `tests/test_training/test_logger.py`:

```python
class TestGrokLogging:
    def test_log_grok_metrics_writes_scalars(self, tmp_path: pathlib.Path) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        metrics = {
            "grok/weight_norm_total": 42.0,
            "grok/erank/layer_0": 15.3,
            "grok/state": 0.0,
        }
        logger.log_grok_metrics(step=100, metrics=metrics)
        logger.close()
        event_files = list((tmp_path / "test").glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_log_grok_state_transition_writes_text(self, tmp_path: pathlib.Path) -> None:
        logger = TrainingLogger(log_dir=tmp_path, run_name="test")
        logger.log_grok_state_transition(
            step=5000, old_state="BASELINE", new_state="ONSET_DETECTED",
            trigger="weight_norm decreased 6.2%",
        )
        logger.close()
        text = (tmp_path / "test" / "metrics.log").read_text()
        assert "GROKKING" in text
        assert "ONSET_DETECTED" in text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_logger.py::TestGrokLogging -v`
Expected: FAIL — `AttributeError: 'TrainingLogger' object has no attribute 'log_grok_metrics'`

**Step 3: Add grokking logging methods to TrainingLogger**

In `src/denoisr/training/logger.py`, add before `_write_text` (line 117):

```python
    def log_grok_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log grokking detection metrics to TensorBoard."""
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def log_grok_state_transition(
        self,
        step: int,
        old_state: str,
        new_state: str,
        trigger: str,
    ) -> None:
        """Log grokking state transition to text log."""
        self._write_text(
            f"GROKKING step={step}\t{old_state}->{new_state}\ttrigger={trigger}"
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_logger.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/logger.py tests/test_training/test_logger.py
git commit -m "feat: add grokking metrics logging to TrainingLogger"
```

---

### Task 8: Integrate GrokTracker into SupervisedTrainer

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py:12-161`
- Modify: `tests/test_training/test_supervised_trainer.py`

**Step 1: Write failing test for Grokfast integration**

Append to `tests/test_training/test_supervised_trainer.py`:

```python
from denoisr.training.grokfast import GrokfastFilter


class TestSupervisedTrainerGrokfast:
    @pytest.fixture
    def trainer_with_grokfast(self, device: torch.device) -> SupervisedTrainer:
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
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        return SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
            grokfast_filter=gf,
        )

    def test_train_step_with_grokfast(
        self, trainer_with_grokfast: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        loss, breakdown = trainer_with_grokfast.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_grokfast_ema_populated_after_step(
        self, trainer_with_grokfast: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        trainer_with_grokfast.train_step(batch)
        assert trainer_with_grokfast._grokfast_filter is not None
        assert len(trainer_with_grokfast._grokfast_filter.grads) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py::TestSupervisedTrainerGrokfast -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'grokfast_filter'`

**Step 3: Modify SupervisedTrainer to accept GrokfastFilter**

In `src/denoisr/training/supervised_trainer.py`:

Add import at top:

```python
from denoisr.training.grokfast import GrokfastFilter
```

Add parameter to `__init__` (after `min_lr` param at line 33):

```python
        grokfast_filter: GrokfastFilter | None = None,
```

Store it:

```python
        self._grokfast_filter = grokfast_filter
```

In `_forward_backward`, insert Grokfast application between `unscale_` and `clip_grad_norm_` (after line 86):

```python
        self.scaler.unscale_(self.optimizer)
        if self._grokfast_filter is not None:
            for module in [self.encoder, self.backbone, self.policy_head, self.value_head]:
                self._grokfast_filter.apply(module)
        total_norm = torch.nn.utils.clip_grad_norm_(
```

Note: `scaler.unscale_` is already called at line 86. The Grokfast filter goes between unscale and clip.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/supervised_trainer.py tests/test_training/test_supervised_trainer.py
git commit -m "feat: integrate GrokfastFilter into SupervisedTrainer"
```

---

### Task 9: Integrate everything into train_phase1.py

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py`
- Test: manual — run `uv run denoisr-train-phase1 --help` and verify new flags appear

**Step 1: Update train_phase1.py**

In `src/denoisr/scripts/train_phase1.py`, add imports:

```python
from denoisr.data.holdout_splitter import StratifiedHoldoutSplitter
from denoisr.training.grok_tracker import GrokTracker
from denoisr.training.grokfast import GrokfastFilter
```

In the `main()` function, after constructing `trainer` (line 161) and before the `with TrainingLogger` block (line 163):

1. Create structured holdout splits (replacing simple holdout):

```python
    # --- Holdout splits ---
    if tcfg.grok_tracking:
        splitter = StratifiedHoldoutSplitter(
            holdout_frac=args.holdout_frac,
            endgame_threshold=6,
        )
        splits = splitter.split(all_examples)
        train = splits.train
        holdout_sets = {
            "random": splits.random,
            "game_level": splits.game_level,
            "opening_family": splits.opening_family,
            "piece_count": splits.piece_count,
        }
        # Remove empty splits
        holdout_sets = {k: v for k, v in holdout_sets.items() if v}
        holdout = splits.random  # Primary holdout for phase gate
    else:
        holdout_n = max(1, int(len(all_examples) * args.holdout_frac))
        holdout = all_examples[:holdout_n]
        train = all_examples[holdout_n:]
        holdout_sets = {"random": holdout}
```

2. Create GrokTracker and GrokfastFilter:

```python
    # --- Grokking detection ---
    grok_tracker: GrokTracker | None = None
    if tcfg.grok_tracking:
        grok_tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            erank_freq=tcfg.grok_erank_freq,
            spectral_freq=tcfg.grok_spectral_freq,
            onset_threshold=tcfg.grok_onset_threshold,
        )

    grokfast_filter: GrokfastFilter | None = None
    if tcfg.grokfast:
        grokfast_filter = GrokfastFilter(
            alpha=tcfg.grokfast_alpha,
            lamb=tcfg.grokfast_lamb,
        )
```

3. Pass `grokfast_filter` to trainer constructor:

```python
    trainer = SupervisedTrainer(
        ...,
        grokfast_filter=grokfast_filter,
    )
```

4. In the training loop, after `logger.log_train_step()` (line 235), add:

```python
                if grok_tracker is not None:
                    grok_metrics = grok_tracker.step(global_step, breakdown, breakdown.get("grad_norm", 0.0))
                    logger.log_grok_metrics(global_step, grok_metrics)
```

5. At epoch end, evaluate all holdout sets and feed to tracker (after line 252):

```python
            # Evaluate all holdout sets for grokking detection
            if grok_tracker is not None:
                holdout_results: dict[str, tuple[float, float]] = {}
                for split_name, split_examples in holdout_sets.items():
                    if split_examples:
                        split_top1, _ = measure_accuracy(trainer, split_examples, device)
                        holdout_results[split_name] = (split_top1, avg_loss)
                grok_epoch_metrics = grok_tracker.epoch(epoch, avg_loss, holdout_results)
                logger.log_grok_metrics(epoch, grok_epoch_metrics)
```

6. Close tracker at end of training:

```python
    if grok_tracker is not None:
        grok_tracker.close()
```

**Step 2: Verify --help shows new flags**

Run: `uv run denoisr-train-phase1 --help`
Expected: Shows `--grok-tracking`, `--grokfast`, etc.

**Step 3: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: all PASS

**Step 4: Commit**

```bash
git add src/denoisr/scripts/train_phase1.py
git commit -m "feat: integrate grokking detection into Phase 1 training loop"
```

---

### Task 10: Linting, type checking, and final verification

**Files:** All modified files

**Step 1: Run ruff**

Run: `uvx ruff check src/denoisr/training/grok_tracker.py src/denoisr/training/grokfast.py src/denoisr/data/holdout_splitter.py`
Expected: no errors. Fix any issues found.

**Step 2: Run mypy**

Run: `uv run --with mypy mypy --strict src/denoisr/training/grok_tracker.py src/denoisr/training/grokfast.py src/denoisr/data/holdout_splitter.py`
Expected: no errors. Fix any issues found.

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: all PASS

**Step 4: Commit any fixups**

```bash
git add -A && git commit -m "fix: lint and type check fixes for grokking detection"
```

---

### Task 11: Update README with grokking detection section

**Files:**
- Modify: `README.md`

**Step 1: Add grokking detection section**

After the "Training logs" section in README.md, add a section documenting:
- `--grok-tracking` flag and what it enables
- `--grokfast` flag and what it does
- New TensorBoard panels (grok/* metrics)
- Console alert format
- Example command:

```bash
uv run denoisr-train-phase1 \
    --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt \
    --grok-tracking \
    --run-name grok-experiment
```

And the Grokfast acceleration example:

```bash
uv run denoisr-train-phase1 \
    --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt \
    --grok-tracking \
    --grokfast \
    --grokfast-alpha 0.98 \
    --grokfast-lamb 2.0 \
    --run-name grokfast-experiment
```

**Step 2: Add grokking flags to hyperparameter tables**

Add to the "Training optimization" table:

| `--grok-tracking` | `false` | Enable grokking detection metrics, structured holdouts, and adaptive alerts |
| `--grokfast` | `false` | Enable Grokfast EMA gradient filtering (~50x grokking acceleration) |

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add grokking detection documentation to README"
```
