# Phase 1 Pipeline Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the accuracy evaluation bug, sharpen policy targets, slow LR decay, and add per-batch accuracy logging to pass the Phase 1 gate (30% top-1).

**Architecture:** No architectural changes. Four surgical fixes to the training pipeline: (1) add legal-move masking to accuracy evaluation, (2) add illegal-move logit penalty to loss, (3) make Stockfish policy temperature configurable with label smoothing, (4) slow the cosine LR schedule. Per-batch masked accuracy provides visibility.

**Tech Stack:** Python 3.12, PyTorch, uv, pytest. Run commands via `uv run`. Lint with `uvx ruff check`. Type-check with `uv run --with mypy mypy --strict`.

---

### Task 1: Fix `measure_accuracy()` — Legal Move Masking

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py:48-83` (the `measure_accuracy` function)
- Test: `tests/test_scripts/test_measure_accuracy.py` (new)

**Step 1: Write the failing test**

Create `tests/test_scripts/__init__.py` if it doesn't exist, then create the test file:

```python
# tests/test_scripts/test_measure_accuracy.py
import torch
import pytest

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

from conftest import SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM


def _make_trainer(device: torch.device) -> SupervisedTrainer:
    encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
    backbone = ChessPolicyBackbone(
        d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
        num_layers=SMALL_NUM_LAYERS, ffn_dim=SMALL_FFN_DIM,
    ).to(device)
    policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
    value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
    loss_fn = ChessLossComputer()
    return SupervisedTrainer(
        encoder=encoder, backbone=backbone,
        policy_head=policy_head, value_head=value_head,
        loss_fn=loss_fn, lr=1e-3, device=device,
    )


class TestMeasureAccuracy:
    def test_accuracy_masks_illegal_moves(self, device: torch.device) -> None:
        """Model with high logits at illegal positions should NOT count as correct."""
        from denoisr.scripts.train_phase1 import measure_accuracy

        trainer = _make_trainer(device)
        # Create example where target has ONE legal move at (12, 28)
        board = BoardTensor(torch.randn(12, 8, 8))
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0  # only legal move
        policy = PolicyTarget(policy_data)
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        examples = [TrainingExample(board=board, policy=policy, value=value)]

        # Accuracy should be between 0 and 1 (not crash, not measure unmasked)
        top1, top5 = measure_accuracy(trainer, examples, device)
        assert 0.0 <= top1 <= 1.0
        assert 0.0 <= top5 <= 1.0

    def test_perfect_model_gets_100_percent(self, device: torch.device) -> None:
        """A model trained to predict the exact target should reach high accuracy."""
        from denoisr.scripts.train_phase1 import measure_accuracy

        trainer = _make_trainer(device)

        board_data = torch.randn(12, 8, 8)
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0
        examples = [
            TrainingExample(
                board=BoardTensor(board_data),
                policy=PolicyTarget(policy_data),
                value=ValueTarget(win=1.0, draw=0.0, loss=0.0),
            )
        ]

        # Overtrain on a single example
        for _ in range(200):
            trainer.train_step(examples)

        top1, top5 = measure_accuracy(trainer, examples, device)
        # After overtraining, should get this right with masking
        assert top1 > 0.5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scripts/test_measure_accuracy.py -v -x`

Expected: The `test_perfect_model_gets_100_percent` test will likely fail because the current `measure_accuracy` picks from 4096 unmasked entries.

**Step 3: Fix `measure_accuracy` in `train_phase1.py`**

Replace lines 75-81 in `measure_accuracy()`:

```python
# BEFORE (broken):
pred_flat = logits.reshape(len(batch), -1)
target_flat = targets.reshape(len(batch), -1)
target_idx = target_flat.argmax(dim=-1)
top5 = pred_flat.topk(5, dim=-1).indices
correct_1 += (top5[:, 0] == target_idx).sum().item()
correct_5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()

# AFTER (fixed):
pred_flat = logits.reshape(len(batch), -1)
target_flat = targets.reshape(len(batch), -1)
legal_mask = target_flat > 0
masked_logits = pred_flat.masked_fill(~legal_mask, float("-inf"))
target_idx = target_flat.argmax(dim=-1)
top5 = masked_logits.topk(5, dim=-1).indices
correct_1 += (top5[:, 0] == target_idx).sum().item()
correct_5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scripts/test_measure_accuracy.py -v -x`

Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q -k "not stockfish"`

Expected: All tests pass.

**Step 6: Commit**

```bash
git add tests/test_scripts/test_measure_accuracy.py src/denoisr/scripts/train_phase1.py
git commit -m "fix: mask illegal moves in measure_accuracy evaluation"
```

---

### Task 2: Add Illegal-Move Logit Penalty to Loss

**Files:**
- Modify: `src/denoisr/training/loss.py:24-35` (add `illegal_penalty_weight` param), `loss.py:50-94` (add penalty in `compute()`)
- Test: `tests/test_training/test_loss.py` (add new tests)

**Step 1: Write the failing test**

Add to `tests/test_training/test_loss.py`:

```python
def test_illegal_penalty_increases_total_loss(self) -> None:
    """With illegal_penalty_weight > 0, total loss should increase when
    illegal logits are large."""
    loss_fn_no_penalty = ChessLossComputer(illegal_penalty_weight=0.0)
    loss_fn_with_penalty = ChessLossComputer(illegal_penalty_weight=0.01)

    pred_policy = torch.randn(2, 64, 64)
    # Make illegal positions have large logits
    pred_policy[:, 0, 0] = 100.0  # illegal position (target=0 there)
    pred_value = torch.randn(2, 3)
    target_policy = torch.zeros(2, 64, 64)
    target_policy[:, 12, 28] = 1.0  # only one legal move
    target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

    total_no, _ = loss_fn_no_penalty.compute(
        pred_policy, pred_value, target_policy, target_value
    )
    total_with, breakdown = loss_fn_with_penalty.compute(
        pred_policy, pred_value, target_policy, target_value
    )
    assert total_with.item() > total_no.item()
    assert "illegal_penalty" in breakdown

def test_illegal_penalty_gradient_flows(self) -> None:
    """Illegal penalty should produce gradients at illegal positions."""
    loss_fn = ChessLossComputer(illegal_penalty_weight=0.01)
    pred_policy = torch.randn(2, 64, 64, requires_grad=True)
    pred_value = torch.randn(2, 3)
    target_policy = torch.zeros(2, 64, 64)
    target_policy[:, 12, 28] = 1.0
    target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

    total, _ = loss_fn.compute(
        pred_policy, pred_value, target_policy, target_value
    )
    total.backward()
    # Gradient should be nonzero at illegal positions
    assert pred_policy.grad is not None
    assert pred_policy.grad[0, 0, 0].abs().item() > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_loss.py::TestChessLossComputer::test_illegal_penalty_increases_total_loss -v`

Expected: FAIL — `illegal_penalty_weight` parameter doesn't exist yet.

**Step 3: Implement the illegal-move penalty**

In `src/denoisr/training/loss.py`, add `illegal_penalty_weight` to `__init__` and compute it in `compute()`:

```python
# In __init__, add parameter:
def __init__(
    self,
    # ... existing params ...
    illegal_penalty_weight: float = 0.0,
) -> None:
    # ... existing code ...
    self._illegal_penalty_weight = illegal_penalty_weight

# In compute(), after policy_loss calculation (after line 70):
if self._illegal_penalty_weight > 0:
    illegal_logits = pred_flat.masked_fill(legal_mask, 0.0)
    illegal_penalty = (illegal_logits ** 2).mean()
    losses["illegal_penalty"] = illegal_penalty
```

Also add `"illegal_penalty"` to the base weights dict with the configured weight.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_loss.py -v -x`

Expected: All loss tests pass including new ones.

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q -k "not stockfish"`

**Step 6: Commit**

```bash
git add src/denoisr/training/loss.py tests/test_training/test_loss.py
git commit -m "feat: add illegal-move logit penalty to loss function"
```

---

### Task 3: Configurable Policy Temperature + Label Smoothing in StockfishOracle

**Files:**
- Modify: `src/denoisr/data/stockfish_oracle.py:9-11,20-48` (add temperature/smoothing params)
- Modify: `src/denoisr/scripts/generate_data.py:48-51,245-292` (add CLI flags, pass to oracle)
- Test: `tests/test_data/test_stockfish_oracle.py` (add new tests)

**Step 1: Write the failing test**

Add to `tests/test_data/test_stockfish_oracle.py`:

```python
def test_temperature_changes_distribution_sharpness(
    self, oracle: StockfishOracle
) -> None:
    """Higher temperature should produce sharper distributions."""
    board = chess.Board()

    # Create oracle with default temp (will be 150 after this task)
    assert STOCKFISH_PATH is not None
    sharp_oracle = StockfishOracle(
        path=STOCKFISH_PATH, depth=10, policy_temperature=150.0
    )
    soft_oracle = StockfishOracle(
        path=STOCKFISH_PATH, depth=10, policy_temperature=30.0
    )

    sharp_policy, _, _ = sharp_oracle.evaluate(board)
    soft_policy, _, _ = soft_oracle.evaluate(board)

    # Sharper distribution should have higher max probability
    assert sharp_policy.data.max().item() > soft_policy.data.max().item()

    sharp_oracle.close()
    soft_oracle.close()

def test_label_smoothing_redistributes_mass(
    self, oracle: StockfishOracle
) -> None:
    """With label smoothing, minimum probability on legal moves should be > 0."""
    board = chess.Board()
    assert STOCKFISH_PATH is not None
    smoothed = StockfishOracle(
        path=STOCKFISH_PATH, depth=10,
        policy_temperature=150.0, label_smoothing=0.1,
    )
    policy, _, _ = smoothed.evaluate(board)
    # All legal moves should have nonzero probability
    legal_probs = policy.data[policy.data > 0]
    assert legal_probs.min().item() > 0.001  # smoothing ensures minimum
    smoothed.close()
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_data/test_stockfish_oracle.py::TestStockfishOracle::test_temperature_changes_distribution_sharpness -v`

Expected: FAIL — `policy_temperature` parameter doesn't exist.

**Step 3: Implement configurable temperature and label smoothing**

In `src/denoisr/data/stockfish_oracle.py`:

```python
class StockfishOracle:
    def __init__(
        self,
        path: str,
        depth: int = 12,
        policy_temperature: float = 150.0,
        label_smoothing: float = 0.1,
    ) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._depth = depth
        self._policy_temperature = policy_temperature
        self._label_smoothing = label_smoothing
```

And in `_get_policy()`, replace the softmax line:

```python
# Replace: probs = torch.softmax(t / 30.0, dim=0)
probs = torch.softmax(t / self._policy_temperature, dim=0)
if self._label_smoothing > 0:
    n_legal = len(legal_moves)
    probs = (1 - self._label_smoothing) * probs + self._label_smoothing / n_legal
```

In `src/denoisr/scripts/generate_data.py`, add CLI flags and pass them to the worker initializer. Add to `_init_worker`:

```python
def _init_worker(
    stockfish_path: str,
    stockfish_depth: int,
    policy_temperature: float,
    label_smoothing: float,
) -> None:
    global _oracle, _encoder
    _oracle = StockfishOracle(
        path=stockfish_path,
        depth=stockfish_depth,
        policy_temperature=policy_temperature,
        label_smoothing=label_smoothing,
    )
    _encoder = ExtendedBoardEncoder()
    atexit.register(_cleanup_oracle)
```

Update `generate_examples` signature:

```python
def generate_examples(
    pgn_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
    num_workers: int,
    policy_temperature: float = 150.0,
    label_smoothing: float = 0.1,
) -> list[TrainingExample]:
```

And pass the new params to `Pool(initargs=(...))`.

Add CLI flags in `main()`:

```python
parser.add_argument("--policy-temperature", type=float, default=150.0,
    help="Softmax temperature for policy targets (default: 150.0)")
parser.add_argument("--label-smoothing", type=float, default=0.1,
    help="Label smoothing epsilon (default: 0.1)")
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_data/test_stockfish_oracle.py -v -x`

Expected: All oracle tests pass.

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q -k "not stockfish"`

Expected: Non-stockfish tests pass (generate_data tests may need the flag).

**Step 6: Commit**

```bash
git add src/denoisr/data/stockfish_oracle.py src/denoisr/scripts/generate_data.py tests/test_data/test_stockfish_oracle.py
git commit -m "feat: configurable policy temperature and label smoothing for Stockfish targets"
```

---

### Task 4: Slow LR Decay (T_max * 2)

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py:65-66` (change T_max calculation)
- Test: `tests/test_training/test_supervised_trainer.py` (add new test)

**Step 1: Write the failing test**

Add to `tests/test_training/test_supervised_trainer.py`:

```python
def test_lr_stays_above_half_peak_at_midpoint(
    self, device: torch.device
) -> None:
    """LR should still be above 50% of peak at the training midpoint."""
    total_epochs = 100
    warmup = 5
    encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
    backbone = ChessPolicyBackbone(
        d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
        num_layers=SMALL_NUM_LAYERS, ffn_dim=SMALL_FFN_DIM,
    ).to(device)
    policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
    value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
    loss_fn = ChessLossComputer()
    trainer = SupervisedTrainer(
        encoder=encoder, backbone=backbone,
        policy_head=policy_head, value_head=value_head,
        loss_fn=loss_fn, lr=3e-4, device=device,
        total_epochs=total_epochs, warmup_epochs=warmup,
    )

    # Advance to epoch 50 (midpoint)
    for _ in range(50):
        trainer.scheduler_step()

    head_lr = trainer.optimizer.param_groups[2]["lr"]
    peak_lr = 3e-4
    # With T_max*2, LR at midpoint should be > 50% of peak
    assert head_lr > peak_lr * 0.5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py::TestSupervisedTrainer::test_lr_stays_above_half_peak_at_midpoint -v`

Expected: FAIL — with current `T_max = 95`, LR at epoch 50 is ~49.6% of peak (just below 50%).

**Step 3: Implement the fix**

In `src/denoisr/training/supervised_trainer.py`, line 65-66, change:

```python
# BEFORE:
self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
)

# AFTER:
self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer, T_max=max(1, (total_epochs - warmup_epochs) * 2), eta_min=min_lr
)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v -x`

Expected: PASS — all trainer tests including the new one.

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q -k "not stockfish"`

**Step 6: Commit**

```bash
git add src/denoisr/training/supervised_trainer.py tests/test_training/test_supervised_trainer.py
git commit -m "fix: slow LR decay by doubling cosine T_max"
```

---

### Task 5: Per-Batch Masked Accuracy Logging

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py:71-113` (add batch_top1 to `_forward_backward`)
- Modify: `src/denoisr/scripts/train_phase1.py:312` (log batch_top1 to TensorBoard)
- Test: `tests/test_training/test_supervised_trainer.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_training/test_supervised_trainer.py`:

```python
def test_breakdown_includes_batch_top1(
    self, trainer: SupervisedTrainer
) -> None:
    batch = _make_batch(4)
    _, breakdown = trainer.train_step(batch)
    assert "batch_top1" in breakdown
    assert 0.0 <= breakdown["batch_top1"] <= 1.0
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py::TestSupervisedTrainer::test_breakdown_includes_batch_top1 -v`

Expected: FAIL — `batch_top1` not in breakdown.

**Step 3: Add per-batch accuracy computation**

In `src/denoisr/training/supervised_trainer.py`, inside `_forward_backward()`, after the `with autocast(...)` block and before `self.optimizer.zero_grad()`, add:

```python
with torch.no_grad():
    B = boards.shape[0]
    pf = pred_policy.detach().reshape(B, -1)
    tf = target_policies.reshape(B, -1)
    mask = tf > 0
    masked = pf.masked_fill(~mask, float("-inf"))
    batch_top1 = (masked.argmax(-1) == tf.argmax(-1)).float().mean().item()
```

Then add `breakdown["batch_top1"] = batch_top1` after the breakdown is populated.

In `train_phase1.py`, inside the batch loop, the existing `logger.log_train_step(global_step, loss, breakdown)` already logs everything in `breakdown` to TensorBoard. The `batch_top1` key will be logged as `loss/batch_top1` automatically. To put it in the `accuracy/` namespace instead, add after the log_train_step call:

```python
if "batch_top1" in breakdown:
    logger._writer.add_scalar("accuracy/batch_top1", breakdown["batch_top1"], global_step)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v -x`

Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q -k "not stockfish"`

**Step 6: Commit**

```bash
git add src/denoisr/training/supervised_trainer.py src/denoisr/scripts/train_phase1.py tests/test_training/test_supervised_trainer.py
git commit -m "feat: add per-batch masked accuracy logging"
```

---

### Task 6: Wire Up Illegal Penalty + Update Defaults

**Files:**
- Modify: `src/denoisr/scripts/config.py` (add `illegal_penalty_weight` to TrainingConfig)
- Modify: `src/denoisr/scripts/train_phase1.py:177-182` (pass to ChessLossComputer)
- Modify: `src/denoisr/scripts/config.py:332-557` (add CLI flag)

**Step 1: Add to TrainingConfig**

In `src/denoisr/scripts/config.py`, add to `TrainingConfig`:

```python
# Weight for illegal-move logit L2 penalty. Encourages the model to
# output low logits at illegal positions, improving accuracy evaluation
# robustness. Small values (0.01) prevent interference with policy loss.
illegal_penalty_weight: float = 0.01
```

**Step 2: Add CLI flag**

In `add_training_args()`:

```python
g.add_argument(
    "--illegal-penalty-weight", type=float, default=0.01,
    help="L2 penalty weight on illegal-move logits (default: 0.01)",
)
```

**Step 3: Update `training_config_from_args()`**

Add: `illegal_penalty_weight=args.illegal_penalty_weight,`

**Step 4: Wire into loss function in `train_phase1.py`**

Update the `ChessLossComputer` construction:

```python
loss_fn = ChessLossComputer(
    policy_weight=tcfg.policy_weight,
    value_weight=tcfg.value_weight,
    use_harmony_dream=tcfg.use_harmony_dream,
    harmony_ema_decay=tcfg.harmony_ema_decay,
    illegal_penalty_weight=tcfg.illegal_penalty_weight,
)
```

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q -k "not stockfish"`

**Step 6: Commit**

```bash
git add src/denoisr/scripts/config.py src/denoisr/scripts/train_phase1.py
git commit -m "feat: wire illegal penalty weight through config and CLI"
```

---

### Task 7: Update README Defaults + Final Verification

**Files:**
- Modify: `README.md` (update defaults table for temperature, label smoothing, stockfish-depth, illegal_penalty_weight)

**Step 1: Update README tables**

In the `denoisr-generate-data` flags table, add:

```markdown
| `--policy-temperature` | `150`                      | Softmax temperature for policy targets          |
| `--label-smoothing`    | `0.1`                      | Label smoothing epsilon for policy targets       |
```

Update `--stockfish-depth` default from 10 to 12.

In the Training optimization table, add:

```markdown
| `--illegal-penalty-weight` | `0.01` | L2 penalty weight on illegal-move logits |
```

**Step 2: Lint and type-check**

Run: `uvx ruff check src/ tests/`

Run: `uv run --with mypy mypy --strict src/denoisr/training/loss.py src/denoisr/training/supervised_trainer.py src/denoisr/data/stockfish_oracle.py src/denoisr/scripts/train_phase1.py src/denoisr/scripts/generate_data.py src/denoisr/scripts/config.py`

**Step 3: Full test suite**

Run: `uv run pytest tests/ -x -q`

Expected: All tests pass.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README with new defaults and flags"
```

---

### Execution Order and Dependencies

```
Task 1 (measure_accuracy fix) ─┐
Task 2 (illegal penalty)  ─────┤──> Task 6 (wire up config)
Task 3 (temperature/smoothing)─┘
Task 4 (LR schedule) ──────────────> independent
Task 5 (batch accuracy) ───────────> depends on Task 1 pattern
Task 7 (README + verification) ────> depends on all above
```

Tasks 1-4 can be done in parallel. Task 5 depends on the masking pattern from Task 1. Task 6 wires Task 2's penalty into config. Task 7 is final.
