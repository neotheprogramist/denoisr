# Phase 1 Training Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 7 issues causing Phase 1 top-1 accuracy to plateau at 19%, enabling the 30% gate.

**Architecture:** Sharpen Stockfish policy targets (T=30), switch to 110-plane ExtendedBoardEncoder, add cosine LR scheduling, rebalance loss weights, mask illegal moves, improve the metric, and add board-flip augmentation.

**Tech Stack:** PyTorch, python-chess, multiprocessing, tqdm

---

### Task 1: Sharpen policy target temperature

**Files:**
- Modify: `src/denoisr/data/stockfish_oracle.py:42`
- Test: `tests/test_data/test_stockfish_oracle.py`

**Step 1: Write the failing test**

Add to `TestStockfishOracle` in `tests/test_data/test_stockfish_oracle.py`:

```python
def test_best_move_has_substantial_probability(self, oracle: StockfishOracle) -> None:
    """With T=30, the best move should get meaningful probability mass."""
    board = chess.Board()
    policy, _, _ = oracle.evaluate(board)
    max_prob = policy.data.max().item()
    # With T=30, best move in starting position should get >10%
    assert max_prob > 0.10
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data/test_stockfish_oracle.py::TestStockfishOracle::test_best_move_has_substantial_probability -v`
Expected: FAIL (at T=100 the max prob is ~5-8% for the starting position)

**Step 3: Change temperature**

In `src/denoisr/data/stockfish_oracle.py` line 42, change:
```python
probs = torch.softmax(t / 100.0, dim=0)
```
to:
```python
probs = torch.softmax(t / 30.0, dim=0)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data/test_stockfish_oracle.py -v`
Expected: All tests PASS

**Step 5: Commit**

```
feat: sharpen policy target temperature from T=100 to T=30
```

---

### Task 2: Legal move masking in loss

**Files:**
- Modify: `src/denoisr/training/loss.py:55-59`
- Test: `tests/test_training/test_loss.py`

**Step 1: Write the failing test**

Add to `TestChessLossComputer` in `tests/test_training/test_loss.py`:

```python
def test_illegal_logits_do_not_affect_loss(self, loss_fn: ChessLossComputer) -> None:
    """Changing logits at positions where target=0 should not change the loss."""
    B = 2
    pred_policy = torch.randn(B, 64, 64)
    pred_value = torch.softmax(torch.randn(B, 3), dim=-1)
    # Sparse target: only a few legal moves have nonzero probability
    target_policy = torch.zeros(B, 64, 64)
    target_policy[0, 4, 4] = 0.6  # e2-e4
    target_policy[0, 4, 12] = 0.4  # e2-e5
    target_policy[1, 1, 18] = 1.0  # single move
    target_value = torch.tensor([[0.4, 0.3, 0.3], [0.5, 0.2, 0.3]])

    loss_a, _ = loss_fn.compute(pred_policy, pred_value, target_policy, target_value)

    # Wildly change logits at illegal positions (where target is 0)
    pred_policy_b = pred_policy.clone()
    pred_policy_b[:, 0, 0] += 1000.0  # a1-a1 is never legal
    pred_policy_b[:, 7, 7] -= 1000.0

    loss_b, _ = loss_fn.compute(pred_policy_b, pred_value, target_policy, target_value)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_loss.py::TestChessLossComputer::test_illegal_logits_do_not_affect_loss -v`
Expected: FAIL (changing illegal logits currently changes the softmax normalization)

**Step 3: Add legal move masking**

In `src/denoisr/training/loss.py`, change the policy loss computation (lines 55-59) from:
```python
B = pred_policy.shape[0]
pred_flat = pred_policy.reshape(B, -1)
target_flat = target_policy.reshape(B, -1)
log_probs = F.log_softmax(pred_flat, dim=-1)
policy_loss = -(target_flat * log_probs).sum(dim=-1).mean()
```
to:
```python
B = pred_policy.shape[0]
pred_flat = pred_policy.reshape(B, -1)
target_flat = target_policy.reshape(B, -1)
# Mask illegal moves: set logits to -inf where target is zero
legal_mask = target_flat > 0
masked_logits = pred_flat.masked_fill(~legal_mask, float("-inf"))
log_probs = F.log_softmax(masked_logits, dim=-1)
policy_loss = -(target_flat * log_probs).sum(dim=-1).mean()
```

**Step 4: Run all loss tests**

Run: `uv run pytest tests/test_training/test_loss.py -v`
Expected: All tests PASS

**Step 5: Commit**

```
feat: mask illegal move logits in policy loss computation
```

---

### Task 3: Add num_planes to ModelConfig and switch to ExtendedBoardEncoder

**Files:**
- Modify: `src/denoisr/scripts/config.py:20-40`
- Modify: `src/denoisr/scripts/generate_data.py`
- Test: `tests/test_nn/test_encoder.py`

**Step 1: Write the failing test**

Add to `TestChessEncoder` in `tests/test_nn/test_encoder.py`:

```python
def test_110_plane_input(self, device: torch.device) -> None:
    """Encoder should accept 110-plane input from ExtendedBoardEncoder."""
    enc = ChessEncoder(num_planes=110, d_s=SMALL_D_S).to(device)
    x = torch.randn(2, 110, 8, 8, device=device)
    out = enc(x)
    assert out.shape == (2, 64, SMALL_D_S)
```

**Step 2: Run test to verify it passes (encoder already supports any num_planes)**

Run: `uv run pytest tests/test_nn/test_encoder.py::TestChessEncoder::test_110_plane_input -v`
Expected: PASS (encoder is parameterized by num_planes already)

**Step 3: Add num_planes to ModelConfig**

In `src/denoisr/scripts/config.py`, add `num_planes` field to `ModelConfig`:

```python
@dataclass(frozen=True)
class ModelConfig:
    num_planes: int = 110
    d_s: int = 256
    num_heads: int = 8
    num_layers: int = 15
    ffn_dim: int = 1024
    num_timesteps: int = 100
    world_model_layers: int = 12
    diffusion_layers: int = 6
    proj_dim: int = 256
```

Update `build_encoder` to use `cfg.num_planes`:

```python
def build_encoder(cfg: ModelConfig) -> ChessEncoder:
    return ChessEncoder(num_planes=cfg.num_planes, d_s=cfg.d_s)
```

Remove the `num_planes` parameter from `build_encoder` signature (it now always reads from config).

Update `add_model_args` to include `--num-planes`:

```python
g.add_argument("--num-planes", type=int, default=110, help="board encoder planes")
```

Update `config_from_args` to include `num_planes=args.num_planes`.

**Step 4: Update generate_data.py to use ExtendedBoardEncoder and move sequences**

Change the work item type from FEN strings to move sequences. This ensures `ExtendedBoardEncoder` has access to `board.move_stack` for history planes.

Replace `_extract_fens` with `_extract_positions`:

```python
_MoveSeq = list[tuple[int, int, int | None]]

def _extract_positions(pgn_path: Path, max_positions: int) -> list[_MoveSeq]:
    streamer = SimplePGNStreamer()
    positions: list[_MoveSeq] = []
    pbar = tqdm(total=max_positions, desc="Extracting positions", unit="pos", smoothing=0.3)

    for record in streamer.stream(pgn_path):
        moves_so_far: _MoveSeq = []
        for action in record.actions:
            if len(positions) >= max_positions:
                break
            positions.append(list(moves_so_far))
            pbar.update(1)
            moves_so_far.append((action.from_square, action.to_square, action.promotion))

        if len(positions) >= max_positions:
            break

    pbar.close()
    return positions
```

Update `_init_worker` to use `ExtendedBoardEncoder`:

```python
from denoisr.data.extended_board_encoder import ExtendedBoardEncoder

def _init_worker(stockfish_path: str, stockfish_depth: int) -> None:
    global _oracle, _encoder
    _oracle = StockfishOracle(path=stockfish_path, depth=stockfish_depth)
    _encoder = ExtendedBoardEncoder()
    atexit.register(_cleanup_oracle)
```

Update `_evaluate_position` to accept move sequences:

```python
def _evaluate_position(moves: _MoveSeq) -> _EvalResult:
    if _oracle is None or _encoder is None:
        raise RuntimeError("Worker not initialized")
    board = chess.Board()
    for from_sq, to_sq, promo in moves:
        board.push(chess.Move(from_sq, to_sq, promo))
    board_tensor = _encoder.encode(board)
    policy, value, _ = _oracle.evaluate(board)
    return (
        board_tensor.data.numpy(),
        policy.data.numpy(),
        (value.win, value.draw, value.loss),
    )
```

Update `generate_examples` to call `_extract_positions` instead of `_extract_fens`:

```python
def generate_examples(...) -> list[TrainingExample]:
    positions = _extract_positions(pgn_path, max_examples)
    print(f"Extracted {len(positions)} positions, evaluating with {num_workers} workers")

    examples: list[TrainingExample] = []

    with multiprocessing.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(stockfish_path, stockfish_depth),
    ) as pool:
        results = pool.imap_unordered(_evaluate_position, positions)
        for board_np, policy_np, (win, draw, loss) in tqdm(
            results, total=len(positions), desc="Evaluating positions", unit="pos",
            smoothing=0.1,
        ):
            ...
```

Update the module docstring to reflect `[N, 110, 8, 8]` boards.

Remove `SimpleBoardEncoder` import, add `ExtendedBoardEncoder` import.

**Step 5: Fix any callers that pass num_planes to build_encoder**

Search for `build_encoder(cfg,` calls and remove the second argument since `num_planes` now comes from config. Files to check: `init_model.py`, `train_phase1.py`, `train_phase2.py`, `train_phase3.py`, `play.py`.

**Step 6: Run tests**

Run: `uv run pytest tests/test_nn/test_encoder.py tests/test_data/test_extended_board_encoder.py -v`
Expected: All PASS

Run: `uvx ruff check src/`
Expected: All checks passed

**Step 7: Commit**

```
feat: switch to ExtendedBoardEncoder with 110 planes and move-sequence work items
```

---

### Task 4: Add cosine LR scheduler with warmup

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py`
- Modify: `src/denoisr/scripts/train_phase1.py`
- Test: `tests/test_training/test_supervised_trainer.py`

**Step 1: Write the failing test**

Add to `TestSupervisedTrainer` in `tests/test_training/test_supervised_trainer.py`:

```python
def test_scheduler_reduces_lr(self, trainer: SupervisedTrainer) -> None:
    """After stepping the scheduler, learning rates should decrease."""
    initial_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
    batch = _make_batch(8)
    # Simulate several epochs
    for _ in range(5):
        trainer.train_step(batch)
        trainer.scheduler_step()
    current_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
    # At least one group should have a lower LR
    assert any(c < i for c, i in zip(current_lrs, initial_lrs))
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py::TestSupervisedTrainer::test_scheduler_reduces_lr -v`
Expected: FAIL (no `scheduler_step` method)

**Step 3: Add scheduler to SupervisedTrainer**

In `src/denoisr/training/supervised_trainer.py`:

Add `total_epochs` and `warmup_epochs` parameters to `__init__`:

```python
def __init__(
    self,
    encoder: nn.Module,
    backbone: nn.Module,
    policy_head: nn.Module,
    value_head: nn.Module,
    loss_fn: ChessLossComputer,
    lr: float = 1e-4,
    device: torch.device | None = None,
    total_epochs: int = 100,
    warmup_epochs: int = 3,
) -> None:
```

Change encoder LR multiplier from 0.1 to 0.3:

```python
param_groups = [
    {"params": list(encoder.parameters()), "lr": lr * 0.3},
    {"params": list(backbone.parameters()), "lr": lr * 0.3},
    {"params": list(policy_head.parameters()), "lr": lr},
    {"params": list(value_head.parameters()), "lr": lr},
]
```

Add scheduler after optimizer creation:

```python
self._warmup_epochs = warmup_epochs
self._base_lrs = [g["lr"] for g in param_groups]
self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6
)
self._epoch = 0
```

Add `scheduler_step` method:

```python
def scheduler_step(self) -> None:
    self._epoch += 1
    if self._epoch <= self._warmup_epochs:
        # Linear warmup
        frac = self._epoch / self._warmup_epochs
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * frac
    else:
        self._scheduler.step()
```

**Step 4: Update train_phase1.py to call scheduler_step**

In `src/denoisr/scripts/train_phase1.py`, after `pbar.close()` (end of epoch batches), add:

```python
trainer.scheduler_step()
```

Pass `total_epochs` and `warmup_epochs` to SupervisedTrainer constructor:

```python
trainer = SupervisedTrainer(
    ...,
    total_epochs=args.epochs,
    warmup_epochs=3,
)
```

**Step 5: Update existing trainer tests that construct SupervisedTrainer**

The existing `trainer` fixture in `test_supervised_trainer.py` creates a `SupervisedTrainer`. It doesn't pass `total_epochs`/`warmup_epochs`, so defaults are fine. No changes needed.

**Step 6: Run tests**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -v`
Expected: All PASS

**Step 7: Commit**

```
feat: add cosine LR scheduler with linear warmup to SupervisedTrainer
```

---

### Task 5: Disable HarmonyDream and reweight loss for Phase 1

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py`

**Step 1: Change loss function instantiation**

In `src/denoisr/scripts/train_phase1.py`, change:

```python
loss_fn = ChessLossComputer(use_harmony_dream=True)
```

to:

```python
loss_fn = ChessLossComputer(
    policy_weight=2.0,
    value_weight=0.5,
    use_harmony_dream=False,
)
```

**Step 2: Run existing tests**

Run: `uv run pytest tests/test_training/test_loss.py tests/test_training/test_supervised_trainer.py -v`
Expected: All PASS

**Step 3: Commit**

```
fix: disable HarmonyDream for Phase 1, use 2:0.5 policy:value weights
```

---

### Task 6: Improve the accuracy metric

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py` (measure_top1 function)

**Step 1: Fix mode switching and add top-5 accuracy**

Replace the `measure_top1` function with `measure_accuracy` that returns both top-1 and top-5:

```python
def measure_accuracy(
    trainer: SupervisedTrainer,
    examples: list[TrainingExample],
    device: torch.device,
) -> tuple[float, float]:
    correct_1 = 0
    correct_5 = 0
    total = 0

    trainer.encoder.eval()
    trainer.backbone.eval()
    trainer.policy_head.eval()

    with torch.no_grad():
        for ex in examples:
            board = ex.board.data.unsqueeze(0).to(device)
            latent = trainer.encoder(board)
            features = trainer.backbone(latent)
            logits = trainer.policy_head(features).squeeze(0)

            pred_flat = logits.reshape(-1)
            target_flat = ex.policy.data.reshape(-1)
            target_idx = target_flat.argmax().item()

            top5 = pred_flat.topk(5).indices.tolist()
            if top5[0] == target_idx:
                correct_1 += 1
            if target_idx in top5:
                correct_5 += 1
            total += 1

    return correct_1 / max(total, 1), correct_5 / max(total, 1)
```

**Step 2: Update the training loop to use the new function**

Replace all references to `measure_top1` with `measure_accuracy`. Update the epoch printing:

```python
top1, top5 = measure_accuracy(trainer, holdout, device)

print(
    f"Epoch {epoch+1}/{args.epochs}: "
    f"avg_loss={avg_loss:.4f} top1={top1:.1%} top5={top5:.1%}"
)
```

Update the gate check and best_acc tracking to use `top1`.

**Step 3: Run tests**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS

**Step 4: Commit**

```
fix: use proper .eval() mode and add top-5 accuracy to Phase 1 metrics
```

---

### Task 7: Board color-flip augmentation

**Files:**
- Create: `src/denoisr/training/augmentation.py`
- Test: `tests/test_training/test_augmentation.py`
- Modify: `src/denoisr/scripts/train_phase1.py`

**Step 1: Write the failing test**

Create `tests/test_training/test_augmentation.py`:

```python
import chess
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.training.augmentation import flip_board, flip_policy, flip_value


class TestBoardFlip:
    def test_flip_is_involution(self) -> None:
        """Flipping twice returns the original."""
        board = torch.randn(12, 8, 8)
        assert torch.allclose(flip_board(flip_board(board, 12), 12), board)

    def test_flip_swaps_colors(self) -> None:
        """White pawns (plane 0) become black pawns (plane 6) after flip."""
        encoder = SimpleBoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board).data
        flipped = flip_board(tensor, 12)
        # White pawns on rank 1 should now be black pawns on rank 6
        assert flipped[6, 6, :].sum() == 8.0  # 8 pawns

    def test_flip_mirrors_ranks(self) -> None:
        """Rank 0 becomes rank 7 after flip."""
        encoder = SimpleBoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board).data
        flipped = flip_board(tensor, 12)
        # White rook was at (plane=3, rank=0, file=0), after flip
        # it becomes black rook at (plane=9, rank=7, file=0)
        assert flipped[9, 7, 0] == 1.0


class TestPolicyFlip:
    def test_flip_is_involution(self) -> None:
        """Flipping twice returns the original."""
        policy = torch.randn(64, 64)
        assert torch.allclose(flip_policy(flip_policy(policy)), policy)

    def test_flip_mirrors_squares(self) -> None:
        """Move from e2(12) to e4(28) becomes e7(52) to e5(36)."""
        policy = torch.zeros(64, 64)
        policy[12, 28] = 1.0  # e2-e4
        flipped = flip_policy(policy)
        # e2 = rank1*8+file4 = 12, flipped rank = 7-1=6, sq = 6*8+4 = 52
        # e4 = rank3*8+file4 = 28, flipped rank = 7-3=4, sq = 4*8+4 = 36
        assert flipped[52, 36] == 1.0


class TestValueFlip:
    def test_flip_swaps_win_loss(self) -> None:
        """Win and loss swap, draw unchanged."""
        win, draw, loss = 0.6, 0.1, 0.3
        fw, fd, fl = flip_value(win, draw, loss)
        assert fw == loss
        assert fd == draw
        assert fl == win
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_augmentation.py -v`
Expected: FAIL (module not found)

**Step 3: Implement augmentation functions**

Create `src/denoisr/training/augmentation.py`:

```python
"""Board color-flip augmentation for training data.

Flips a chess position by mirroring ranks and swapping colors.
This doubles the effective dataset: every White-to-play position
becomes an equivalent Black-to-play position (and vice versa).
"""

import torch
from torch import Tensor

# Precomputed square flip: rank r -> rank 7-r, file unchanged
# Square index = rank*8 + file. Flipped = (7-rank)*8 + file.
_SQUARE_FLIP = torch.tensor([(7 - (i // 8)) * 8 + (i % 8) for i in range(64)])


def flip_board(board: Tensor, num_planes: int) -> Tensor:
    """Flip board tensor [C, 8, 8]: mirror ranks, swap white/black planes.

    Handles both 12-plane (simple) and 110-plane (extended) encoders.
    """
    flipped = board.clone()
    # Mirror ranks (rank 0 <-> rank 7)
    flipped = flipped.flip(1)

    # Swap white/black piece planes in groups of 12
    # Planes 0-5 = white pieces, 6-11 = black pieces
    num_piece_groups = min(num_planes, 96) // 12
    for g in range(num_piece_groups):
        offset = g * 12
        white = flipped[offset : offset + 6].clone()
        black = flipped[offset + 6 : offset + 12].clone()
        flipped[offset : offset + 6] = black
        flipped[offset + 6 : offset + 12] = white

    # Extended encoder metadata planes (starting at plane 96)
    if num_planes > 96:
        meta = 96
        # Castling: swap white (meta+0, meta+1) <-> black (meta+2, meta+3)
        wk, wq = flipped[meta].clone(), flipped[meta + 1].clone()
        flipped[meta] = flipped[meta + 2]
        flipped[meta + 1] = flipped[meta + 3]
        flipped[meta + 2] = wk
        flipped[meta + 3] = wq
        # En passant (meta+4): already rank-flipped above
        # Rule50 (meta+5): unchanged
        # Repetition (meta+6): unchanged
        # Side to move (meta+7): invert
        flipped[meta + 7] = 1.0 - flipped[meta + 7]
        # Material: swap white (meta+8) <-> black (meta+9)
        w_mat = flipped[meta + 8].clone()
        flipped[meta + 8] = flipped[meta + 9]
        flipped[meta + 9] = w_mat
        # Check attackers: swap white (meta+10) <-> black (meta+11)
        w_chk = flipped[meta + 10].clone()
        flipped[meta + 10] = flipped[meta + 11]
        flipped[meta + 11] = w_chk
        # Bishops (meta+12, meta+13): unchanged (symmetric property)

    return flipped


def flip_policy(policy: Tensor) -> Tensor:
    """Flip policy tensor [64, 64]: mirror source and destination squares."""
    return policy[_SQUARE_FLIP][:, _SQUARE_FLIP]


def flip_value(win: float, draw: float, loss: float) -> tuple[float, float, float]:
    """Swap win and loss probabilities."""
    return loss, draw, win
```

**Step 4: Run augmentation tests**

Run: `uv run pytest tests/test_training/test_augmentation.py -v`
Expected: All PASS

**Step 5: Integrate augmentation into train_phase1.py**

In `src/denoisr/scripts/train_phase1.py`, add a random flip to each training batch. In the training loop, after slicing the batch:

```python
import random as _random
from denoisr.training.augmentation import flip_board, flip_policy, flip_value

# Inside the batch loop, after `batch = train[i : i + bs]`:
augmented: list[TrainingExample] = []
for ex in batch:
    if _random.random() < 0.5:
        augmented.append(
            TrainingExample(
                board=BoardTensor(flip_board(ex.board.data, ex.board.data.shape[0])),
                policy=PolicyTarget(flip_policy(ex.policy.data)),
                value=ValueTarget(*flip_value(ex.value.win, ex.value.draw, ex.value.loss)),
            )
        )
    else:
        augmented.append(ex)
batch = augmented
```

Add the necessary imports at the top of the file.

**Step 6: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS

**Step 7: Commit**

```
feat: add board color-flip augmentation for Phase 1 training
```

---

### Task 8: Full verification

**Step 1: Ruff**

Run: `uvx ruff check src/ tests/`
Expected: All checks passed

**Step 2: Mypy**

Run: `uv run --with mypy mypy --strict src/denoisr/`
Expected: 0 errors

**Step 3: Full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass (197 + new tests)

**Step 4: CLI help verification**

Run: `uv run denoisr-generate-data --help`
Expected: Shows all flags

Run: `uv run denoisr-train-phase1 --help`
Expected: Shows `--data` flag, no `--pgn`/`--stockfish` flags

Run: `uv run denoisr-init --help`
Expected: Shows `--num-planes` flag with default 110

**Step 5: Final commit (if any loose changes)**

```
chore: verify all Phase 1 training improvements pass CI checks
```
