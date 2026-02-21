# Phase 1 Pipeline Overhaul Design

## Problem

Phase 1 training ran 100 epochs (~9.2 hours) on CUDA but accuracy plateaued at 4.2% top-1 (30% required for Phase 1 gate). Loss continued decreasing (5.76 → 3.70), indicating the model was learning but the accuracy metric was broken.

Root cause analysis identified 4 issues, one critical:

1. **Critical: `measure_accuracy()` doesn't mask illegal moves** — evaluates raw 4096-dim logits instead of the ~25 legal moves
2. **High: Policy targets too soft** — Stockfish centipawn temperature=30 produces near-uniform distributions over legal moves
3. **High: LR decays too fast** — cosine schedule reaches near-minimum by epoch 70
4. **Medium: No per-batch accuracy visibility** — accuracy measured only per-epoch on holdout

## Approach: Surgical Fixes + Training Improvements

Fix the measurement bug, sharpen training signal, extend learning schedule, and improve observability — without architectural changes.

## Fix 1: Legal Move Masking in `measure_accuracy()`

### Current (broken)

```python
pred_flat = logits.reshape(len(batch), -1)    # [B, 4096] — ALL entries
top5 = pred_flat.topk(5, dim=-1).indices      # picks from 4096 including ~4070 illegal
correct_1 += (top5[:, 0] == target_idx).sum().item()
```

### Fixed

```python
pred_flat = logits.reshape(len(batch), -1)
target_flat = targets.reshape(len(batch), -1)
legal_mask = target_flat > 0
masked_logits = pred_flat.masked_fill(~legal_mask, float("-inf"))
target_idx = target_flat.argmax(dim=-1)
top5 = masked_logits.topk(5, dim=-1).indices
correct_1 += (top5[:, 0] == target_idx).sum().item()
correct_5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()
```

### Additionally: Illegal-move logit penalty

Add a small L2 penalty on logits at illegal positions to the loss function. This encourages the model to actively suppress illegal entries rather than relying solely on masking:

```python
# In ChessLossComputer.compute(), after policy_loss:
illegal_logits = pred_flat.masked_fill(legal_mask, 0.0)
illegal_penalty = (illegal_logits ** 2).mean() * 0.01
```

The penalty is small (0.01 weight) — it shouldn't interfere with the primary policy loss but gradually pushes illegal logits toward zero over training.

## Fix 2: Sharper Policy Targets + Label Smoothing

### Current

```python
# stockfish_oracle.py:42
probs = torch.softmax(t / 30.0, dim=0)
```

Temperature=30 means a move 30cp better than the next-best gets 2.7x probability weight. In positions with several reasonable moves (common in chess), the target distribution is nearly uniform.

### Proposed

```python
probs = torch.softmax(t / 150.0, dim=0)
# Apply label smoothing
n_legal = len(legal_moves)
epsilon = 0.1
smoothed = (1 - epsilon) * probs + epsilon / n_legal
```

Temperature=150 gives a 90cp advantage ~65% probability (vs ~95% at temp=30 for the same gap). Label smoothing (epsilon=0.1) redistributes 10% of probability mass uniformly over legal moves, preventing overconfident peaks.

### CLI changes

Add `--policy-temperature` (default 150) and `--label-smoothing` (default 0.1) flags to `denoisr-generate-data`.

### Data regeneration required

Existing training data uses temperature=30. Must regenerate:

```bash
uv run denoisr-generate-data \
    --pgn data/lichess_elite_2025-01.pgn \
    --max-examples 100000 \
    --stockfish-depth 12 \
    --policy-temperature 150 \
    --label-smoothing 0.1 \
    --output outputs/training_data_v2.pt
```

## Fix 3: Slower LR Decay

### Current

```python
self._scheduler = CosineAnnealingLR(
    optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
)
```

With 100 epochs and 5 warmup, `T_max=95`. LR reaches 50% of peak by epoch 52, ~10% by epoch 82.

### Proposed

```python
self._scheduler = CosineAnnealingLR(
    optimizer, T_max=max(1, (total_epochs - warmup_epochs) * 2), eta_min=min_lr
)
```

Doubling `T_max` means the cosine curve only completes half a cycle during training. LR stays above 50% of peak until epoch ~67. The model gets ~15 more epochs of effective learning.

## Fix 4: Per-Batch Masked Accuracy Logging

### Current

Accuracy is only measured once per epoch on the holdout set (~100 data points per run).

### Proposed

Compute masked top-1 accuracy on each training batch (cheap — just argmax comparison):

```python
# In _forward_backward(), after computing pred_policy:
with torch.no_grad():
    pf = pred_policy.reshape(B, -1)
    tf = target_policies.reshape(B, -1)
    mask = tf > 0
    masked = pf.masked_fill(~mask, float("-inf"))
    batch_top1 = (masked.argmax(-1) == tf.argmax(-1)).float().mean().item()
breakdown["batch_top1"] = batch_top1
```

Log to TensorBoard under `accuracy/batch_top1`. This gives per-step visibility into whether the model is actually learning to select the right move.

## Files Changed

| File | Change |
|---|---|
| `scripts/train_phase1.py` | Fix `measure_accuracy()` with legal-move masking |
| `training/loss.py` | Add illegal-move logit penalty term |
| `training/supervised_trainer.py` | Add per-batch masked accuracy to `_forward_backward()`, slow LR schedule (T_max * 2) |
| `data/stockfish_oracle.py` | Configurable temperature and label smoothing |
| `scripts/generate_data.py` | Add `--policy-temperature` and `--label-smoothing` CLI flags |

## Files NOT Changed

| File | Reason |
|---|---|
| `nn/policy_backbone.py` | Smolgen per-layer deferred — architectural change |
| `nn/policy_head.py` | Head enlargement deferred — architectural change |
| `nn/encoder.py` | No issues found |
| `nn/value_head.py` | Working correctly |
| `training/augmentation.py` | Working correctly |
| `training/dataset.py` | Working correctly |

## Expected Impact

| Metric | Before | Expected After |
|---|---|---|
| Top-1 accuracy (measured) | 4.2% | 25-40%+ (masking fix reveals true accuracy) |
| Top-5 accuracy (measured) | 11.7% | 50-70%+ |
| Training signal quality | Noisy (soft targets) | Clean (sharp + smoothed) |
| Effective learning epochs | ~50 (LR too low after) | ~80 |
| Observability | Per-epoch only | Per-batch + per-epoch |

## Success Criteria

1. `measure_accuracy()` uses legal-move masking — same mask as loss function
2. Illegal-move penalty is active and logged
3. Policy temperature and label smoothing are configurable CLI flags
4. Training data regenerated with temperature=150, smoothing=0.1
5. LR schedule uses `T_max * 2` for slower decay
6. Per-batch masked accuracy visible in TensorBoard
7. All existing tests pass; new tests for masking, temperature, smoothing
8. Phase 1 gate (30% top-1) is achievable within 100 epochs
