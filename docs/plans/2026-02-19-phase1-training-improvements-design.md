# Phase 1 Training Improvements

Phase 1 training plateaus at ~19% top-1 accuracy, unable to reach the 30% gate. Seven compounding issues identified and fixed.

## Problem

Loss drops steadily (6.23 to 3.13) but top-1 accuracy stalls at ~19% after epoch 25. The model learns the distribution but cannot reliably predict the argmax of nearly-uniform targets.

## Fixes

### 1. Sharpen policy targets (T=100 to T=30)

`stockfish_oracle.py` — softmax temperature over centipawn scores reduced from T=100 to T=30. At T=100, the best move gets ~8% probability in a 30-move position; at T=30, it gets ~40-60%. This makes the argmax stable and the top-1 metric meaningful.

Requires data regeneration.

### 2. ExtendedBoardEncoder (12 to 110 planes)

Switch from `SimpleBoardEncoder` (12 piece planes only) to `ExtendedBoardEncoder` (12 pieces + 84 history + 14 metadata). Critical additions: side-to-move plane (the model currently cannot tell whose turn it is), castling rights, en passant, repetition detection.

`config.py` — add `num_planes` field to `ModelConfig`, default 110. `build_encoder` reads from config instead of hardcoding 12.

`generate_data.py` — use `ExtendedBoardEncoder`. Change work items from FEN strings to move sequences so workers can replay history (required for history planes).

Requires data regeneration and model re-initialization.

### 3. Cosine LR scheduler with warmup

`supervised_trainer.py` — add `CosineAnnealingLR` with linear warmup (3 epochs default). Relax encoder LR multiplier from 0.1x to 0.3x. Currently the encoder learns at 1e-5 with no decay, stalling in a saddle point after epoch 25.

### 4. Disable HarmonyDream for Phase 1

`train_phase1.py` — use fixed weights: `policy_weight=2.0, value_weight=0.5, use_harmony_dream=False`. HarmonyDream balances losses to equal magnitude, but policy CE (~8.3 nats over 4096 logits) is 7.5x larger than value CE (~1.1 nats over 3 classes). HarmonyDream downweights policy 0.56x and boosts value 4.5x — backwards for a policy-accuracy gate.

### 5. Improve evaluation metric

`train_phase1.py` — use proper `.eval()` method instead of manual `m.training = False`. Add top-5 accuracy alongside top-1 for diagnostic visibility.

### 6. Legal move masking in loss

`loss.py` — mask predicted logits where target is zero before softmax. With T=30, every legal move gets positive probability, so `target > 0` is a valid legal move indicator. Constrains softmax to ~30 legal moves instead of 4096 positions.

### 7. Board color-flip augmentation

`train_phase1.py` — with 50% probability per example during training: mirror board vertically, swap color planes, mirror policy target squares, swap win/loss in value target. Doubles effective dataset size.

## Files changed

| File | Changes |
|------|---------|
| `stockfish_oracle.py` | T=100 to T=30 |
| `generate_data.py` | ExtendedBoardEncoder, move-sequence work items |
| `config.py` | `num_planes` in ModelConfig |
| `train_phase1.py` | LR scheduler, HarmonyDream off, metric fix, augmentation |
| `supervised_trainer.py` | Cosine scheduler, encoder LR 0.3x |
| `loss.py` | Legal move masking |

## Workflow after implementation

```bash
uv run denoisr-init --output outputs/random_model.pt
uv run denoisr-generate-data --pgn data/lichess_elite_2025-01.pgn --max-examples 100000 --output outputs/training_data.pt
uv run denoisr-train-phase1 --checkpoint outputs/random_model.pt --data outputs/training_data.pt --output outputs/phase1.pt
```
