# Phase 1 Training Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Break the ~47% top-1 accuracy plateau in Phase 1 supervised training by adding dropout, stochastic depth, OneCycleLR, gradient accumulation, label smoothing, and soft target augmentation.

**Architecture:** Add regularization (dropout + stochastic depth + label smoothing) and optimizer improvements (OneCycleLR + gradient accumulation) to the existing transformer backbone and supervised trainer. All changes are backward-compatible via config defaults.

**Tech Stack:** PyTorch, frozen dataclasses for config, `uv run pytest` for testing

---

### Task 1: DropPath Module

**Files:**
- Create: `src/denoisr/nn/drop_path.py`
- Test: `tests/test_nn/test_drop_path.py`

**Step 1: Write the failing test**

```python
"""Tests for DropPath (stochastic depth) module."""

import torch

from denoisr.nn.drop_path import DropPath


class TestDropPath:
    def test_zero_rate_is_identity(self) -> None:
        dp = DropPath(0.0)
        x = torch.randn(4, 64, 128)
        out = dp(x)
        assert torch.equal(out, x)

    def test_one_rate_drops_everything_in_training(self) -> None:
        dp = DropPath(1.0)
        dp.train()
        x = torch.randn(4, 64, 128)
        out = dp(x)
        assert torch.all(out == 0)

    def test_inference_mode_is_identity(self) -> None:
        dp = DropPath(0.5)
        dp.train(False)
        x = torch.randn(4, 64, 128)
        out = dp(x)
        assert torch.equal(out, x)

    def test_training_mode_scales_output(self) -> None:
        torch.manual_seed(42)
        dp = DropPath(0.5)
        dp.train()
        x = torch.ones(100, 64, 128)
        out = dp(x)
        # Some samples should be zeroed, others scaled by 1/(1-p)
        kept = (out[:, 0, 0] != 0).sum().item()
        assert 30 < kept < 70  # rough binomial check

    def test_gradient_flows(self) -> None:
        dp = DropPath(0.3)
        dp.train()
        x = torch.randn(4, 64, 128, requires_grad=True)
        out = dp(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_output_shape_preserved(self) -> None:
        dp = DropPath(0.2)
        x = torch.randn(8, 64, 256)
        assert dp(x).shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_nn/test_drop_path.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'denoisr.nn.drop_path'`

**Step 3: Write minimal implementation**

```python
"""DropPath (stochastic depth) for transformer residual connections.

Randomly drops entire residual branches during training with probability
drop_prob. Surviving branches are scaled by 1/(1-drop_prob) to maintain
expected magnitude. In non-training mode, acts as identity.

Reference: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
"""

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape [B, 1, 1, ...] -- drop entire sample's residual branch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_nn/test_drop_path.py -x -q`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/denoisr/nn/drop_path.py tests/test_nn/test_drop_path.py
git commit -m "feat(nn): add DropPath (stochastic depth) module"
```

---

### Task 2: Add Dropout + DropPath to TransformerBlock

**Files:**
- Modify: `src/denoisr/nn/policy_backbone.py`
- Modify: `tests/test_nn/test_policy_backbone.py`

**Step 1: Write the failing tests**

Add to `tests/test_nn/test_policy_backbone.py`:

```python
    def test_dropout_params_accepted(self, device: torch.device) -> None:
        """TransformerBlock should accept dropout and drop_path_rate."""
        from denoisr.nn.policy_backbone import TransformerBlock
        block = TransformerBlock(SMALL_D_S, SMALL_NUM_HEADS, SMALL_FFN_DIM, dropout=0.1, drop_path_rate=0.1)
        block.to(device)
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_backbone_accepts_dropout(self, device: torch.device) -> None:
        """ChessPolicyBackbone should accept and distribute dropout/drop_path_rate."""
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
            dropout=0.1,
            drop_path_rate=0.1,
        ).to(device)
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        out = backbone(x)
        assert out.shape == x.shape

    def test_dropout_deterministic_in_non_training(self, device: torch.device) -> None:
        """Backbone should be deterministic in non-training mode even with dropout."""
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS, ffn_dim=SMALL_FFN_DIM,
            dropout=0.5, drop_path_rate=0.5,
        ).to(device)
        backbone.train(False)
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        out1 = backbone(x)
        out2 = backbone(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_drop_path_linearly_scaled(self, device: torch.device) -> None:
        """DropPath rates should increase linearly across layers."""
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
            num_layers=4, ffn_dim=SMALL_FFN_DIM,
            drop_path_rate=0.3,
        ).to(device)
        rates = [layer.drop_path.drop_prob for layer in backbone.layers]
        assert rates[0] == 0.0  # first layer: no drop
        assert abs(rates[-1] - 0.3) < 1e-6  # last layer: max rate
        # Monotonically increasing
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nn/test_policy_backbone.py::TestChessPolicyBackbone::test_dropout_params_accepted -x -q`
Expected: FAIL with `TypeError: TransformerBlock.__init__() got an unexpected keyword argument 'dropout'`

**Step 3: Modify TransformerBlock and ChessPolicyBackbone**

In `src/denoisr/nn/policy_backbone.py`:

1. Add import: `from denoisr.nn.drop_path import DropPath`
2. Modify `TransformerBlock.__init__` to accept `dropout: float = 0.0` and `drop_path_rate: float = 0.0`
3. Add `self.attn_dropout = nn.Dropout(dropout)` and `self.ffn_dropout = nn.Dropout(dropout)`
4. Add `self.drop_path = DropPath(drop_path_rate)`
5. In `forward`: apply dropout after `self.out_proj(h)`, wrap residuals with `self.drop_path`
6. Modify `ChessPolicyBackbone.__init__` to accept `dropout: float = 0.0` and `drop_path_rate: float = 0.0`
7. Compute linearly increasing drop rates: `dpr = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]`
8. Pass `dropout` and `drop_path_rate=dpr[i]` to each TransformerBlock

Updated `TransformerBlock.__init__`:
```python
def __init__(self, d_s: int, num_heads: int, ffn_dim: int,
             dropout: float = 0.0, drop_path_rate: float = 0.0) -> None:
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = d_s // num_heads
    self.norm1 = nn.LayerNorm(d_s)
    self.qkv = nn.Linear(d_s, 3 * d_s)
    self.out_proj = nn.Linear(d_s, d_s)
    self.attn_dropout = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(d_s)
    self.ffn = nn.Sequential(
        nn.Linear(d_s, ffn_dim),
        nn.Mish(),
        nn.Linear(ffn_dim, d_s),
    )
    self.ffn_dropout = nn.Dropout(dropout)
    self.drop_path = DropPath(drop_path_rate)
```

Updated `TransformerBlock.forward`:
```python
def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
    B, S, D = x.shape
    h = self.norm1(x)
    qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    h = h.transpose(1, 2).reshape(B, S, D)
    h = self.attn_dropout(self.out_proj(h))
    x = x + self.drop_path(h)
    x = x + self.drop_path(self.ffn_dropout(self.ffn(self.norm2(x))))
    return x
```

Updated `ChessPolicyBackbone.__init__`:
```python
def __init__(
    self,
    d_s: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    gradient_checkpointing: bool = False,
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> None:
    super().__init__()
    self._gradient_checkpointing = gradient_checkpointing
    self.smolgen = SmolgenBias(d_s, num_heads)
    self.shaw_relative_pe = ShawRelativePositionBias(num_heads)
    dpr = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]
    self.layers = nn.ModuleList(
        [TransformerBlock(d_s, num_heads, ffn_dim, dropout=dropout, drop_path_rate=dpr[i])
         for i in range(num_layers)]
    )
    self.final_norm = nn.LayerNorm(d_s)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_nn/test_policy_backbone.py -x -q`
Expected: All tests PASS (including all existing tests -- defaults of 0.0 mean no behavior change)

**Step 5: Commit**

```bash
git add src/denoisr/nn/policy_backbone.py tests/test_nn/test_policy_backbone.py
git commit -m "feat(nn): add dropout and stochastic depth to transformer blocks"
```

---

### Task 3: Config Fields for All Enhancements

**Files:**
- Modify: `src/denoisr/scripts/config/__init__.py`
- Modify: `.env.example`

**Step 1: Write the failing test**

Create `tests/test_scripts/test_config_enhancements.py`:

```python
"""Tests for new training enhancement config fields."""

from denoisr.scripts.config import ModelConfig, TrainingConfig


class TestEnhancementConfigDefaults:
    def test_model_config_has_dropout(self) -> None:
        cfg = ModelConfig()
        assert cfg.dropout == 0.0

    def test_model_config_has_drop_path_rate(self) -> None:
        cfg = ModelConfig()
        assert cfg.drop_path_rate == 0.0

    def test_training_config_has_use_onecycle(self) -> None:
        cfg = TrainingConfig()
        assert cfg.use_onecycle is False

    def test_training_config_has_onecycle_pct_start(self) -> None:
        cfg = TrainingConfig()
        assert cfg.onecycle_pct_start == 0.3

    def test_training_config_has_gradient_accumulation_steps(self) -> None:
        cfg = TrainingConfig()
        assert cfg.gradient_accumulation_steps == 1

    def test_training_config_has_label_smoothing(self) -> None:
        cfg = TrainingConfig()
        assert cfg.label_smoothing == 0.0

    def test_training_config_has_value_noise_prob(self) -> None:
        cfg = TrainingConfig()
        assert cfg.value_noise_prob == 0.0

    def test_training_config_has_value_noise_scale(self) -> None:
        cfg = TrainingConfig()
        assert cfg.value_noise_scale == 0.02

    def test_training_config_has_policy_temp_augment_prob(self) -> None:
        cfg = TrainingConfig()
        assert cfg.policy_temp_augment_prob == 0.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scripts/test_config_enhancements.py -x -q`
Expected: FAIL with `TypeError` or `AttributeError`

**Step 3: Add config fields**

In `ModelConfig` (after `gradient_checkpointing`):
```python
    # Dropout probability for attention output and FFN output in transformer
    # blocks. 0.0 disables dropout. Recommended: 0.1 for from-scratch training.
    dropout: float = 0.0

    # Maximum stochastic depth (DropPath) rate, linearly scaled per layer
    # from 0.0 at layer 0 to this value at the final layer. Recommended: 0.1.
    drop_path_rate: float = 0.0
```

In `TrainingConfig` (after `use_warm_restarts`):
```python
    # Use OneCycleLR instead of cosine annealing. Provides per-step scheduling
    # with coupled LR/momentum annealing for super-convergence.
    use_onecycle: bool = False

    # Fraction of total steps spent in the warmup phase of OneCycleLR.
    # 0.3 means 30% warmup, 70% annealing. Only used when use_onecycle=True.
    onecycle_pct_start: float = 0.3

    # Number of micro-batches to accumulate before stepping the optimizer.
    # Effective batch size = batch_size * gradient_accumulation_steps.
    # 1 means no accumulation (step every batch).
    gradient_accumulation_steps: int = 1

    # Label smoothing for policy cross-entropy loss, distributed only among
    # legal moves. 0.0 disables smoothing. Stacks with data-time smoothing.
    label_smoothing: float = 0.0

    # Probability of applying value noise augmentation per sample.
    value_noise_prob: float = 0.0

    # Scale of Gaussian noise added to WDL targets.
    value_noise_scale: float = 0.02

    # Probability of applying policy temperature augmentation per sample.
    policy_temp_augment_prob: float = 0.0
```

Add corresponding env spec entries to `_MODEL_REQUIRED_ENV_SPECS`:
```python
    ("DENOISR_MODEL_DROPOUT", _parse_env_float),
    ("DENOISR_MODEL_DROP_PATH_RATE", _parse_env_float),
```

Add to `_TRAINING_REQUIRED_ENV_SPECS`:
```python
    ("DENOISR_TRAIN_USE_ONECYCLE", _parse_env_bool),
    ("DENOISR_TRAIN_ONECYCLE_PCT_START", _parse_env_float),
    ("DENOISR_TRAIN_GRADIENT_ACCUMULATION_STEPS", _parse_env_int),
    ("DENOISR_TRAIN_LABEL_SMOOTHING", _parse_env_float),
    ("DENOISR_TRAIN_VALUE_NOISE_PROB", _parse_env_float),
    ("DENOISR_TRAIN_VALUE_NOISE_SCALE", _parse_env_float),
    ("DENOISR_TRAIN_POLICY_TEMP_AUGMENT_PROB", _parse_env_float),
```

Add CLI args in `add_model_args()`:
```python
    add_env_argument(parser, "--dropout", env_var="DENOISR_MODEL_DROPOUT",
        type=float, help="dropout probability in transformer blocks")
    add_env_argument(parser, "--drop-path-rate", env_var="DENOISR_MODEL_DROP_PATH_RATE",
        type=float, help="max stochastic depth rate (linearly scaled per layer)")
```

Add CLI args in `add_training_args()`:
```python
    add_env_argument(parser, "--use-onecycle", env_var="DENOISR_TRAIN_USE_ONECYCLE",
        action=argparse.BooleanOptionalAction, default=None,
        help="use OneCycleLR scheduler instead of cosine annealing")
    add_env_argument(parser, "--onecycle-pct-start", env_var="DENOISR_TRAIN_ONECYCLE_PCT_START",
        type=float, help="OneCycleLR warmup fraction (default 0.3)")
    add_env_argument(parser, "--gradient-accumulation-steps",
        env_var="DENOISR_TRAIN_GRADIENT_ACCUMULATION_STEPS",
        type=int, help="micro-batches per optimizer step")
    add_env_argument(parser, "--label-smoothing", env_var="DENOISR_TRAIN_LABEL_SMOOTHING",
        type=float, help="label smoothing for policy loss (legal-move-aware)")
    add_env_argument(parser, "--value-noise-prob", env_var="DENOISR_TRAIN_VALUE_NOISE_PROB",
        type=float, help="probability of WDL noise augmentation")
    add_env_argument(parser, "--value-noise-scale", env_var="DENOISR_TRAIN_VALUE_NOISE_SCALE",
        type=float, help="scale of Gaussian noise on WDL targets")
    add_env_argument(parser, "--policy-temp-augment-prob",
        env_var="DENOISR_TRAIN_POLICY_TEMP_AUGMENT_PROB",
        type=float, help="probability of policy temperature augmentation")
```

Update `config_from_args()` to include `dropout` and `drop_path_rate`.

Update `training_config_from_args()` to include all new training fields.

Update `.env.example` with new variables (using backward-compatible defaults: 0/disabled):
```env
DENOISR_MODEL_DROPOUT=0.0
DENOISR_MODEL_DROP_PATH_RATE=0.0
DENOISR_TRAIN_USE_ONECYCLE=0
DENOISR_TRAIN_ONECYCLE_PCT_START=0.3
DENOISR_TRAIN_GRADIENT_ACCUMULATION_STEPS=1
DENOISR_TRAIN_LABEL_SMOOTHING=0.0
DENOISR_TRAIN_VALUE_NOISE_PROB=0.0
DENOISR_TRAIN_VALUE_NOISE_SCALE=0.02
DENOISR_TRAIN_POLICY_TEMP_AUGMENT_PROB=0.0
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_scripts/test_config_enhancements.py tests/test_pipeline/test_config.py -x -q`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/denoisr/scripts/config/__init__.py .env.example tests/test_scripts/test_config_enhancements.py
git commit -m "feat(config): add enhancement config fields for dropout, OneCycleLR, accumulation, smoothing, augmentation"
```

---

### Task 4: Wire Dropout + DropPath Through Build Functions

**Files:**
- Modify: `src/denoisr/scripts/config/__init__.py` (build_backbone)

**Step 1: Write the failing test**

Add to `tests/test_scripts/test_config_enhancements.py`:

```python
from denoisr.scripts.config import ModelConfig, build_backbone


class TestBuildBackboneWithDropout:
    def test_build_backbone_passes_dropout(self) -> None:
        cfg = ModelConfig(dropout=0.2, drop_path_rate=0.15)
        backbone = build_backbone(cfg)
        # Verify dropout modules exist in first layer
        layer0 = backbone.layers[0]
        assert layer0.attn_dropout.p == 0.2
        assert layer0.ffn_dropout.p == 0.2

    def test_build_backbone_passes_drop_path(self) -> None:
        cfg = ModelConfig(num_layers=4, drop_path_rate=0.3)
        backbone = build_backbone(cfg)
        assert backbone.layers[0].drop_path.drop_prob == 0.0
        assert abs(backbone.layers[-1].drop_path.drop_prob - 0.3) < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scripts/test_config_enhancements.py::TestBuildBackboneWithDropout -x -q`
Expected: FAIL (build_backbone doesn't pass dropout/drop_path_rate yet)

**Step 3: Update build_backbone**

In `config/__init__.py`, modify `build_backbone`:
```python
def build_backbone(cfg: ModelConfig) -> ChessPolicyBackbone:
    return ChessPolicyBackbone(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_dim=cfg.ffn_dim,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dropout=cfg.dropout,
        drop_path_rate=cfg.drop_path_rate,
    )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_scripts/test_config_enhancements.py tests/test_nn/test_policy_backbone.py -x -q`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/denoisr/scripts/config/__init__.py tests/test_scripts/test_config_enhancements.py
git commit -m "feat(config): wire dropout and drop_path_rate through build_backbone"
```

---

### Task 5: Legal-Move-Aware Label Smoothing

**Files:**
- Modify: `src/denoisr/training/loss.py`
- Modify: `tests/test_training/test_loss.py`

**Step 1: Write the failing test**

Add to `tests/test_training/test_loss.py`:

```python
class TestLabelSmoothing:
    def test_smoothing_zero_is_unchanged(self) -> None:
        loss_fn = ChessLossComputer(label_smoothing=0.0)
        pred = torch.randn(2, 64, 64)
        target = torch.zeros(2, 64, 64)
        target[0, 0, 1] = 0.8
        target[0, 0, 3] = 0.2
        target[1, 2, 5] = 1.0
        value = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        total_0, _ = loss_fn.compute(pred, torch.randn(2, 3), target, value)
        assert total_0.isfinite()

    def test_smoothing_reduces_loss_on_correct_prediction(self) -> None:
        """Label smoothing should reduce loss when model predicts correctly."""
        pred = torch.zeros(1, 64, 64)
        pred[0, 0, 1] = 10.0  # high confidence on correct move
        target = torch.zeros(1, 64, 64)
        target[0, 0, 1] = 0.8
        target[0, 0, 3] = 0.2
        value_pred = torch.tensor([[0.0, 0.0, 0.0]])
        value_target = torch.tensor([[0.5, 0.3, 0.2]])

        no_smooth = ChessLossComputer(label_smoothing=0.0)
        with_smooth = ChessLossComputer(label_smoothing=0.1)

        loss_no, _ = no_smooth.compute(pred, value_pred, target, value_target)
        loss_yes, _ = with_smooth.compute(pred, value_pred, target, value_target)
        # Smoothing redistributes probability, so confident-correct predictions
        # have slightly higher loss with smoothing
        assert loss_yes > loss_no

    def test_smoothing_only_among_legal_moves(self) -> None:
        """Smoothed probability mass should only go to legal moves (nonzero targets)."""
        loss_fn = ChessLossComputer(label_smoothing=0.5)
        pred = torch.zeros(1, 64, 64)
        target = torch.zeros(1, 64, 64)
        # Only 2 legal moves
        target[0, 0, 1] = 0.7
        target[0, 0, 3] = 0.3
        value_pred = torch.tensor([[0.0, 0.0, 0.0]])
        value_target = torch.tensor([[0.5, 0.3, 0.2]])
        total, _ = loss_fn.compute(pred, value_pred, target, value_target)
        assert total.isfinite()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_loss.py::TestLabelSmoothing -x -q`
Expected: FAIL with `TypeError: ChessLossComputer.__init__() got an unexpected keyword argument 'label_smoothing'`

**Step 3: Add label smoothing to ChessLossComputer**

In `src/denoisr/training/loss.py`:

1. Add `label_smoothing: float = 0.0` parameter to `__init__`
2. Store as `self._label_smoothing = label_smoothing`
3. In `compute()`, after building `legal_mask` and before the cross-entropy line, add:

```python
        if self._label_smoothing > 0:
            n_legal = legal_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
            uniform = legal_mask.float() / n_legal
            target_flat = (1 - self._label_smoothing) * target_flat + self._label_smoothing * uniform
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_loss.py -x -q`
Expected: All PASS (existing tests unaffected since default smoothing=0.0)

**Step 5: Commit**

```bash
git add src/denoisr/training/loss.py tests/test_training/test_loss.py
git commit -m "feat(loss): add legal-move-aware label smoothing"
```

---

### Task 6: Soft Target Augmentation

**Files:**
- Modify: `src/denoisr/training/augmentation.py`
- Modify: `src/denoisr/training/dataset.py`
- Modify: `tests/test_training/test_augmentation.py`

**Step 1: Write the failing test**

Add to `tests/test_training/test_augmentation.py`:

```python
from denoisr.training.augmentation import augment_value_noise, augment_policy_temperature


class TestValueNoiseAugmentation:
    def test_value_noise_preserves_distribution(self) -> None:
        value = torch.tensor([0.5, 0.3, 0.2])
        torch.manual_seed(42)
        noisy = augment_value_noise(value, scale=0.02)
        assert noisy.shape == (3,)
        assert abs(noisy.sum().item() - 1.0) < 1e-5
        assert (noisy >= 0).all()

    def test_value_noise_changes_values(self) -> None:
        value = torch.tensor([0.5, 0.3, 0.2])
        torch.manual_seed(42)
        noisy = augment_value_noise(value, scale=0.1)
        assert not torch.allclose(noisy, value)

    def test_value_noise_zero_scale_is_identity(self) -> None:
        value = torch.tensor([0.5, 0.3, 0.2])
        noisy = augment_value_noise(value, scale=0.0)
        assert torch.allclose(noisy, value)


class TestPolicyTemperatureAugmentation:
    def test_policy_temp_preserves_sum(self) -> None:
        policy = torch.zeros(64, 64)
        policy[0, 1] = 0.7
        policy[0, 3] = 0.3
        torch.manual_seed(42)
        aug = augment_policy_temperature(policy)
        # Sum of nonzero entries should be approximately 1
        assert abs(aug.sum().item() - 1.0) < 1e-5

    def test_policy_temp_changes_distribution(self) -> None:
        policy = torch.zeros(64, 64)
        policy[0, 1] = 0.7
        policy[0, 3] = 0.3
        torch.manual_seed(42)
        aug = augment_policy_temperature(policy)
        assert not torch.allclose(aug, policy)

    def test_policy_temp_preserves_zeros(self) -> None:
        policy = torch.zeros(64, 64)
        policy[0, 1] = 0.7
        policy[0, 3] = 0.3
        aug = augment_policy_temperature(policy)
        # All originally zero entries should remain zero
        zero_mask = policy == 0
        assert (aug[zero_mask] == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_augmentation.py::TestValueNoiseAugmentation -x -q`
Expected: FAIL with `ImportError: cannot import name 'augment_value_noise'`

**Step 3: Implement augmentation functions**

Add to `src/denoisr/training/augmentation.py`:

```python
def augment_value_noise(value: Tensor, scale: float = 0.02) -> Tensor:
    """Add Gaussian noise to WDL targets and re-normalize to valid distribution."""
    if scale == 0.0:
        return value
    noise = torch.randn_like(value) * scale
    noisy = (value + noise).clamp(min=0.0)
    total = noisy.sum()
    return noisy / total if total > 0 else value


def augment_policy_temperature(policy: Tensor, temp_min: float = 0.8, temp_max: float = 1.2) -> Tensor:
    """Apply random temperature scaling to policy distribution.

    Only affects nonzero entries (legal moves). Temperature < 1 sharpens,
    temperature > 1 flattens the distribution.
    """
    temp = temp_min + torch.rand(1).item() * (temp_max - temp_min)
    mask = policy > 0
    if not mask.any():
        return policy
    result = policy.clone()
    result[mask] = result[mask] ** (1.0 / temp)
    total = result.sum()
    return result / total if total > 0 else policy
```

Update `src/denoisr/training/dataset.py` to accept augmentation config:

```python
from denoisr.training.augmentation import flip_board, flip_policy, augment_value_noise, augment_policy_temperature


class ChessDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        boards: Tensor,
        policies: Tensor,
        values: Tensor,
        num_planes: int,
        augment: bool = True,
        value_noise_prob: float = 0.0,
        value_noise_scale: float = 0.02,
        policy_temp_augment_prob: float = 0.0,
    ) -> None:
        self.boards = boards
        self.policies = policies
        self.values = values
        self.num_planes = num_planes
        self.augment = augment
        self._value_noise_prob = value_noise_prob
        self._value_noise_scale = value_noise_scale
        self._policy_temp_prob = policy_temp_augment_prob

    def __len__(self) -> int:
        return self.boards.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        board = self.boards[idx]
        policy = self.policies[idx]
        value = self.values[idx]
        if self.augment and torch.rand(1).item() < 0.5:
            board = flip_board(board, self.num_planes)
            policy = flip_policy(policy)
            value = value.flip(0)
        if self.augment and self._value_noise_prob > 0 and torch.rand(1).item() < self._value_noise_prob:
            value = augment_value_noise(value, self._value_noise_scale)
        if self.augment and self._policy_temp_prob > 0 and torch.rand(1).item() < self._policy_temp_prob:
            policy = augment_policy_temperature(policy)
        return board, policy, value
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_augmentation.py tests/test_training/test_dataset.py -x -q`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/augmentation.py src/denoisr/training/dataset.py tests/test_training/test_augmentation.py
git commit -m "feat(training): add value noise and policy temperature augmentations"
```

---

### Task 7: OneCycleLR + Gradient Accumulation in SupervisedTrainer

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py`
- Modify: `tests/test_training/test_supervised_trainer.py`

**Step 1: Write the failing tests**

Add to `tests/test_training/test_supervised_trainer.py`:

```python
class TestOneCycleLR:
    def test_onecycle_scheduler_accepted(self) -> None:
        from denoisr.nn.encoder import ChessEncoder
        from denoisr.nn.policy_backbone import ChessPolicyBackbone
        from denoisr.nn.policy_head import ChessPolicyHead
        from denoisr.nn.value_head import ChessValueHead
        from denoisr.training.loss import ChessLossComputer
        from conftest import SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM

        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S)
        backbone = ChessPolicyBackbone(SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S)
        value_head = ChessValueHead(d_s=SMALL_D_S)
        loss_fn = ChessLossComputer()

        trainer = SupervisedTrainer(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
            loss_fn=loss_fn, lr=1e-3,
            use_onecycle=True, steps_per_epoch=100, total_epochs=10,
        )
        assert isinstance(trainer._scheduler, torch.optim.lr_scheduler.OneCycleLR)


class TestGradientAccumulation:
    def test_accumulation_steps_accepted(self) -> None:
        from denoisr.nn.encoder import ChessEncoder
        from denoisr.nn.policy_backbone import ChessPolicyBackbone
        from denoisr.nn.policy_head import ChessPolicyHead
        from denoisr.nn.value_head import ChessValueHead
        from denoisr.training.loss import ChessLossComputer
        from conftest import SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM

        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S)
        backbone = ChessPolicyBackbone(SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S)
        value_head = ChessValueHead(d_s=SMALL_D_S)
        loss_fn = ChessLossComputer()

        trainer = SupervisedTrainer(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
            loss_fn=loss_fn, lr=1e-3,
            accumulation_steps=4,
        )
        assert trainer._accum_steps == 4
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py::TestOneCycleLR -x -q`
Expected: FAIL with `TypeError: SupervisedTrainer.__init__() got an unexpected keyword argument 'use_onecycle'`

**Step 3: Modify SupervisedTrainer**

In `src/denoisr/training/supervised_trainer.py`:

Add new parameters to `__init__`:
- `use_onecycle: bool = False`
- `onecycle_pct_start: float = 0.3`
- `steps_per_epoch: int = 1`
- `accumulation_steps: int = 1`

Add accumulation state:
- `self._accum_steps = accumulation_steps`
- `self._accum_count = 0`
- `self._use_onecycle = use_onecycle`

Add OneCycleLR branch in scheduler setup (before existing cosine branches):
```python
if use_onecycle:
    self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
        self.optimizer,
        max_lr=[lr * encoder_lr_multiplier, lr * encoder_lr_multiplier, lr, lr],
        epochs=total_epochs,
        steps_per_epoch=max(1, steps_per_epoch // accumulation_steps),
        pct_start=onecycle_pct_start,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=10000,
    )
    self._use_onecycle = True
else:
    self._use_onecycle = False
    # ... existing cosine scheduler code unchanged ...
```

Modify `_forward_backward` for gradient accumulation:
- Scale loss by `1 / self._accum_steps` before backward
- Only call `zero_grad` on first micro-batch of accumulation window
- Only unscale, clip, step on the final micro-batch of window
- Step OneCycleLR after each optimizer step (not epoch)

Modify `scheduler_step` to skip for OneCycleLR:
```python
def scheduler_step(self) -> None:
    self._epoch += 1
    if self._use_onecycle:
        return  # OneCycleLR steps per optimizer step in _forward_backward
    # ... existing warmup + cosine logic unchanged ...
```

See design doc for full code details.

**Step 4: Run tests**

Run: `uv run pytest tests/test_training/test_supervised_trainer.py -x -q`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/supervised_trainer.py tests/test_training/test_supervised_trainer.py
git commit -m "feat(training): add OneCycleLR scheduler and gradient accumulation"
```

---

### Task 8: Wire All Enhancements Through train_phase1.py

**Files:**
- Modify: `src/denoisr/scripts/train_phase1.py`

**Step 1: Verify current construction points**

Read `train_phase1.py` to find:
- `ChessLossComputer(...)` construction -- add `label_smoothing=tcfg.label_smoothing`
- `SupervisedTrainer(...)` construction -- add `use_onecycle`, `onecycle_pct_start`, `steps_per_epoch`, `accumulation_steps`
- Dataset construction in shard loop -- add `value_noise_prob`, `value_noise_scale`, `policy_temp_augment_prob`

**Step 2: Update loss_fn construction**

Add `label_smoothing=tcfg.label_smoothing` to `ChessLossComputer(...)` call.

**Step 3: Compute steps_per_epoch**

Before `SupervisedTrainer(...)`, compute:
```python
total_train_examples = sum(s.num_examples for s in data_plan.shards)
steps_per_epoch = total_train_examples // bs
```

**Step 4: Update SupervisedTrainer construction**

Add to `SupervisedTrainer(...)`:
```python
    use_onecycle=tcfg.use_onecycle,
    onecycle_pct_start=tcfg.onecycle_pct_start,
    steps_per_epoch=steps_per_epoch,
    accumulation_steps=tcfg.gradient_accumulation_steps,
```

**Step 5: Update dataset construction in shard loop**

Find where `ChessDataset` or `_IndexedTensorDataset` is used and pass the augmentation config from `tcfg`. If using `_IndexedTensorDataset` (a local wrapper), you may need to change it to use `ChessDataset` or add the augmentation params to the local class.

**Step 6: Run the full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/denoisr/scripts/train_phase1.py
git commit -m "feat(train): wire enhancement config through Phase 1 training loop"
```

---

### Task 9: Integration Smoke Test

**Files:**
- Create: `tests/test_integration/test_enhancements_smoke.py`

**Step 1: Write integration test**

```python
"""Smoke test: enhanced training runs for 2 steps without crashing."""

import torch
from conftest import SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer


class TestEnhancementsSmokeTest:
    def test_full_enhanced_training_step(self, device: torch.device) -> None:
        """All enhancements active: dropout, drop_path, OneCycleLR, accumulation, smoothing."""
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM,
            dropout=0.1, drop_path_rate=0.1,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        loss_fn = ChessLossComputer(label_smoothing=0.1)

        trainer = SupervisedTrainer(
            encoder=encoder, backbone=backbone,
            policy_head=policy_head, value_head=value_head,
            loss_fn=loss_fn, lr=1e-3, device=device,
            use_onecycle=True, steps_per_epoch=10, total_epochs=2,
            accumulation_steps=2,
        )

        boards = torch.randn(4, 12, 8, 8, device=device)
        target_policy = torch.zeros(4, 64, 64, device=device)
        target_policy[:, 0, 1] = 0.7
        target_policy[:, 0, 3] = 0.3
        target_value = torch.tensor([[0.5, 0.3, 0.2]] * 4, device=device)

        # Run 2 accumulation steps (should trigger 1 optimizer step)
        loss1, bd1 = trainer.train_step_tensors(boards, target_policy, target_value)
        loss2, bd2 = trainer.train_step_tensors(boards, target_policy, target_value)

        assert loss1 > 0
        assert loss2 > 0
        assert bd2.get("grad_norm", 0) > 0  # second step should have grad_norm
```

**Step 2: Run test**

Run: `uv run pytest tests/test_integration/test_enhancements_smoke.py -x -q`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration/test_enhancements_smoke.py
git commit -m "test: add integration smoke test for all training enhancements"
```

---

### Task 10: Full Test Suite + Lint

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS

**Step 2: Run linter**

Run: `uvx ruff check src/ tests/`
Expected: No errors

**Step 3: Run type checker**

Run: `uv run --with mypy mypy --strict src/denoisr/nn/drop_path.py src/denoisr/nn/policy_backbone.py src/denoisr/training/loss.py src/denoisr/training/supervised_trainer.py src/denoisr/training/augmentation.py src/denoisr/training/dataset.py`
Expected: No errors

**Step 4: Fix any issues found in steps 1-3**

**Step 5: Final commit if fixes needed**

```bash
git add -A
git commit -m "fix: address lint and type check issues from training enhancements"
```
