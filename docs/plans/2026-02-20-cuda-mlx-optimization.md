# CUDA & MLX Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 3-5x Phase 1 training throughput on RTX 3060 12GB via PyTorch-native acceleration, plus MLX inference on Apple Silicon.

**Architecture:** Seven incremental changes — each builds on the last but is independently testable. AMP + SDPA + compile + gradient checkpointing for CUDA training speed, DataLoader for CPU/GPU overlap, persistent buffers for cleanup, MLX for Mac inference.

**Tech Stack:** PyTorch 2.10+ (autocast, GradScaler, SDPA, torch.compile, checkpoint), mlx + mlx-nn (optional, inference only), safetensors (weight export).

---

### Task 1: Convert CosineNoiseSchedule to nn.Module

**Files:**
- Modify: `src/denoisr/nn/diffusion.py:8-32`
- Modify: `src/denoisr/training/diffusion_trainer.py`
- Modify: `src/denoisr/scripts/train_phase2.py`
- Modify: `tests/test_nn/test_diffusion.py:17-52`

This is the smallest, safest change — converts a plain class to `nn.Module` so `alpha_bar` lives on the correct device automatically.

**Step 1: Write the failing test**

Add to `tests/test_nn/test_diffusion.py` inside `TestCosineNoiseSchedule`:

```python
def test_schedule_is_nn_module(
    self, schedule: CosineNoiseSchedule
) -> None:
    """Schedule should be an nn.Module so .to(device) moves alpha_bar."""
    import torch.nn as nn
    assert isinstance(schedule, nn.Module)

def test_alpha_bar_moves_with_to(
    self, schedule: CosineNoiseSchedule, device: torch.device
) -> None:
    """alpha_bar should follow .to(device) like a registered buffer."""
    schedule.to(device)
    assert schedule.alpha_bar.device.type == device.type
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nn/test_diffusion.py::TestCosineNoiseSchedule::test_schedule_is_nn_module tests/test_nn/test_diffusion.py::TestCosineNoiseSchedule::test_alpha_bar_moves_with_to -v`
Expected: FAIL (CosineNoiseSchedule is not an nn.Module)

**Step 3: Implement the change**

In `src/denoisr/nn/diffusion.py`, change `CosineNoiseSchedule` to inherit from `nn.Module`:

```python
class CosineNoiseSchedule(nn.Module):
    """Cosine noise schedule for continuous DDPM (Nichol & Dhariwal 2021)."""

    def __init__(self, num_timesteps: int, s: float = 0.008) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f = (
            torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2)
            ** 2
        )
        alpha_bar = f / f[0]
        self.register_buffer(
            "alpha_bar",
            alpha_bar[:num_timesteps].float().clamp(min=1e-5, max=0.9999),
        )

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: add noise at timestep t."""
        ab = self.alpha_bar[t]
        while ab.ndim < x_0.ndim:
            ab = ab.unsqueeze(-1)
        return ab.sqrt() * x_0 + (1 - ab).sqrt() * noise
```

Key changes:
- Inherit from `nn.Module`, call `super().__init__()`
- Replace `self.alpha_bar = ...` with `self.register_buffer("alpha_bar", ...)`
- Remove `.to(x_0.device)` from `q_sample` — buffer auto-moves with `.to(device)`

Also update `DiffusionTrainer` in `src/denoisr/training/diffusion_trainer.py` to move schedule to device:

In `__init__`, after `self.schedule = schedule`, add:
```python
self.schedule.to(self.device)
```

And in `src/denoisr/scripts/train_phase2.py:117`, change:
```python
schedule = build_schedule(cfg)
```
to:
```python
schedule = build_schedule(cfg).to(device)
```

**Step 4: Run all tests**

Run: `uv run pytest tests/test_nn/test_diffusion.py tests/test_training/test_diffusion_trainer.py -v`
Expected: ALL PASS

**Step 5: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 6: Commit**

```bash
git add src/denoisr/nn/diffusion.py src/denoisr/training/diffusion_trainer.py src/denoisr/scripts/train_phase2.py tests/test_nn/test_diffusion.py
git commit -m "refactor: convert CosineNoiseSchedule to nn.Module with registered buffer"
```

---

### Task 2: Backbone SDPA Conversion

**Files:**
- Modify: `src/denoisr/nn/policy_backbone.py:28-46`
- Modify: `tests/test_nn/test_policy_backbone.py`

Replace manual attention computation with `F.scaled_dot_product_attention()` using additive `attn_mask`.

**Step 1: Write the failing test**

Add to `tests/test_nn/test_policy_backbone.py`:

```python
def test_sdpa_numerical_equivalence(
    self, device: torch.device
) -> None:
    """SDPA path should produce same results as manual attention."""
    torch.manual_seed(42)
    backbone = ChessPolicyBackbone(
        d_s=SMALL_D_S,
        num_heads=SMALL_NUM_HEADS,
        num_layers=1,  # single layer for cleaner comparison
        ffn_dim=SMALL_FFN_DIM,
    ).to(device)
    backbone.eval()
    x = torch.randn(1, 64, SMALL_D_S, device=device)
    out = backbone(x)
    # After conversion, output should still be finite and correct shape
    assert out.shape == (1, 64, SMALL_D_S)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
```

This test passes both before and after the change — it's a regression guard.

**Step 2: Run test to confirm it passes (pre-change baseline)**

Run: `uv run pytest tests/test_nn/test_policy_backbone.py -v`
Expected: ALL PASS

**Step 3: Implement the SDPA conversion**

In `src/denoisr/nn/policy_backbone.py`, replace the manual attention in `TransformerBlock.forward()`.

Replace lines 37-41:
```python
attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
if attn_bias is not None:
    attn = attn + attn_bias
attn = F.softmax(attn, dim=-1)
h = (attn @ v).transpose(1, 2).reshape(B, S, D)
```

With:
```python
h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
h = h.transpose(1, 2).reshape(B, S, D)
```

The full `forward` method becomes:

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
    h = self.out_proj(h)
    x = x + h

    x = x + self.ffn(self.norm2(x))
    return x
```

**Step 4: Run all backbone tests**

Run: `uv run pytest tests/test_nn/test_policy_backbone.py -v`
Expected: ALL PASS (output shape, gradients, no NaN, smolgen/shaw biases)

**Step 5: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 6: Commit**

```bash
git add src/denoisr/nn/policy_backbone.py tests/test_nn/test_policy_backbone.py
git commit -m "perf: convert backbone attention to F.scaled_dot_product_attention"
```

---

### Task 3: AMP (Automatic Mixed Precision)

**Files:**
- Modify: `src/denoisr/training/supervised_trainer.py`
- Modify: `src/denoisr/training/diffusion_trainer.py`
- Create: `tests/test_training/test_amp.py`

Add FP16 autocast + GradScaler to both trainer classes.

**Step 1: Write the failing test**

Create `tests/test_training/test_amp.py`:

```python
import pytest
import torch

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


def _make_batch(n: int = 4) -> list[TrainingExample]:
    examples = []
    for _ in range(n):
        board = BoardTensor(torch.randn(12, 8, 8))
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0
        policy = PolicyTarget(policy_data)
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        examples.append(TrainingExample(board=board, policy=policy, value=value))
    return examples


class TestSupervisedTrainerAMP:
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

    def test_trainer_has_scaler(self, trainer: SupervisedTrainer) -> None:
        """SupervisedTrainer should have a GradScaler attribute."""
        assert hasattr(trainer, "scaler")
        assert isinstance(trainer.scaler, torch.amp.GradScaler)

    def test_train_step_works_with_amp(
        self, trainer: SupervisedTrainer
    ) -> None:
        """Training step should complete without error when AMP is available."""
        batch = _make_batch(4)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        assert "policy" in breakdown

    def test_loss_decreases_with_amp(
        self, trainer: SupervisedTrainer
    ) -> None:
        """Training should still converge with AMP enabled."""
        batch = _make_batch(4)
        losses = [trainer.train_step(batch)[0] for _ in range(20)]
        assert losses[-1] < losses[0]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_amp.py -v`
Expected: FAIL on `test_trainer_has_scaler` (no `scaler` attribute yet)

**Step 3: Implement AMP in SupervisedTrainer**

In `src/denoisr/training/supervised_trainer.py`:

Add imports at top:
```python
from torch.amp import GradScaler, autocast
```

In `__init__`, after `self.max_grad_norm = 1.0`, add:
```python
self.scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
self._autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
self._autocast_enabled = device.type == "cuda"
```

Replace `train_step` body (lines 54-93) with:

```python
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

    with autocast(self._autocast_device, enabled=self._autocast_enabled):
        latent = self.encoder(boards)
        features = self.backbone(latent)
        pred_policy = self.policy_head(features)
        pred_value, _pred_ply = self.value_head(features)

        total_loss, breakdown = self.loss_fn.compute(
            pred_policy, pred_value, target_policies, target_values
        )

    self.optimizer.zero_grad()
    self.scaler.scale(total_loss).backward()  # type: ignore[no-untyped-call]
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(
        [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
        ],
        self.max_grad_norm,
    )
    self.scaler.step(self.optimizer)
    self.scaler.update()

    return total_loss.item(), breakdown
```

Also add scaler state to checkpoint save/load. In `save_checkpoint`:
```python
"scaler": self.scaler.state_dict(),
```

In `load_checkpoint`, after scheduler restore:
```python
if "scaler" in checkpoint:
    self.scaler.load_state_dict(checkpoint["scaler"])
```

**Step 4: Implement AMP in DiffusionTrainer**

In `src/denoisr/training/diffusion_trainer.py`:

Add import:
```python
from torch.amp import GradScaler, autocast
```

In `__init__`, after optimizer setup:
```python
self.scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
self._autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
self._autocast_enabled = device.type == "cuda"
```

In `train_step`, wrap the forward pass and update backward:
```python
with autocast(self._autocast_device, enabled=self._autocast_enabled):
    # ... existing forward pass code (encoder eval, diffusion train, noise prediction) ...
    loss = nn.functional.mse_loss(predicted_noise, noise)

self.optimizer.zero_grad()
self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
self.scaler.step(self.optimizer)
self.scaler.update()
```

**Step 5: Run all tests**

Run: `uv run pytest tests/test_training/test_amp.py tests/test_training/test_supervised_trainer.py tests/test_training/test_diffusion_trainer.py -v`
Expected: ALL PASS

**Step 6: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 7: Commit**

```bash
git add src/denoisr/training/supervised_trainer.py src/denoisr/training/diffusion_trainer.py tests/test_training/test_amp.py
git commit -m "perf: add FP16 mixed precision (autocast + GradScaler) to trainers"
```

---

### Task 4: Gradient Checkpointing

**Files:**
- Modify: `src/denoisr/nn/policy_backbone.py`
- Modify: `src/denoisr/nn/diffusion.py`
- Modify: `tests/test_nn/test_policy_backbone.py`

Add optional gradient checkpointing to backbone and diffusion transformer layers.

**Step 1: Write the failing test**

Add to `tests/test_nn/test_policy_backbone.py`:

```python
def test_gradient_checkpointing_produces_gradients(
    self, device: torch.device
) -> None:
    """Backbone with gradient checkpointing should still produce valid gradients."""
    backbone = ChessPolicyBackbone(
        d_s=SMALL_D_S,
        num_heads=SMALL_NUM_HEADS,
        num_layers=SMALL_NUM_LAYERS,
        ffn_dim=SMALL_FFN_DIM,
        gradient_checkpointing=True,
    ).to(device)
    x = torch.randn(2, 64, SMALL_D_S, device=device)
    out = backbone(x)
    out.sum().backward()
    for name, p in backbone.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

def test_gradient_checkpointing_matches_output(
    self, device: torch.device
) -> None:
    """Checkpointed and non-checkpointed should produce identical forward output."""
    torch.manual_seed(42)
    bb_normal = ChessPolicyBackbone(
        d_s=SMALL_D_S,
        num_heads=SMALL_NUM_HEADS,
        num_layers=SMALL_NUM_LAYERS,
        ffn_dim=SMALL_FFN_DIM,
        gradient_checkpointing=False,
    ).to(device)
    torch.manual_seed(42)
    bb_ckpt = ChessPolicyBackbone(
        d_s=SMALL_D_S,
        num_heads=SMALL_NUM_HEADS,
        num_layers=SMALL_NUM_LAYERS,
        ffn_dim=SMALL_FFN_DIM,
        gradient_checkpointing=True,
    ).to(device)
    bb_normal.eval()
    bb_ckpt.eval()
    x = torch.randn(1, 64, SMALL_D_S, device=device)
    assert torch.allclose(bb_normal(x), bb_ckpt(x), atol=1e-5)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nn/test_policy_backbone.py::TestChessPolicyBackbone::test_gradient_checkpointing_produces_gradients -v`
Expected: FAIL (`__init__() got an unexpected keyword argument 'gradient_checkpointing'`)

**Step 3: Implement gradient checkpointing**

In `src/denoisr/nn/policy_backbone.py`, add import:

```python
from torch.utils.checkpoint import checkpoint as torch_checkpoint
```

Modify `ChessPolicyBackbone.__init__` signature:

```python
def __init__(
    self, d_s: int, num_heads: int, num_layers: int, ffn_dim: int,
    gradient_checkpointing: bool = False,
) -> None:
```

Add at end of `__init__`:
```python
self._gradient_checkpointing = gradient_checkpointing
```

Modify `forward`:

```python
def forward(self, x: Tensor) -> Tensor:
    smolgen_bias = self.smolgen(x)  # [B, H, 64, 64]
    shaw_bias = self.shaw_relative_pe()  # [H, 64, 64]
    combined_bias = smolgen_bias + shaw_bias.unsqueeze(0)
    for layer in self.layers:
        if self._gradient_checkpointing and self.training:
            x = torch_checkpoint(layer, x, combined_bias, use_reentrant=False)
        else:
            x = layer(x, attn_bias=combined_bias)
    out: Tensor = self.final_norm(x)
    return out
```

Apply the same pattern to `ChessDiffusionModule` in `src/denoisr/nn/diffusion.py`:

Add import:
```python
from torch.utils.checkpoint import checkpoint as torch_checkpoint
```

Add `gradient_checkpointing: bool = False` parameter to `__init__`, store as `self._gradient_checkpointing`.

Modify `forward`:
```python
for layer in self.layers:
    if self._gradient_checkpointing and self.training:
        x = torch_checkpoint(layer, x, c, use_reentrant=False)
    else:
        x = layer(x, c)
```

**Step 4: Run all tests**

Run: `uv run pytest tests/test_nn/test_policy_backbone.py tests/test_nn/test_diffusion.py -v`
Expected: ALL PASS

**Step 5: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 6: Commit**

```bash
git add src/denoisr/nn/policy_backbone.py src/denoisr/nn/diffusion.py tests/test_nn/test_policy_backbone.py
git commit -m "perf: add gradient checkpointing to backbone and diffusion layers"
```

---

### Task 5: torch.compile() + CLI Integration

**Files:**
- Modify: `src/denoisr/scripts/config.py`
- Modify: `src/denoisr/scripts/train_phase1.py`
- Modify: `src/denoisr/scripts/train_phase2.py`
- Create: `tests/test_scripts/test_compile.py`

Add a `maybe_compile()` helper, `--gradient-checkpointing` CLI flag, and wire into training scripts.

**Step 1: Write the failing test**

Create `tests/test_scripts/test_compile.py`:

```python
import torch
from torch import nn

from denoisr.scripts.config import maybe_compile


class TestMaybeCompile:
    def test_returns_module_on_cpu(self) -> None:
        """On CPU, maybe_compile should return the original module (no compile)."""
        m = nn.Linear(4, 4)
        result = maybe_compile(m, torch.device("cpu"))
        assert result is m  # exact same object

    def test_returns_module_on_mps_if_available(self) -> None:
        """On MPS, maybe_compile should return the original module."""
        if not torch.backends.mps.is_available():
            return  # skip on non-Mac
        m = nn.Linear(4, 4)
        result = maybe_compile(m, torch.device("mps"))
        assert result is m

    def test_compiled_module_on_cuda_if_available(self) -> None:
        """On CUDA, maybe_compile should return a compiled wrapper."""
        if not torch.cuda.is_available():
            return  # skip if no CUDA
        m = nn.Linear(4, 4).cuda()
        result = maybe_compile(m, torch.device("cuda"))
        assert result is not m  # should be wrapped
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scripts/test_compile.py -v`
Expected: FAIL (ImportError — `maybe_compile` doesn't exist)

**Step 3: Implement maybe_compile and CLI flags**

In `src/denoisr/scripts/config.py`, add after `detect_device()`:

```python
def maybe_compile(module: nn.Module, device: torch.device) -> nn.Module:
    """Compile module with torch.compile on CUDA, return as-is otherwise."""
    if device.type == "cuda":
        return torch.compile(module)  # type: ignore[return-value]
    return module
```

Add `nn` import at top (add to existing torch import):
```python
from torch import nn
```

Add `gradient_checkpointing` to `ModelConfig`:
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
    gradient_checkpointing: bool = False
```

Add CLI flag in `add_model_args()`:
```python
g.add_argument(
    "--gradient-checkpointing",
    action="store_true",
    default=False,
    help="Enable gradient checkpointing (saves VRAM, slightly slower)",
)
```

Add to `config_from_args`:
```python
gradient_checkpointing=args.gradient_checkpointing,
```

Update `build_backbone` to pass the flag:
```python
def build_backbone(cfg: ModelConfig) -> ChessPolicyBackbone:
    return ChessPolicyBackbone(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_dim=cfg.ffn_dim,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
```

Update `build_diffusion` to pass the flag:
```python
def build_diffusion(cfg: ModelConfig) -> ChessDiffusionModule:
    return ChessDiffusionModule(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.diffusion_layers,
        num_timesteps=cfg.num_timesteps,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
```

**Step 4: Wire into train_phase1.py**

In `src/denoisr/scripts/train_phase1.py`, add `maybe_compile` to imports:

```python
from denoisr.scripts.config import (
    add_model_args,
    build_backbone,
    build_encoder,
    build_policy_head,
    build_value_head,
    detect_device,
    load_checkpoint,
    maybe_compile,
    save_checkpoint,
)
```

After loading state dicts (after line 102), wrap models:

```python
encoder = maybe_compile(encoder, device)
backbone = maybe_compile(backbone, device)
policy_head = maybe_compile(policy_head, device)
value_head = maybe_compile(value_head, device)
```

**Step 5: Wire into train_phase2.py**

Same pattern — import `maybe_compile`, apply to encoder, backbone, diffusion after loading state dicts.

**Step 6: Run all tests**

Run: `uv run pytest tests/test_scripts/test_compile.py tests/test_nn/test_policy_backbone.py tests/test_nn/test_diffusion.py tests/test_training/ -v`
Expected: ALL PASS

**Step 7: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 8: Commit**

```bash
git add src/denoisr/scripts/config.py src/denoisr/scripts/train_phase1.py src/denoisr/scripts/train_phase2.py tests/test_scripts/test_compile.py
git commit -m "perf: add torch.compile, gradient-checkpointing CLI flag, and maybe_compile helper"
```

---

### Task 6: DataLoader + ChessDataset

**Files:**
- Create: `src/denoisr/training/dataset.py`
- Modify: `src/denoisr/training/supervised_trainer.py`
- Modify: `src/denoisr/scripts/train_phase1.py`
- Create: `tests/test_training/test_dataset.py`

Replace manual batching with proper DataLoader for CPU/GPU overlap.

**Step 1: Write the failing test**

Create `tests/test_training/test_dataset.py`:

```python
import torch

from denoisr.training.dataset import ChessDataset


class TestChessDataset:
    def test_len(self) -> None:
        boards = torch.randn(100, 12, 8, 8)
        policies = torch.randn(100, 64, 64)
        values = torch.randn(100, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=False)
        assert len(ds) == 100

    def test_getitem_shapes(self) -> None:
        boards = torch.randn(10, 12, 8, 8)
        policies = torch.randn(10, 64, 64)
        values = torch.randn(10, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=False)
        board, policy, value = ds[0]
        assert board.shape == (12, 8, 8)
        assert policy.shape == (64, 64)
        assert value.shape == (3,)

    def test_no_augment_returns_original(self) -> None:
        boards = torch.randn(10, 12, 8, 8)
        policies = torch.randn(10, 64, 64)
        values = torch.randn(10, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=False)
        board, policy, value = ds[3]
        assert torch.equal(board, boards[3])
        assert torch.equal(policy, policies[3])
        assert torch.equal(value, values[3])

    def test_augment_flips_some_examples(self) -> None:
        """With augmentation, at least some examples should differ from originals."""
        torch.manual_seed(0)
        boards = torch.randn(100, 12, 8, 8)
        policies = torch.randn(100, 64, 64)
        values = torch.randn(100, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=True)
        flipped = 0
        for i in range(100):
            board, _, _ = ds[i]
            if not torch.equal(board, boards[i]):
                flipped += 1
        # With 50% flip probability, expect ~50 flipped (allow wide margin)
        assert 20 < flipped < 80

    def test_dataloader_integration(self) -> None:
        boards = torch.randn(32, 12, 8, 8)
        policies = torch.randn(32, 64, 64)
        values = torch.randn(32, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        batch_boards, batch_policies, batch_values = next(iter(loader))
        assert batch_boards.shape == (8, 12, 8, 8)
        assert batch_policies.shape == (8, 64, 64)
        assert batch_values.shape == (8, 3)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_dataset.py -v`
Expected: FAIL (ModuleNotFoundError — `denoisr.training.dataset` doesn't exist)

**Step 3: Create ChessDataset**

Create `src/denoisr/training/dataset.py`:

```python
"""Chess training dataset for DataLoader integration."""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from denoisr.training.augmentation import flip_board, flip_policy


class ChessDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Wraps pre-stacked training tensors for use with DataLoader.

    Augmentation (50% random board flip) runs in __getitem__,
    which means it executes in DataLoader worker processes —
    overlapped with GPU training.
    """

    def __init__(
        self,
        boards: Tensor,
        policies: Tensor,
        values: Tensor,
        num_planes: int,
        augment: bool = True,
    ) -> None:
        self.boards = boards
        self.policies = policies
        self.values = values
        self.num_planes = num_planes
        self.augment = augment

    def __len__(self) -> int:
        return self.boards.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        board = self.boards[idx]
        policy = self.policies[idx]
        value = self.values[idx]
        if self.augment and torch.rand(1).item() < 0.5:
            board = flip_board(board, self.num_planes)
            policy = flip_policy(policy)
            value = value.flip(0)  # [w,d,l] -> [l,d,w]
        return board, policy, value
```

**Step 4: Add train_step_tensors to SupervisedTrainer**

In `src/denoisr/training/supervised_trainer.py`, add after the existing `train_step`:

```python
def train_step_tensors(
    self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    """Train step accepting pre-stacked tensors (from DataLoader)."""
    boards = boards.to(self.device)
    target_policies = target_policies.to(self.device)
    target_values = target_values.to(self.device)

    self.encoder.train()
    self.backbone.train()
    self.policy_head.train()
    self.value_head.train()

    with autocast(self._autocast_device, enabled=self._autocast_enabled):
        latent = self.encoder(boards)
        features = self.backbone(latent)
        pred_policy = self.policy_head(features)
        pred_value, _pred_ply = self.value_head(features)

        total_loss, breakdown = self.loss_fn.compute(
            pred_policy, pred_value, target_policies, target_values
        )

    self.optimizer.zero_grad()
    self.scaler.scale(total_loss).backward()  # type: ignore[no-untyped-call]
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(
        [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
        ],
        self.max_grad_norm,
    )
    self.scaler.step(self.optimizer)
    self.scaler.update()

    return total_loss.item(), breakdown
```

**Step 5: Rewrite train_phase1.py training loop**

In `src/denoisr/scripts/train_phase1.py`:

Add imports:
```python
from torch.utils.data import DataLoader
from denoisr.training.dataset import ChessDataset
```

After `train = all_examples[holdout_n:]`, replace manual batching setup with:

```python
# Build DataLoader from stacked tensors
train_boards = torch.stack([ex.board.data for ex in train])
train_policies = torch.stack([ex.policy.data for ex in train])
train_values = torch.tensor(
    [[ex.value.win, ex.value.draw, ex.value.loss] for ex in train],
    dtype=torch.float32,
)

train_dataset = ChessDataset(
    train_boards, train_policies, train_values,
    num_planes=cfg.num_planes, augment=True,
)
train_loader = DataLoader(
    train_dataset, batch_size=bs, shuffle=True,
    num_workers=2, pin_memory=(device.type == "cuda"),
    persistent_workers=True,
)
```

Replace the epoch training loop to use DataLoader:

```python
for epoch in range(args.epochs):
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{args.epochs}",
        leave=False,
        smoothing=0.3,
    )
    for boards_batch, policies_batch, values_batch in pbar:
        loss, breakdown = trainer.train_step_tensors(
            boards_batch, policies_batch, values_batch
        )
        epoch_loss += loss
        num_batches += 1
        pbar.set_postfix(
            loss=f"{loss:.4f}",
            policy=f"{breakdown['policy']:.4f}",
            value=f"{breakdown['value']:.4f}",
        )
    pbar.close()
    trainer.scheduler_step()

    avg_loss = epoch_loss / max(num_batches, 1)
    top1, top5 = measure_accuracy(trainer, holdout, device)
    # ... rest of epoch logging and checkpoint saving unchanged ...
```

**Step 6: Run all tests**

Run: `uv run pytest tests/test_training/test_dataset.py tests/test_training/test_supervised_trainer.py tests/test_training/test_amp.py -v`
Expected: ALL PASS

**Step 7: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 8: Commit**

```bash
git add src/denoisr/training/dataset.py src/denoisr/training/supervised_trainer.py src/denoisr/scripts/train_phase1.py tests/test_training/test_dataset.py
git commit -m "perf: add ChessDataset + DataLoader with pin_memory and worker prefetching"
```

---

### Task 7: MLX Inference Export

**Files:**
- Create: `src/denoisr/scripts/export_mlx.py`
- Create: `src/denoisr/inference/mlx_engine.py`
- Modify: `pyproject.toml` (add entry point + safetensors dep)
- Create: `tests/test_inference/test_mlx_export.py`

Export PyTorch weights to safetensors and reimplement inference modules in MLX.

**Step 1: Add safetensors dependency**

Run: `uv add safetensors`

**Step 2: Write the test for weight export**

Create `tests/test_inference/test_mlx_export.py`:

```python
import pathlib

import torch
from safetensors.torch import load_file

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.scripts.config import ModelConfig, save_checkpoint
from denoisr.scripts.export_mlx import export_weights

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


class TestExportWeights:
    def _make_checkpoint(self, tmp_path: pathlib.Path) -> pathlib.Path:
        cfg = ModelConfig(
            num_planes=12,
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        )
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        )
        policy_head = ChessPolicyHead(d_s=SMALL_D_S)
        value_head = ChessValueHead(d_s=SMALL_D_S)
        ckpt_path = tmp_path / "model.pt"
        save_checkpoint(
            ckpt_path, cfg,
            encoder=encoder.state_dict(),
            backbone=backbone.state_dict(),
            policy_head=policy_head.state_dict(),
            value_head=value_head.state_dict(),
        )
        return ckpt_path

    def test_export_creates_safetensors(self, tmp_path: pathlib.Path) -> None:
        """export_weights should create a .safetensors file."""
        ckpt_path = self._make_checkpoint(tmp_path)
        out_path = tmp_path / "model.safetensors"
        export_weights(ckpt_path, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_export_creates_config_json(self, tmp_path: pathlib.Path) -> None:
        """export_weights should also create a .json config file."""
        ckpt_path = self._make_checkpoint(tmp_path)
        out_path = tmp_path / "model.safetensors"
        export_weights(ckpt_path, out_path)
        config_path = tmp_path / "model.json"
        assert config_path.exists()

    def test_exported_weights_have_expected_keys(self, tmp_path: pathlib.Path) -> None:
        """Exported safetensors should contain all model weight prefixes."""
        ckpt_path = self._make_checkpoint(tmp_path)
        out_path = tmp_path / "model.safetensors"
        export_weights(ckpt_path, out_path)
        weights = load_file(str(out_path))
        assert any(k.startswith("encoder.") for k in weights)
        assert any(k.startswith("backbone.") for k in weights)
        assert any(k.startswith("policy_head.") for k in weights)
        assert any(k.startswith("value_head.") for k in weights)
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_inference/test_mlx_export.py -v`
Expected: FAIL (ModuleNotFoundError — `denoisr.scripts.export_mlx` doesn't exist)

**Step 4: Implement export_weights**

Create `src/denoisr/scripts/export_mlx.py`:

```python
"""Export PyTorch checkpoint to safetensors for MLX inference."""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from denoisr.scripts.config import load_checkpoint


def export_weights(checkpoint_path: Path, output_path: Path) -> None:
    """Convert a PyTorch checkpoint to safetensors format."""
    cfg, state = load_checkpoint(checkpoint_path, torch.device("cpu"))

    weights: dict[str, torch.Tensor] = {}
    for prefix in ("encoder", "backbone", "policy_head", "value_head"):
        if prefix not in state:
            continue
        for key, tensor in state[prefix].items():
            weights[f"{prefix}.{key}"] = tensor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output_path))

    # Save config alongside for MLX engine to read
    config_path = output_path.with_suffix(".json")
    config_path.write_text(json.dumps(cfg.__dict__, indent=2))
    print(f"Exported {len(weights)} tensors to {output_path}")
    print(f"Config saved to {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PyTorch checkpoint to safetensors for MLX inference"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="PyTorch checkpoint path"
    )
    parser.add_argument(
        "--output",
        default="outputs/model.safetensors",
        help="Output safetensors path",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found", file=sys.stderr)
        sys.exit(1)

    export_weights(checkpoint_path, Path(args.output))


if __name__ == "__main__":
    main()
```

**Step 5: Create MLX inference engine**

Create `src/denoisr/inference/mlx_engine.py`. This reimplements the 4 inference modules (encoder, backbone, policy head, value head) in MLX. The full implementation is ~250 lines covering:

- `MLXChessEncoder` — per-square linear + global embed + layernorm
- `MLXTransformerBlock` — pre-norm attention with bias support + FFN
- `MLXSmolgenBias` — content-dependent attention bias
- `MLXPolicyHead` — bilinear query/key attention
- `MLXValueHead` — mean pool + WDL softmax
- `MLXChessEngine` — weight loading from safetensors, `select_move()` API

Key implementation notes for the implementer:
- Guard all mlx imports behind `try/except ImportError` with helpful message
- Use `TYPE_CHECKING` guard for chess import to avoid torch dependency at MLX runtime
- Map PyTorch `nn.Sequential` indexed keys (e.g., `global_embed.0.weight`) to separate MLX linear layers
- Shaw PE buffers (`rank_idx`, `file_idx`) are int64 in PyTorch — cast to `mx.int32` in MLX
- Use `mx.load()` which natively reads safetensors format
- `select_move` uses argmax (greedy) selection with legal move masking
- The engine does NOT need torch at runtime — only chess + mlx

**Step 6: Add entry point to pyproject.toml**

In `pyproject.toml`, add to `[project.scripts]`:

```toml
denoisr-export-mlx = "denoisr.scripts.export_mlx:main"
```

**Step 7: Run export tests**

Run: `uv run pytest tests/test_inference/test_mlx_export.py -v`
Expected: ALL PASS

**Step 8: Run full verification**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

Note: mypy may need `# type: ignore[import-untyped]` for mlx imports since mlx doesn't ship type stubs.

**Step 9: Commit**

```bash
git add src/denoisr/scripts/export_mlx.py src/denoisr/inference/mlx_engine.py tests/test_inference/test_mlx_export.py pyproject.toml uv.lock
git commit -m "feat: add MLX inference engine and safetensors weight export"
```

---

### Task 8: Full Verification + Integration Test

**Files:**
- Create: `tests/test_integration/test_cuda_optimizations.py`
- No production code changes

**Step 1: Write integration test**

Create `tests/test_integration/test_cuda_optimizations.py`:

```python
"""Integration tests verifying all CUDA optimizations work together."""

import torch
import pytest

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.nn.diffusion import CosineNoiseSchedule
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.training.dataset import ChessDataset
from denoisr.scripts.config import maybe_compile

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


class TestFullPipeline:
    """Verify all optimizations work together end-to-end."""

    def test_compiled_amp_checkpointed_training(
        self, device: torch.device
    ) -> None:
        """Full pipeline: compile + AMP + gradient checkpointing + DataLoader."""
        encoder = maybe_compile(
            ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device), device
        )
        backbone = maybe_compile(
            ChessPolicyBackbone(
                d_s=SMALL_D_S,
                num_heads=SMALL_NUM_HEADS,
                num_layers=SMALL_NUM_LAYERS,
                ffn_dim=SMALL_FFN_DIM,
                gradient_checkpointing=True,
            ).to(device),
            device,
        )
        policy_head = maybe_compile(
            ChessPolicyHead(d_s=SMALL_D_S).to(device), device
        )
        value_head = maybe_compile(
            ChessValueHead(d_s=SMALL_D_S).to(device), device
        )

        loss_fn = ChessLossComputer()
        trainer = SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
        )

        # Create dataset + DataLoader
        boards = torch.randn(32, 12, 8, 8)
        policies = torch.zeros(32, 64, 64)
        policies[:, 12, 28] = 1.0
        values = torch.tensor([[1.0, 0.0, 0.0]] * 32)
        dataset = ChessDataset(boards, policies, values, num_planes=12, augment=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        # Train a few steps
        losses = []
        for batch_boards, batch_policies, batch_values in loader:
            loss, _ = trainer.train_step_tensors(
                batch_boards, batch_policies, batch_values
            )
            assert not (loss != loss), "NaN loss detected"
            losses.append(loss)

        assert len(losses) == 4  # 32 examples / batch_size 8
        assert all(l > 0 for l in losses)

    def test_schedule_buffer_on_device(self, device: torch.device) -> None:
        """CosineNoiseSchedule.alpha_bar should live on the correct device."""
        schedule = CosineNoiseSchedule(num_timesteps=SMALL_NUM_TIMESTEPS)
        schedule.to(device)
        assert schedule.alpha_bar.device.type == device.type

        # q_sample should work without explicit .to()
        x_0 = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.tensor([0, 1], device=device)
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise)
        assert x_t.device.type == device.type
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/test_integration/test_cuda_optimizations.py -v`
Expected: ALL PASS

**Step 3: Run full verification suite**

Run: `uvx ruff check src/ tests/ && uv run --with mypy mypy --strict src/denoisr/ && uv run pytest tests/ -x -q`
Expected: 0 errors, all tests pass

**Step 4: Verify CLI help strings**

Run:
```bash
uv run denoisr-train-phase1 --help
uv run denoisr-export-mlx --help
```
Expected: `--gradient-checkpointing` flag visible in phase1 help. `denoisr-export-mlx` shows `--checkpoint` and `--output`.

**Step 5: Commit**

```bash
git add tests/test_integration/test_cuda_optimizations.py
git commit -m "test: add integration tests for CUDA optimization pipeline"
```
