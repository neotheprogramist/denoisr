# CUDA & MLX Optimization Design

**Goal:** 3-5x Phase 1 training throughput on RTX 3060 12GB (CUDA primary), plus MLX inference on Apple Silicon (secondary).

**Approach:** PyTorch-native acceleration — no external dependencies beyond `mlx` for inference export. Seven changes ordered by impact.

**Target hardware:**
- Primary: NVIDIA GeForce RTX 3060, 12GB VRAM, Ampere architecture
- Secondary: Apple Silicon (M-series) for inference via MLX

---

## Change Summary

| # | Change | Impact | Memory Effect |
|---|--------|--------|---------------|
| 1 | AMP (FP16 autocast + GradScaler) | ~1.7x throughput | -50% VRAM |
| 2 | Backbone SDPA conversion | ~1.3x (memory-efficient attention) | -30% attention VRAM |
| 3 | `torch.compile()` on forward pass | ~1.2-1.4x | neutral |
| 4 | Gradient checkpointing on backbone | enables larger batch | -40% activation VRAM |
| 5 | Proper DataLoader + pin_memory | eliminates CPU stalls | neutral |
| 6 | Persistent device buffers | removes per-batch copies | neutral |
| 7 | MLX inference export | Mac-native inference | N/A |

---

## 1. AMP (Automatic Mixed Precision)

**Where:** `SupervisedTrainer`, `DiffusionTrainer`, `measure_accuracy`

FP16 autocast on forward pass + GradScaler for stable backward. RTX 3060 has dedicated FP16 Tensor Cores — this halves memory and roughly doubles arithmetic throughput.

**Design:**
- FP16 (not BF16) — RTX 3060 lacks dedicated BF16 Tensor Cores
- autocast wraps forward + loss only; backward runs in same dtypes autocast chose
- GradScaler with `unscale_()` before existing `clip_grad_norm_`, then `step()` + `update()`
- Disabled on non-CUDA via `enabled=(device.type == "cuda")` — MPS falls back to FP32

**Interaction with grad clipping:**
```python
scaler.scale(total_loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(params, max_norm)
scaler.step(optimizer)
scaler.update()
```

---

## 2. Backbone SDPA Conversion

**Where:** `TransformerBlock.forward()` in `policy_backbone.py`

The backbone currently computes attention manually (matmul + bias + softmax + matmul), bypassing PyTorch's `F.scaled_dot_product_attention()`. This misses fused kernel backends.

**Fix:** Replace manual attention with:
```python
h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
```

SDPA's `attn_mask` is additive when passed as float — exactly matching the smolgen + shaw bias pattern. FlashAttention v2 doesn't support arbitrary additive masks, but the memory-efficient attention backend does and is still significantly faster than manual matmul on Ampere.

DiT blocks (`diffusion.py`) already use SDPA — no change needed there.

---

## 3. `torch.compile()`

**Where:** Model instantiation in training scripts and inference engines.

Compile the four core modules individually (not the whole training step):
```python
encoder = torch.compile(build_encoder(cfg).to(device))
```

**Decisions:**
- Default mode (not `reduce-overhead`) — CUDA graphs interact poorly with GradScaler
- Compile individual modules to avoid graph breaks from Python control flow (augmentation, DataLoader)
- Guard with device check — only compile on CUDA (MPS compile is experimental)
- Helper: `maybe_compile(module, device)` returns compiled or original module
- Warmup cost (~30-60s first batch) is negligible over 100-epoch training
- `state_dict()` works through `torch.compile` wrapper with default `fullgraph=False`

---

## 4. Gradient Checkpointing

**Where:** `ChessPolicyBackbone.forward()`, `ChessDiffusionModule.forward()`

Use `torch.utils.checkpoint.checkpoint()` per transformer block:
```python
for layer in self.layers:
    x = checkpoint(layer, x, combined_bias, use_reentrant=False)
```

**Decisions:**
- `use_reentrant=False` — modern API, compatible with `torch.compile()` and AMP
- Per-layer granularity (standard pattern)
- ~33% more compute for ~40% less activation memory — strongly worth it on 12GB
- May allow doubling batch size from 64 to 128
- Configurable via `--gradient-checkpointing` CLI flag (default True on CUDA, False on CPU/MPS)
- Applied to both backbone (15 layers) and diffusion (6 layers)

---

## 5. Proper DataLoader + pin_memory

**Where:** `train_phase1.py`, `train_phase2.py`

Replace manual batching with `torch.utils.data.DataLoader`.

**New class `ChessDataset(Dataset)`:**
- Wraps pre-stacked tensors (boards, policies, values) directly
- Augmentation (flip_board, flip_policy, value flip) in `__getitem__` — runs in worker processes
- Eliminates per-batch `torch.stack()` of individual TrainingExample objects

**DataLoader config:**
- `pin_memory=True` — async DMA transfer on CUDA
- `num_workers=2` — dataset is in-memory, workers handle augmentation overlap
- `persistent_workers=True` — avoids respawn overhead across epochs
- `shuffle=True` — replaces manual `random.shuffle()`

**SupervisedTrainer.train_step simplifies** — receives pre-stacked `(boards, policies, values)` tensors instead of `list[TrainingExample]`.

---

## 6. Persistent Device Buffers

**`CosineNoiseSchedule`** — convert to `nn.Module`, register `alpha_bar` as buffer:
```python
class CosineNoiseSchedule(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.register_buffer("alpha_bar", alpha_bar)
```
Eliminates CPU-to-GPU copy every diffusion forward pass. `schedule.to(device)` works naturally.

**`_SQUARE_FLIP` in augmentation** — no change needed. With DataLoader workers, augmentation runs on CPU. The `.to(policy.device)` call is a CPU-to-CPU no-op.

**Inference legal mask** — no change needed. Too small (~16KB per move) to matter.

---

## 7. MLX Inference Export

**Two new files:**

### Weight export script (`src/denoisr/scripts/export_mlx.py`)
- Load PyTorch checkpoint, flatten state dicts to numpy arrays
- Save as `.safetensors` (MLX's native format)
- New CLI command: `denoisr-export-mlx`

### MLX inference engine (`src/denoisr/inference/mlx_engine.py`)
- Reimplement 4 modules in `mlx.nn`: encoder, backbone, policy head, value head (~200 lines)
- Load weights from `.safetensors`
- Same `select_move(board) -> chess.Move` API as PyTorch engine
- Only inference modules — no world model, diffusion, consistency projector

**`mlx` as optional dependency** — not in `[project.dependencies]`. Import-guarded with helpful error message. No torch dependency at MLX inference time.

---

## What We're NOT Doing

- **No BF16** — RTX 3060 doesn't have dedicated BF16 Tensor Cores
- **No multi-GPU / DDP** — single RTX 3060 target
- **No `flash-attn` package** — SDPA's memory-efficient backend is sufficient; avoids fragile build deps
- **No custom Triton kernels** — `torch.compile()` handles kernel fusion
- **No CUDA graphs** — incompatible with GradScaler's dynamic scaling
- **No MLX training** — inference only on Mac
- **No ONNX export** — MLX safetensors is simpler and native
