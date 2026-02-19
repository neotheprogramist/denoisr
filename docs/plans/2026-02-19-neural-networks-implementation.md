# Neural Networks Implementation Plan (Tiers 4–5)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build all 8 neural network modules — the learned components of the chess engine.

**Architecture:** Square-as-token transformer with smolgen dynamic attention biases. Encoder maps board tensors to 64 latent tokens. Policy backbone processes them through a 15-layer transformer. World model predicts latent dynamics via a 12-layer causal transformer. Diffusion module uses continuous DDPM with a DiT backbone for implicit search.

**Tech Stack:** PyTorch (MPS/CUDA), frozen dataclasses from `denoisr.types`, Protocols from `denoisr.nn.protocols`

**Prerequisite:** Tiers 1–3 must have 100% passing tests.

**Testing note:** All nn tests use `SMALL_D_S = 64` (from conftest.py), `num_heads=4`, `num_layers=2` for speed. Full-scale hyperparameters are only used in training.

---

### Task 0: NN Protocols + Test Helpers

**Files:**

- Create: `src/denoisr/nn/protocols.py`
- Modify: `tests/conftest.py` (add nn fixtures)

**Step 1: Write protocols**

`src/denoisr/nn/protocols.py`:

```python
from typing import Protocol

import torch


class Encoder(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, 8, 8] -> [B, 64, d_s]"""
        ...


class SmolgenBias(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_s] -> [B, num_heads, 64, 64]"""
        ...


class RelativePositionBias(Protocol):
    def forward(self) -> torch.Tensor:
        """-> [num_heads, 64, 64] topology-aware position biases"""
        ...


class PolicyBackbone(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_s] -> [B, 64, d_s]"""
        ...


class PolicyHead(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_s] -> [B, 64, 64]"""
        ...


class ValueHead(Protocol):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 64, d_s] -> (wdl [B, 3], ply [B, 1])
        WDL probabilities sum to 1. Ply is predicted game length.
        """
        ...


class WorldModel(Protocol):
    def forward(
        self,
        states: torch.Tensor,
        action_from: torch.Tensor,
        action_to: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        states: [B, T, 64, d_s], action_from: [B, T], action_to: [B, T]
        -> next_states [B, T, 64, d_s], rewards [B, T]
        """
        ...


class DiffusionModule(Protocol):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [B, 64, d_s] noisy latent
        t: [B] timestep indices
        cond: [B, 64, d_s] condition
        -> [B, 64, d_s] predicted noise
        """
        ...


class ConsistencyProjector(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_s] -> [B, proj_dim]"""
        ...
```

**Step 2: Add nn test fixtures to conftest.py**

Append to `tests/conftest.py`:

```python
SMALL_NUM_HEADS = 4
SMALL_NUM_LAYERS = 2
SMALL_FFN_DIM = 128
SMALL_NUM_TIMESTEPS = 20


@pytest.fixture
def small_latent(device: torch.device) -> torch.Tensor:
    return torch.randn(2, 64, SMALL_D_S, device=device)


@pytest.fixture
def small_board_tensor(device: torch.device) -> torch.Tensor:
    return torch.randn(2, 12, 8, 8, device=device)
```

**Step 3: Commit**

```bash
git add src/denoisr/nn/protocols.py tests/conftest.py
git commit -m "feat: add nn protocols and test fixtures (T4-T5 setup)"
```

---

### Task 1: Encoder

**Files:**

- Create: `src/denoisr/nn/encoder.py`
- Test: `tests/test_nn/test_encoder.py`

**Step 1: Write failing tests**

`tests/test_nn/test_encoder.py`:

```python
import pytest
import torch

from denoisr.nn.encoder import ChessEncoder

from tests.conftest import SMALL_D_S


class TestChessEncoder:
    @pytest.fixture
    def encoder(self, device: torch.device) -> ChessEncoder:
        return ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)

    def test_output_shape(
        self, encoder: ChessEncoder, small_board_tensor: torch.Tensor
    ) -> None:
        out = encoder(small_board_tensor)
        assert out.shape == (2, 64, SMALL_D_S)

    def test_single_batch(
        self, encoder: ChessEncoder, device: torch.device
    ) -> None:
        x = torch.randn(1, 12, 8, 8, device=device)
        out = encoder(x)
        assert out.shape == (1, 64, SMALL_D_S)

    def test_gradient_flows(
        self, encoder: ChessEncoder, small_board_tensor: torch.Tensor
    ) -> None:
        out = encoder(small_board_tensor)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            assert p.grad is not None
            assert not torch.all(p.grad == 0)

    def test_deterministic(
        self, encoder: ChessEncoder, small_board_tensor: torch.Tensor
    ) -> None:
        encoder.eval()
        out1 = encoder(small_board_tensor)
        out2 = encoder(small_board_tensor)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(
        self, encoder: ChessEncoder, device: torch.device
    ) -> None:
        x1 = torch.randn(1, 12, 8, 8, device=device)
        x2 = torch.randn(1, 12, 8, 8, device=device)
        encoder.eval()
        assert not torch.allclose(encoder(x1), encoder(x2))

    def test_no_nan(
        self, encoder: ChessEncoder, small_board_tensor: torch.Tensor
    ) -> None:
        out = encoder(small_board_tensor)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nn/test_encoder.py -v`
Expected: ImportError.

**Step 3: Implement**

`src/denoisr/nn/encoder.py`:

```python
import torch
from torch import Tensor, nn


class ChessEncoder(nn.Module):
    """Encodes board tensor [B, C, 8, 8] into latent tokens [B, 64, d_s].

    Uses per-square linear projection plus a global board embedding
    (following BT3/BT4's approach for encoding whole-board context from layer 0).
    """

    def __init__(self, num_planes: int, d_s: int) -> None:
        super().__init__()
        self.square_embed = nn.Linear(num_planes, d_s)
        self.global_embed = nn.Sequential(
            nn.Linear(num_planes * 64, d_s),
            nn.Mish(),
            nn.Linear(d_s, d_s),
        )
        self.norm = nn.LayerNorm(d_s)

    def forward(self, x: Tensor) -> Tensor:
        B, C, _H, _W = x.shape
        # Per-square features: [B, 64, C]
        squares = x.reshape(B, C, 64).permute(0, 2, 1)
        local = self.square_embed(squares)

        # Global context: flatten entire board, project, broadcast
        flat = x.reshape(B, C * 64)
        glob = self.global_embed(flat).unsqueeze(1).expand(-1, 64, -1)

        return self.norm(local + glob)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_nn/test_encoder.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/nn/encoder.py tests/test_nn/test_encoder.py
git commit -m "feat: add chess encoder with global embedding (T4)"
```

---

### Task 2: Smolgen Dynamic Attention Bias

**Files:**

- Create: `src/denoisr/nn/smolgen.py`
- Test: `tests/test_nn/test_smolgen.py`

**Step 1: Write failing tests**

`tests/test_nn/test_smolgen.py`:

```python
import pytest
import torch

from denoisr.nn.smolgen import SmolgenBias

from tests.conftest import SMALL_D_S, SMALL_NUM_HEADS


class TestSmolgenBias:
    @pytest.fixture
    def smolgen(self, device: torch.device) -> SmolgenBias:
        return SmolgenBias(
            d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS
        ).to(device)

    def test_output_shape(
        self, smolgen: SmolgenBias, small_latent: torch.Tensor
    ) -> None:
        out = smolgen(small_latent)
        assert out.shape == (2, SMALL_NUM_HEADS, 64, 64)

    def test_content_dependent(
        self, smolgen: SmolgenBias, device: torch.device
    ) -> None:
        smolgen.eval()
        x1 = torch.randn(1, 64, SMALL_D_S, device=device)
        x2 = torch.randn(1, 64, SMALL_D_S, device=device)
        assert not torch.allclose(smolgen(x1), smolgen(x2))

    def test_gradient_flows(
        self, smolgen: SmolgenBias, small_latent: torch.Tensor
    ) -> None:
        out = smolgen(small_latent)
        out.sum().backward()
        for p in smolgen.parameters():
            assert p.grad is not None

    def test_no_nan(
        self, smolgen: SmolgenBias, small_latent: torch.Tensor
    ) -> None:
        out = smolgen(small_latent)
        assert not torch.isnan(out).any()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nn/test_smolgen.py -v`

**Step 3: Implement**

`src/denoisr/nn/smolgen.py`:

```python
import torch
from torch import Tensor, nn


class SmolgenBias(nn.Module):
    """Generates dynamic per-head attention biases from the full board state.

    Compresses 64 token embeddings into a small vector, then projects
    to H x 64 x 64 attention bias matrices. This lets the attention
    pattern adapt to the specific position (e.g., suppress long-range
    connections in closed positions).
    """

    def __init__(
        self, d_s: int, num_heads: int, compress_dim: int = 256
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.compress = nn.Sequential(
            nn.Linear(64 * d_s, compress_dim),
            nn.Mish(),
        )
        self.project = nn.Linear(compress_dim, num_heads * 64 * 64)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        flat = x.reshape(B, -1)
        compressed = self.compress(flat)
        biases = self.project(compressed)
        return biases.reshape(B, self.num_heads, 64, 64)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_nn/test_smolgen.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/nn/smolgen.py tests/test_nn/test_smolgen.py
git commit -m "feat: add smolgen dynamic attention bias (T4)"
```

---

### Task 3: Policy Head

**Files:**

- Create: `src/denoisr/nn/policy_head.py`
- Test: `tests/test_nn/test_policy_head.py`

**Step 1: Write failing tests**

`tests/test_nn/test_policy_head.py`:

```python
import pytest
import torch

from denoisr.nn.policy_head import ChessPolicyHead

from tests.conftest import SMALL_D_S


class TestChessPolicyHead:
    @pytest.fixture
    def head(self, device: torch.device) -> ChessPolicyHead:
        return ChessPolicyHead(d_s=SMALL_D_S).to(device)

    def test_output_shape(
        self, head: ChessPolicyHead, small_latent: torch.Tensor
    ) -> None:
        out = head(small_latent)
        assert out.shape == (2, 64, 64)

    def test_softmax_sums_to_one(
        self, head: ChessPolicyHead, small_latent: torch.Tensor
    ) -> None:
        logits = head(small_latent)
        probs = torch.softmax(logits.reshape(-1, 64 * 64), dim=-1)
        assert torch.allclose(
            probs.sum(dim=-1),
            torch.ones(probs.shape[0], device=probs.device),
        )

    def test_gradient_flows(
        self, head: ChessPolicyHead, small_latent: torch.Tensor
    ) -> None:
        out = head(small_latent)
        out.sum().backward()
        for p in head.parameters():
            assert p.grad is not None

    def test_no_nan(
        self, head: ChessPolicyHead, small_latent: torch.Tensor
    ) -> None:
        out = head(small_latent)
        assert not torch.isnan(out).any()
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_nn/test_policy_head.py -v`

**Step 3: Implement**

`src/denoisr/nn/policy_head.py`:

```python
import torch
from torch import Tensor, nn


class ChessPolicyHead(nn.Module):
    """Source-destination attention policy head.

    Computes bilinear attention between source-square queries and
    destination-square keys, producing a [B, 64, 64] logit matrix
    where entry (i, j) is the unnormalized log-probability of moving
    from square i to square j.
    """

    def __init__(self, d_s: int, d_head: int = 128) -> None:
        super().__init__()
        self.query = nn.Linear(d_s, d_head)
        self.key = nn.Linear(d_s, d_head)
        self.scale = d_head**-0.5

    def forward(self, x: Tensor) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        return torch.bmm(q, k.transpose(1, 2)) * self.scale
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_nn/test_policy_head.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/nn/policy_head.py tests/test_nn/test_policy_head.py
git commit -m "feat: add source-destination policy head (T4)"
```

---

### Task 4: WDLP Value Head (Win/Draw/Loss + Ply Prediction)

**Spec reference:** "WDLP value head — predict expected game length as auxiliary signal."

**Files:**

- Create: `src/denoisr/nn/value_head.py`
- Test: `tests/test_nn/test_value_head.py`

**Step 1: Write failing tests**

`tests/test_nn/test_value_head.py`:

```python
import pytest
import torch

from denoisr.nn.value_head import ChessValueHead

from tests.conftest import SMALL_D_S


class TestChessValueHead:
    @pytest.fixture
    def head(self, device: torch.device) -> ChessValueHead:
        return ChessValueHead(d_s=SMALL_D_S).to(device)

    def test_wdl_output_shape(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, ply = head(small_latent)
        assert wdl.shape == (2, 3)

    def test_ply_output_shape(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, ply = head(small_latent)
        assert ply.shape == (2, 1)

    def test_wdl_sums_to_one(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, _ = head(small_latent)
        sums = wdl.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_wdl_in_zero_one(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, _ = head(small_latent)
        assert (wdl >= 0).all()
        assert (wdl <= 1).all()

    def test_ply_non_negative(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        _, ply = head(small_latent)
        assert (ply >= 0).all()

    def test_gradient_flows(
        self, head: ChessValueHead, small_latent: torch.Tensor
    ) -> None:
        wdl, ply = head(small_latent)
        (wdl.sum() + ply.sum()).backward()
        for p in head.parameters():
            assert p.grad is not None
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_nn/test_value_head.py -v`

**Step 3: Implement**

`src/denoisr/nn/value_head.py`:

```python
import torch
from torch import Tensor, nn


class ChessValueHead(nn.Module):
    """WDLP value head: Win/Draw/Loss probabilities + Ply prediction.

    Mean-pools the 64 token embeddings, normalizes, and produces:
    - WDL: 3-class probability distribution via softmax
    - Ply: predicted game length (non-negative) via softplus

    The ply prediction serves as an auxiliary training signal that
    helps the model understand position complexity (spec: WDLP).
    """

    def __init__(self, d_s: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_s)
        self.wdl_linear = nn.Linear(d_s, 3)
        self.ply_linear = nn.Linear(d_s, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self.norm(x.mean(dim=1))
        wdl = torch.softmax(self.wdl_linear(pooled), dim=-1)
        ply = torch.nn.functional.softplus(self.ply_linear(pooled))
        return wdl, ply
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_nn/test_value_head.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/nn/value_head.py tests/test_nn/test_value_head.py
git commit -m "feat: add WDL value head (T4)"
```

---

### Task 5: Policy Backbone (Transformer + Smolgen + Shaw Relative PE)

**Spec reference:** "Shaw relative position encoding / GAB — superior to RoPE for chess topology." Smolgen provides content-dependent bias; Shaw provides topology-aware bias. Both are additive.

**Files:**

- Create: `src/denoisr/nn/relative_pos.py`
- Create: `src/denoisr/nn/policy_backbone.py`
- Test: `tests/test_nn/test_relative_pos.py`
- Test: `tests/test_nn/test_policy_backbone.py`

**Step 1: Write failing tests**

`tests/test_nn/test_relative_pos.py`:

```python
import pytest
import torch

from denoisr.nn.relative_pos import ShawRelativePositionBias

from tests.conftest import SMALL_NUM_HEADS


class TestShawRelativePositionBias:
    @pytest.fixture
    def pe(self) -> ShawRelativePositionBias:
        return ShawRelativePositionBias(num_heads=SMALL_NUM_HEADS)

    def test_output_shape(self, pe: ShawRelativePositionBias) -> None:
        out = pe()
        assert out.shape == (SMALL_NUM_HEADS, 64, 64)

    def test_topology_aware(self, pe: ShawRelativePositionBias) -> None:
        """Adjacent squares should have different bias than distant ones."""
        out = pe()
        # e2 (sq=12) to e4 (sq=28): rank diff=2, file diff=0
        # e2 (sq=12) to a8 (sq=56): rank diff=6, file diff=-4
        # These should have different biases
        assert not torch.allclose(out[:, 12, 28], out[:, 12, 56])

    def test_deterministic(self, pe: ShawRelativePositionBias) -> None:
        out1 = pe()
        out2 = pe()
        assert torch.equal(out1, out2)

    def test_gradient_flows(self, pe: ShawRelativePositionBias) -> None:
        out = pe()
        out.sum().backward()
        for p in pe.parameters():
            assert p.grad is not None
```

`tests/test_nn/test_policy_backbone.py`:

```python
import pytest
import torch

from denoisr.nn.policy_backbone import ChessPolicyBackbone

from tests.conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


class TestChessPolicyBackbone:
    @pytest.fixture
    def backbone(self, device: torch.device) -> ChessPolicyBackbone:
        return ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)

    def test_output_shape_preserved(
        self, backbone: ChessPolicyBackbone, small_latent: torch.Tensor
    ) -> None:
        out = backbone(small_latent)
        assert out.shape == small_latent.shape

    def test_gradient_flows_through_all_layers(
        self, backbone: ChessPolicyBackbone, small_latent: torch.Tensor
    ) -> None:
        out = backbone(small_latent)
        out.sum().backward()
        for name, p in backbone.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

    def test_smolgen_biases_used(
        self, backbone: ChessPolicyBackbone, device: torch.device
    ) -> None:
        smolgen_params = [
            (n, p)
            for n, p in backbone.named_parameters()
            if "smolgen" in n
        ]
        assert len(smolgen_params) > 0
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        backbone(x).sum().backward()
        for name, p in smolgen_params:
            assert p.grad is not None, f"No gradient for {name}"

    def test_shaw_pe_biases_used(
        self, backbone: ChessPolicyBackbone, device: torch.device
    ) -> None:
        shaw_params = [
            (n, p)
            for n, p in backbone.named_parameters()
            if "shaw" in n or "relative" in n
        ]
        assert len(shaw_params) > 0
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        backbone(x).sum().backward()
        for name, p in shaw_params:
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan(
        self, backbone: ChessPolicyBackbone, small_latent: torch.Tensor
    ) -> None:
        out = backbone(small_latent)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_inputs_different_outputs(
        self, backbone: ChessPolicyBackbone, device: torch.device
    ) -> None:
        backbone.eval()
        x1 = torch.randn(1, 64, SMALL_D_S, device=device)
        x2 = torch.randn(1, 64, SMALL_D_S, device=device)
        assert not torch.allclose(backbone(x1), backbone(x2))
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_nn/test_relative_pos.py tests/test_nn/test_policy_backbone.py -v`

**Step 3: Implement Shaw Relative Position Bias**

`src/denoisr/nn/relative_pos.py`:

```python
import chess
import torch
from torch import Tensor, nn


class ShawRelativePositionBias(nn.Module):
    """Shaw relative position encoding for chess board topology.

    Learns a bias table indexed by (delta_rank, delta_file) between
    every pair of squares. Superior to RoPE for chess because it
    directly captures the spatial relationships of the 8x8 board
    (adjacent squares, diagonals, knight jumps) rather than treating
    positions as a 1D sequence.

    The bias table has shape [num_heads, 15, 15] for rank/file
    differences in range [-7, +7]. The output is [num_heads, 64, 64].
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Rank and file differences range from -7 to +7 = 15 values
        self.bias_table = nn.Parameter(
            torch.zeros(num_heads, 15, 15)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # Precompute (rank_diff, file_diff) for all 64x64 square pairs
        coords = torch.tensor(
            [(chess.square_rank(sq), chess.square_file(sq)) for sq in range(64)]
        )
        rank_diff = coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0) + 7  # [64, 64], shifted to [0, 14]
        file_diff = coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0) + 7  # [64, 64], shifted to [0, 14]
        self.register_buffer("rank_idx", rank_diff.long())
        self.register_buffer("file_idx", file_diff.long())

    def forward(self) -> Tensor:
        # Index into bias table: [num_heads, 64, 64]
        return self.bias_table[:, self.rank_idx, self.file_idx]
```

**Step 4: Implement Policy Backbone with Shaw PE**

`src/denoisr/nn/policy_backbone.py`:

```python
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from denoisr.nn.relative_pos import ShawRelativePositionBias
from denoisr.nn.smolgen import SmolgenBias


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with additive attention bias support.

    Accepts combined smolgen + Shaw PE biases.
    """

    def __init__(self, d_s: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads
        self.norm1 = nn.LayerNorm(d_s)
        self.qkv = nn.Linear(d_s, 3 * d_s)
        self.out_proj = nn.Linear(d_s, d_s)
        self.norm2 = nn.LayerNorm(d_s)
        self.ffn = nn.Sequential(
            nn.Linear(d_s, ffn_dim),
            nn.Mish(),
            nn.Linear(ffn_dim, d_s),
        )

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + h

        x = x + self.ffn(self.norm2(x))
        return x


class ChessPolicyBackbone(nn.Module):
    """Encoder-only transformer with smolgen + Shaw relative PE.

    Two types of attention bias are combined additively:
    - Smolgen: content-dependent (adapts to specific position)
    - Shaw relative PE: topology-aware (captures spatial relationships)

    Both are computed once and shared across all transformer layers.
    """

    def __init__(
        self, d_s: int, num_heads: int, num_layers: int, ffn_dim: int
    ) -> None:
        super().__init__()
        self.smolgen = SmolgenBias(d_s, num_heads)
        self.shaw_relative_pe = ShawRelativePositionBias(num_heads)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_s, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_s)

    def forward(self, x: Tensor) -> Tensor:
        smolgen_bias = self.smolgen(x)  # [B, H, 64, 64]
        shaw_bias = self.shaw_relative_pe()  # [H, 64, 64]
        combined_bias = smolgen_bias + shaw_bias.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, attn_bias=combined_bias)
        return self.final_norm(x)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_nn/test_policy_backbone.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/nn/policy_backbone.py tests/test_nn/test_policy_backbone.py
git commit -m "feat: add transformer policy backbone with smolgen (T5)"
```

---

### Task 6: World Model

**Files:**

- Create: `src/denoisr/nn/world_model.py`
- Test: `tests/test_nn/test_world_model.py`

**Step 1: Write failing tests**

`tests/test_nn/test_world_model.py`:

```python
import pytest
import torch

from denoisr.nn.world_model import ChessWorldModel

from tests.conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


class TestChessWorldModel:
    @pytest.fixture
    def model(self, device: torch.device) -> ChessWorldModel:
        return ChessWorldModel(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)

    def test_output_shapes(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        B, T = 2, 5
        states = torch.randn(B, T, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (B, T), device=device)
        act_to = torch.randint(0, 64, (B, T), device=device)
        next_states, rewards = model(states, act_from, act_to)
        assert next_states.shape == (B, T, 64, SMALL_D_S)
        assert rewards.shape == (B, T)

    def test_single_step(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        states = torch.randn(1, 1, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (1, 1), device=device)
        act_to = torch.randint(0, 64, (1, 1), device=device)
        next_states, rewards = model(states, act_from, act_to)
        assert next_states.shape == (1, 1, 64, SMALL_D_S)

    def test_causal_masking(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        """Changing future inputs should not affect past outputs."""
        B, T = 1, 4
        states = torch.randn(B, T, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (B, T), device=device)
        act_to = torch.randint(0, 64, (B, T), device=device)

        model.eval()
        out1, _ = model(states, act_from, act_to)

        states2 = states.clone()
        states2[:, -1] = torch.randn(1, 64, SMALL_D_S, device=device)
        out2, _ = model(states2, act_from, act_to)

        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_gradient_flows(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        states = torch.randn(2, 3, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (2, 3), device=device)
        act_to = torch.randint(0, 64, (2, 3), device=device)
        next_states, rewards = model(states, act_from, act_to)
        (next_states.sum() + rewards.sum()).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        states = torch.randn(2, 3, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (2, 3), device=device)
        act_to = torch.randint(0, 64, (2, 3), device=device)
        ns, rw = model(states, act_from, act_to)
        assert not torch.isnan(ns).any()
        assert not torch.isnan(rw).any()
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_nn/test_world_model.py -v`

**Step 3: Implement**

`src/denoisr/nn/world_model.py`:

```python
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalTransformerBlock(nn.Module):
    """Pre-norm transformer block with causal (autoregressive) masking."""

    def __init__(self, d_s: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads
        self.norm1 = nn.LayerNorm(d_s)
        self.qkv = nn.Linear(d_s, 3 * d_s)
        self.out_proj = nn.Linear(d_s, d_s)
        self.norm2 = nn.LayerNorm(d_s)
        self.ffn = nn.Sequential(
            nn.Linear(d_s, ffn_dim),
            nn.Mish(),
            nn.Linear(ffn_dim, d_s),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).reshape(B, T, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class ChessWorldModel(nn.Module):
    """Causal transformer world model (UniZero-style).

    Each timestep's (state, action) pair is compressed into a single
    token via mean-pooling + action embedding + MLP fusion (STORM approach).
    The causal transformer processes the token sequence, predicting
    next latent states and rewards.
    """

    def __init__(
        self, d_s: int, num_heads: int, num_layers: int, ffn_dim: int
    ) -> None:
        super().__init__()
        self.d_s = d_s
        self.state_compress = nn.Sequential(
            nn.Linear(d_s, d_s),
            nn.Mish(),
        )
        self.from_embed = nn.Embedding(64, d_s // 2)
        self.to_embed = nn.Embedding(64, d_s // 2)
        self.fuse = nn.Sequential(
            nn.Linear(d_s * 2, d_s),
            nn.Mish(),
            nn.Linear(d_s, d_s),
        )
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(d_s, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_s)
        self.state_pred = nn.Linear(d_s, 64 * d_s)
        self.reward_pred = nn.Linear(d_s, 1)

    def forward(
        self,
        states: Tensor,
        action_from: Tensor,
        action_to: Tensor,
    ) -> tuple[Tensor, Tensor]:
        B, T = states.shape[:2]

        compressed = self.state_compress(states.mean(dim=2))
        act_emb = torch.cat(
            [self.from_embed(action_from), self.to_embed(action_to)], dim=-1
        )
        fused = self.fuse(torch.cat([compressed, act_emb], dim=-1))

        h = fused
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)

        next_states = self.state_pred(h).reshape(B, T, 64, self.d_s)
        rewards = self.reward_pred(h).squeeze(-1)
        return next_states, rewards
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_nn/test_world_model.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/nn/world_model.py tests/test_nn/test_world_model.py
git commit -m "feat: add causal transformer world model (T5)"
```

---

### Task 7: Consistency Projector

**Files:**

- Create: `src/denoisr/nn/consistency.py`
- Test: `tests/test_nn/test_consistency.py`

**Step 1: Write failing tests**

`tests/test_nn/test_consistency.py`:

```python
import pytest
import torch

from denoisr.nn.consistency import ChessConsistencyProjector

from tests.conftest import SMALL_D_S


class TestChessConsistencyProjector:
    @pytest.fixture
    def proj(self, device: torch.device) -> ChessConsistencyProjector:
        return ChessConsistencyProjector(d_s=SMALL_D_S, proj_dim=32).to(
            device
        )

    def test_output_shape(
        self, proj: ChessConsistencyProjector, small_latent: torch.Tensor
    ) -> None:
        out = proj(small_latent)
        assert out.shape == (2, 32)

    def test_cosine_similarity_defined(
        self, proj: ChessConsistencyProjector, device: torch.device
    ) -> None:
        x1 = torch.randn(2, 64, SMALL_D_S, device=device)
        x2 = torch.randn(2, 64, SMALL_D_S, device=device)
        p1 = proj(x1)
        p2 = proj(x2)
        cos_sim = torch.nn.functional.cosine_similarity(p1, p2)
        assert cos_sim.shape == (2,)
        assert ((cos_sim >= -1.0) & (cos_sim <= 1.0)).all()

    def test_gradient_flows(
        self, proj: ChessConsistencyProjector, small_latent: torch.Tensor
    ) -> None:
        out = proj(small_latent)
        out.sum().backward()
        for p in proj.parameters():
            assert p.grad is not None

    def test_stop_gradient_target(
        self, proj: ChessConsistencyProjector, device: torch.device
    ) -> None:
        x_pred = torch.randn(
            1, 64, SMALL_D_S, device=device, requires_grad=True
        )
        x_target = torch.randn(
            1, 64, SMALL_D_S, device=device, requires_grad=True
        )
        p_pred = proj(x_pred)
        with torch.no_grad():
            p_target = proj(x_target)
        loss = -torch.nn.functional.cosine_similarity(
            p_pred, p_target
        ).mean()
        loss.backward()
        assert x_pred.grad is not None
        assert x_target.grad is None
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_nn/test_consistency.py -v`

**Step 3: Implement**

`src/denoisr/nn/consistency.py`:

```python
from torch import Tensor, nn


class ChessConsistencyProjector(nn.Module):
    """SimSiam-style consistency projector.

    Mean-pools the 64 latent tokens and projects to a low-dimensional
    space for computing consistency loss between predicted and actual
    next states. Used with stop-gradient on the target branch to
    prevent latent state collapse (EfficientZero).
    """

    def __init__(self, d_s: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_s, proj_dim),
            nn.Mish(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1)
        return self.projector(pooled)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_nn/test_consistency.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/nn/consistency.py tests/test_nn/test_consistency.py
git commit -m "feat: add SimSiam consistency projector (T5)"
```

---

### Task 8: Diffusion Module (DiT with DDPM)

This is the most complex module. It uses continuous DDPM (Gaussian noise) in latent space — not D3PM (discrete), since our latent tokens are continuous float tensors.

**Intuition — the corruption-and-recovery game:** During training, the diffusion module plays a game with itself: (1) take a real future trajectory from self-play, (2) corrupt it with Gaussian noise at a random timestep `t`, (3) ask the model to recover the original clean latent. At heavy noise (high `t`), almost everything is masked — the model must hallucinate plausible futures from the current position alone (strategic reasoning). At light noise (low `t`), only the distant future is unclear — the model refines tactical details. Learning to denoise at _every_ noise level teaches both local tactics and long-range strategy in a single module. See the component design doc's "How the Diffusion Model Learns to Score Moves Through Self-Play" section for the full explanation.

**Files:**

- Create: `src/denoisr/nn/diffusion.py`
- Test: `tests/test_nn/test_diffusion.py`

**Step 1: Write failing tests**

`tests/test_nn/test_diffusion.py`:

```python
import pytest
import torch

from denoisr.nn.diffusion import (
    ChessDiffusionModule,
    CosineNoiseSchedule,
)

from tests.conftest import (
    SMALL_D_S,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


class TestCosineNoiseSchedule:
    @pytest.fixture
    def schedule(self) -> CosineNoiseSchedule:
        return CosineNoiseSchedule(num_timesteps=SMALL_NUM_TIMESTEPS)

    def test_alpha_bar_monotonic_decreasing(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        ab = schedule.alpha_bar
        assert ab.shape == (SMALL_NUM_TIMESTEPS,)
        for i in range(len(ab) - 1):
            assert ab[i] > ab[i + 1]

    def test_alpha_bar_bounds(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        assert schedule.alpha_bar[0] > 0.9
        assert schedule.alpha_bar[-1] < 0.1

    def test_q_sample_shape(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        x_0 = torch.randn(2, 64, SMALL_D_S)
        t = torch.tensor([0, SMALL_NUM_TIMESTEPS - 1])
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise)
        assert x_t.shape == x_0.shape

    def test_q_sample_t0_close_to_clean(
        self, schedule: CosineNoiseSchedule
    ) -> None:
        x_0 = torch.randn(1, 64, SMALL_D_S)
        t = torch.tensor([0])
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise)
        assert torch.allclose(x_t, x_0, atol=0.2)


class TestChessDiffusionModule:
    @pytest.fixture
    def diffusion(self, device: torch.device) -> ChessDiffusionModule:
        return ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)

    def test_output_shape(
        self, diffusion: ChessDiffusionModule, device: torch.device
    ) -> None:
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.randint(0, SMALL_NUM_TIMESTEPS, (2,), device=device)
        cond = torch.randn(2, 64, SMALL_D_S, device=device)
        out = diffusion(x, t, cond)
        assert out.shape == (2, 64, SMALL_D_S)

    def test_different_timesteps_different_outputs(
        self, diffusion: ChessDiffusionModule, device: torch.device
    ) -> None:
        diffusion.eval()
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        cond = torch.randn(1, 64, SMALL_D_S, device=device)
        t0 = torch.tensor([0], device=device)
        t1 = torch.tensor([SMALL_NUM_TIMESTEPS - 1], device=device)
        out0 = diffusion(x, t0, cond)
        out1 = diffusion(x, t1, cond)
        assert not torch.allclose(out0, out1)

    def test_gradient_flows(
        self, diffusion: ChessDiffusionModule, device: torch.device
    ) -> None:
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.randint(0, SMALL_NUM_TIMESTEPS, (2,), device=device)
        cond = torch.randn(2, 64, SMALL_D_S, device=device)
        out = diffusion(x, t, cond)
        out.sum().backward()
        for name, p in diffusion.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_adaln_zero_init(
        self, diffusion: ChessDiffusionModule
    ) -> None:
        w = diffusion.final_proj.weight
        b = diffusion.final_proj.bias
        assert torch.allclose(w, torch.zeros_like(w))
        assert torch.allclose(b, torch.zeros_like(b))

    def test_no_nan(
        self, diffusion: ChessDiffusionModule, device: torch.device
    ) -> None:
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        t = torch.randint(0, SMALL_NUM_TIMESTEPS, (2,), device=device)
        cond = torch.randn(2, 64, SMALL_D_S, device=device)
        out = diffusion(x, t, cond)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
```

**Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_nn/test_diffusion.py -v`

**Step 3: Implement**

`src/denoisr/nn/diffusion.py`:

```python
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CosineNoiseSchedule:
    """Cosine noise schedule for continuous DDPM (Nichol & Dhariwal 2021).

    Produces alpha_bar_t values that follow a cosine curve, giving
    a gentler noise schedule than linear beta scheduling.
    """

    def __init__(self, num_timesteps: int, s: float = 0.008) -> None:
        self.num_timesteps = num_timesteps
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f = (
            torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2)
            ** 2
        )
        alpha_bar = f / f[0]
        self.alpha_bar = (
            alpha_bar[:num_timesteps].float().clamp(min=1e-5, max=0.9999)
        )

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: add noise at timestep t."""
        ab = self.alpha_bar.to(x_0.device)[t]
        while ab.ndim < x_0.ndim:
            ab = ab.unsqueeze(-1)
        return ab.sqrt() * x_0 + (1 - ab).sqrt() * noise


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN-Zero conditioning."""

    def __init__(self, d_s: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads
        self.norm1 = nn.LayerNorm(d_s, elementwise_affine=False)
        self.qkv = nn.Linear(d_s, 3 * d_s)
        self.out_proj = nn.Linear(d_s, d_s)
        self.norm2 = nn.LayerNorm(d_s, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_s, d_s * 4),
            nn.Mish(),
            nn.Linear(d_s * 4, d_s),
        )
        self.adaln = nn.Sequential(
            nn.Mish(),
            nn.Linear(d_s, 6 * d_s),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        B, S, D = x.shape
        params = self.adaln(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(
            6, dim=-1
        )

        h = self.norm1(x) * (1 + scale1) + shift1
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + gate1 * h

        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.ffn(h)
        x = x + gate2 * h
        return x


class ChessDiffusionModule(nn.Module):
    """DiT-based diffusion module for latent-space trajectory imagination.

    Uses continuous DDPM (Gaussian noise) in the latent space of
    board representations. Conditioned on the current board state
    and diffusion timestep via AdaLN-Zero modulation.

    The final projection is zero-initialized so each block initially
    acts as identity, ensuring stable early training.
    """

    def __init__(
        self,
        d_s: int,
        num_heads: int,
        num_layers: int,
        num_timesteps: int,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Embedding(num_timesteps, d_s),
            nn.Mish(),
            nn.Linear(d_s, d_s),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(d_s, d_s),
            nn.Mish(),
        )
        self.layers = nn.ModuleList(
            [DiTBlock(d_s, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_s)
        self.final_proj = nn.Linear(d_s, d_s)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.time_embed(t)
        c_emb = self.cond_proj(cond.mean(dim=1))
        c = (t_emb + c_emb).unsqueeze(1).expand(-1, 64, -1)

        for layer in self.layers:
            x = layer(x, c)

        return self.final_proj(self.final_norm(x))
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_nn/test_diffusion.py -v`
Expected: All pass.

**Step 5: Run ALL T4-T5 tests**

Run: `uv run pytest tests/test_nn/ -v`
Expected: All pass.

**Step 6: Commit**

```bash
git add src/denoisr/nn/diffusion.py tests/test_nn/test_diffusion.py
git commit -m "feat: add DiT diffusion module with DDPM noise schedule (T5)"
```

---

## Tier 4–5 Gate Check

```bash
uv run pytest tests/ -v --tb=short
```

All tests across T1–T5 must pass before proceeding to Tier 6 (training infrastructure).

**Components validated:**

- Encoder with global embedding
- Smolgen dynamic attention biases
- Shaw relative position encoding (topology-aware)
- Source-destination policy head
- WDLP value head (WDL + ply prediction)
- Transformer policy backbone (smolgen + Shaw PE)
- Causal transformer world model (STORM compression)
- SimSiam consistency projector (stop-gradient)
- DiT diffusion module (continuous DDPM, cosine schedule, AdaLN-Zero)

**Next plan:** `2026-02-19-training-inference-implementation.md`
