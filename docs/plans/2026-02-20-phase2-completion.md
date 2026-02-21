# Phase 2 Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete Phase 2 training by unifying all 6 loss terms (policy, value, diffusion, world model state, world model reward, consistency) in a single Phase2Trainer, with enriched trajectory extraction and automated gate assessment.

**Architecture:** A new `Phase2Trainer` class replaces `DiffusionTrainer` for Phase 2. It freezes the encoder, trains backbone at reduced LR, and trains policy/value heads, world model, diffusion, and consistency modules at full LR. Each step: encode boards to latent, run backbone and heads for policy/value, world model for state/reward prediction, diffusion for noise prediction, consistency projector on predicted vs actual next states. All losses flow through the existing `ChessLossComputer` with optional HarmonyDream balancing.

**Tech Stack:** PyTorch, existing denoisr modules (`ChessLossComputer`, `ChessWorldModel`, `ChessDiffusionModule`, `ChessConsistencyProjector`, `CosineNoiseSchedule`), `chess` library for PGN parsing

**Design doc:** `docs/plans/2026-02-20-phase2-completion-design.md`

---

## Task 1: Extend ChessLossComputer to accept world model state loss

The design specifies 6 loss terms including "world model state" (MSE between predicted and actual next latent). The loss computer currently accepts 4 auxiliaries: consistency, diffusion, reward, ply. We add "state" as a 5th auxiliary.

**Files:**
- Modify: `src/denoisr/training/loss.py:23-45` (init and compute loop)
- Test: `tests/test_training/test_loss.py`

**Step 1: Write the failing test**

Add to `tests/test_training/test_loss.py`:

```python
def test_state_loss_included_in_total(self) -> None:
    loss_fn = ChessLossComputer()
    pred_policy = torch.randn(2, 64, 64)
    pred_value = torch.randn(2, 3)
    target_policy = torch.zeros(2, 64, 64)
    target_policy[:, 0, 0] = 1.0
    target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

    total_base, _ = loss_fn.compute(
        pred_policy, pred_value, target_policy, target_value
    )
    total_with_state, breakdown = loss_fn.compute(
        pred_policy, pred_value, target_policy, target_value,
        state_loss=torch.tensor(0.5),
    )
    assert total_with_state.item() > total_base.item()
    assert "state" in breakdown

def test_state_weight_scales_state_loss(self) -> None:
    loss_fn = ChessLossComputer(state_weight=2.0)
    pred_policy = torch.randn(2, 64, 64)
    pred_value = torch.randn(2, 3)
    target_policy = torch.zeros(2, 64, 64)
    target_policy[:, 0, 0] = 1.0
    target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

    total_w1, _ = ChessLossComputer(state_weight=1.0).compute(
        pred_policy, pred_value, target_policy, target_value,
        state_loss=torch.tensor(1.0),
    )
    total_w2, _ = loss_fn.compute(
        pred_policy, pred_value, target_policy, target_value,
        state_loss=torch.tensor(1.0),
    )
    # state_weight=2.0 adds 1.0 more to total than state_weight=1.0
    assert abs(total_w2.item() - total_w1.item() - 1.0) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_loss.py::TestChessLossComputer::test_state_loss_included_in_total tests/test_training/test_loss.py::TestChessLossComputer::test_state_weight_scales_state_loss -v`

Expected: FAIL (state_loss kwarg is silently ignored, "state" not in breakdown)

**Step 3: Write minimal implementation**

In `src/denoisr/training/loss.py`:

1. Add `state_weight` parameter to `__init__`:

```python
def __init__(
    self,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    consistency_weight: float = 1.0,
    diffusion_weight: float = 1.0,
    reward_weight: float = 1.0,
    ply_weight: float = 0.1,
    state_weight: float = 1.0,  # NEW
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
        "state": state_weight,  # NEW
    }
    # ... rest unchanged
```

2. Add "state" to the auxiliary loop in `compute`:

```python
for name in ("consistency", "diffusion", "reward", "ply", "state"):
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_loss.py -v`

Expected: All tests PASS including the two new ones

**Step 5: Commit**

```bash
git add src/denoisr/training/loss.py tests/test_training/test_loss.py
git commit -m "feat: extend ChessLossComputer with state auxiliary loss"
```

---

## Task 2: TrajectoryBatch dataclass

A frozen dataclass holding enriched trajectory data for Phase 2 training. Each trajectory has T consecutive board states, T-1 actions (with from/to squares), T-1 one-hot policy targets, a single WDL value target per trajectory, and T-1 per-move reward signals.

**Files:**
- Create: `src/denoisr/training/phase2_trainer.py` (initial skeleton with TrajectoryBatch only)
- Create: `tests/test_training/test_phase2_trainer.py`

**Step 1: Write the failing test**

Create `tests/test_training/test_phase2_trainer.py`:

```python
import pytest
import torch

from denoisr.training.phase2_trainer import TrajectoryBatch


class TestTrajectoryBatch:
    def test_valid_shapes(self) -> None:
        B, T, C = 4, 5, 12
        batch = TrajectoryBatch(
            boards=torch.randn(B, T, C, 8, 8),
            actions_from=torch.randint(0, 64, (B, T - 1)),
            actions_to=torch.randint(0, 64, (B, T - 1)),
            policies=torch.zeros(B, T - 1, 64, 64),
            values=torch.tensor([[1.0, 0.0, 0.0]] * B),
            rewards=torch.zeros(B, T - 1),
        )
        assert batch.boards.shape == (B, T, C, 8, 8)
        assert batch.actions_from.shape == (B, T - 1)
        assert batch.policies.shape == (B, T - 1, 64, 64)
        assert batch.values.shape == (B, 3)
        assert batch.rewards.shape == (B, T - 1)

    def test_frozen(self) -> None:
        batch = TrajectoryBatch(
            boards=torch.randn(2, 3, 12, 8, 8),
            actions_from=torch.zeros(2, 2, dtype=torch.long),
            actions_to=torch.zeros(2, 2, dtype=torch.long),
            policies=torch.zeros(2, 2, 64, 64),
            values=torch.tensor([[1.0, 0.0, 0.0]] * 2),
            rewards=torch.zeros(2, 2),
        )
        with pytest.raises(AttributeError):
            batch.boards = torch.randn(2, 3, 12, 8, 8)  # type: ignore[misc]

    def test_rejects_mismatched_time_dims(self) -> None:
        with pytest.raises(ValueError, match="time"):
            TrajectoryBatch(
                boards=torch.randn(2, 5, 12, 8, 8),
                actions_from=torch.zeros(2, 3, dtype=torch.long),  # should be 4
                actions_to=torch.zeros(2, 4, dtype=torch.long),
                policies=torch.zeros(2, 4, 64, 64),
                values=torch.tensor([[1.0, 0.0, 0.0]] * 2),
                rewards=torch.zeros(2, 4),
            )

    def test_rejects_mismatched_batch_dims(self) -> None:
        with pytest.raises(ValueError, match="batch"):
            TrajectoryBatch(
                boards=torch.randn(4, 5, 12, 8, 8),
                actions_from=torch.zeros(2, 4, dtype=torch.long),  # B=2 vs 4
                actions_to=torch.zeros(4, 4, dtype=torch.long),
                policies=torch.zeros(4, 4, 64, 64),
                values=torch.tensor([[1.0, 0.0, 0.0]] * 4),
                rewards=torch.zeros(4, 4),
            )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestTrajectoryBatch -v`

Expected: FAIL (ImportError: cannot import name 'TrajectoryBatch')

**Step 3: Write minimal implementation**

Create `src/denoisr/training/phase2_trainer.py`:

```python
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TrajectoryBatch:
    """Enriched trajectory data for Phase 2 training.

    Each trajectory contains T consecutive board states connected by T-1 actions.
    Boards are raw encoder outputs, not BoardTensor newtypes.
    """

    boards: Tensor       # [N, T, C, 8, 8]
    actions_from: Tensor  # [N, T-1] (int64)
    actions_to: Tensor    # [N, T-1] (int64)
    policies: Tensor      # [N, T-1, 64, 64] one-hot from played move
    values: Tensor        # [N, 3] WDL from game result
    rewards: Tensor       # [N, T-1] per-move reward signal

    def __post_init__(self) -> None:
        B, T = self.boards.shape[:2]
        Tm1 = T - 1
        expected = {
            "actions_from": (B, Tm1),
            "actions_to": (B, Tm1),
            "policies": (B, Tm1, 64, 64),
            "rewards": (B, Tm1),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual[0] != B:
                raise ValueError(
                    f"{name} batch dim {actual[0]} != boards batch dim {B}"
                )
            if actual[1] != Tm1:
                raise ValueError(
                    f"{name} time dim {actual[1]} != expected {Tm1} "
                    f"(boards T={T})"
                )
        if self.values.shape[0] != B:
            raise ValueError(
                f"values batch dim {self.values.shape[0]} != boards batch "
                f"dim {B}"
            )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestTrajectoryBatch -v`

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/phase2_trainer.py tests/test_training/test_phase2_trainer.py
git commit -m "feat: add TrajectoryBatch dataclass for Phase 2"
```

---

## Task 3: Phase2Trainer implementation

The core Phase 2 trainer that computes all 6 loss terms in one training step. Freezes encoder, trains backbone at reduced LR, and trains all other modules at full LR.

**Files:**
- Modify: `src/denoisr/training/phase2_trainer.py` (add Phase2Trainer class)
- Modify: `tests/test_training/test_phase2_trainer.py` (add Phase2Trainer tests)

**Step 1: Write the failing tests**

Add imports and helpers to `tests/test_training/test_phase2_trainer.py`:

```python
from denoisr.nn.consistency import ChessConsistencyProjector
from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.nn.world_model import ChessWorldModel
from denoisr.training.loss import ChessLossComputer
from denoisr.training.phase2_trainer import Phase2Trainer, TrajectoryBatch

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


def _make_trajectory_batch(
    B: int = 2,
    T: int = 5,
    C: int = 12,
    device: torch.device = torch.device("cpu"),
) -> TrajectoryBatch:
    policies = torch.zeros(B, T - 1, 64, 64, device=device)
    for b in range(B):
        for t in range(T - 1):
            policies[b, t, 12, 28] = 1.0
    return TrajectoryBatch(
        boards=torch.randn(B, T, C, 8, 8, device=device),
        actions_from=torch.randint(0, 64, (B, T - 1), device=device),
        actions_to=torch.randint(0, 64, (B, T - 1), device=device),
        policies=policies,
        values=torch.tensor([[1.0, 0.0, 0.0]] * B, device=device),
        rewards=torch.ones(B, T - 1, device=device),
    )


class TestPhase2Trainer:
    @pytest.fixture
    def trainer(self, device: torch.device) -> Phase2Trainer:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS, ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        world_model = ChessWorldModel(
            d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS, ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        diffusion = ChessDiffusionModule(
            d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        consistency = ChessConsistencyProjector(d_s=SMALL_D_S).to(device)
        schedule = CosineNoiseSchedule(
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        loss_fn = ChessLossComputer()
        return Phase2Trainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            world_model=world_model,
            diffusion=diffusion,
            consistency=consistency,
            schedule=schedule,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
        )

    def test_train_step_returns_loss_and_breakdown(
        self, trainer: Phase2Trainer, device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_breakdown_contains_all_6_terms(
        self, trainer: Phase2Trainer, device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        _, breakdown = trainer.train_step(batch)
        for key in (
            "policy", "value", "diffusion",
            "state", "reward", "consistency",
        ):
            assert key in breakdown, f"Missing '{key}' in breakdown"
        assert "grad_norm" in breakdown

    def test_encoder_is_frozen(
        self, trainer: Phase2Trainer, device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        trainer.train_step(batch)
        for p in trainer.encoder.parameters():
            assert not p.requires_grad

    def test_backbone_lr_lower_than_head_lr(
        self, trainer: Phase2Trainer,
    ) -> None:
        groups = trainer.optimizer.param_groups
        backbone_lr = groups[0]["lr"]  # backbone
        head_lr = groups[1]["lr"]       # policy head
        assert backbone_lr < head_lr

    def test_loss_is_finite(
        self, trainer: Phase2Trainer, device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        loss, _ = trainer.train_step(batch)
        assert loss == loss  # NaN check
        assert loss < float("inf")

    def test_loss_decreases_over_steps(
        self, trainer: Phase2Trainer, device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        losses = [trainer.train_step(batch)[0] for _ in range(30)]
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg

    def test_curriculum_advances(self, trainer: Phase2Trainer) -> None:
        initial = trainer.current_max_steps
        for _ in range(100):
            trainer.advance_curriculum()
        assert trainer.current_max_steps > initial
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestPhase2Trainer -v`

Expected: FAIL (ImportError: cannot import name 'Phase2Trainer')

**Step 3: Write minimal implementation**

Add to `src/denoisr/training/phase2_trainer.py` (after TrajectoryBatch):

```python
import torch
from torch import Tensor, nn
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.amp import autocast  # type: ignore[attr-defined]
from torch.nn import functional as F

from denoisr.nn.diffusion import CosineNoiseSchedule
from denoisr.training.loss import ChessLossComputer


class Phase2Trainer:
    """Unified Phase 2 trainer with all 6 loss terms.

    Freezes the encoder, trains backbone at reduced LR, and trains
    policy/value heads, world model, diffusion, and consistency
    projector at full LR. Each step computes policy, value, diffusion,
    world model state, world model reward, and consistency losses.
    """

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        world_model: nn.Module,
        diffusion: nn.Module,
        consistency: nn.Module,
        schedule: CosineNoiseSchedule,
        loss_fn: ChessLossComputer,
        lr: float = 1e-4,
        device: torch.device | None = None,
        max_grad_norm: float = 1.0,
        encoder_lr_multiplier: float = 0.3,
        weight_decay: float = 1e-4,
        curriculum_initial_fraction: float = 0.25,
        curriculum_growth: float = 1.02,
    ) -> None:
        self.encoder = encoder
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head
        self.world_model = world_model
        self.diffusion = diffusion
        self.consistency = consistency
        self.schedule = schedule
        self.loss_fn = loss_fn
        self.device = device or torch.device("cpu")
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler(
            "cuda", enabled=(self.device.type == "cuda"),
        )
        self._autocast_device = (
            self.device.type
            if self.device.type in ("cuda", "cpu")
            else "cpu"
        )
        self._autocast_enabled = self.device.type == "cuda"

        # Freeze encoder
        for p in encoder.parameters():
            p.requires_grad_(False)

        # Single optimizer with differential learning rates
        param_groups = [
            {
                "params": list(backbone.parameters()),
                "lr": lr * encoder_lr_multiplier,
            },
            {"params": list(policy_head.parameters()), "lr": lr},
            {"params": list(value_head.parameters()), "lr": lr},
            {"params": list(world_model.parameters()), "lr": lr},
            {"params": list(diffusion.parameters()), "lr": lr},
            {"params": list(consistency.parameters()), "lr": lr},
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=weight_decay,
        )

        # Diffusion curriculum
        self._curriculum_max_steps = schedule.num_timesteps
        initial_steps = max(
            1, int(schedule.num_timesteps * curriculum_initial_fraction),
        )
        self._current_max_steps_f = float(initial_steps)
        self._current_max_steps = initial_steps
        self._curriculum_growth = curriculum_growth

    def train_step(
        self, batch: TrajectoryBatch,
    ) -> tuple[float, dict[str, float]]:
        """Run one training step on a TrajectoryBatch.

        Returns (total_loss, breakdown_dict) where breakdown_dict
        contains all 6 loss terms plus grad_norm.
        """
        B, T = batch.boards.shape[:2]
        Tm1 = T - 1

        self.encoder.eval()
        self.backbone.train()
        self.policy_head.train()
        self.value_head.train()
        self.world_model.train()
        self.diffusion.train()
        self.consistency.train()

        with autocast(
            self._autocast_device, enabled=self._autocast_enabled,
        ):
            # 1. Encode all boards (frozen encoder)
            with torch.no_grad():
                flat_boards = batch.boards.reshape(
                    B * T, *batch.boards.shape[2:],
                )
                latent_flat = self.encoder(flat_boards)
                latent = latent_flat.reshape(B, T, 64, -1)

            d_s = latent.shape[-1]

            # 2. Backbone processes all T positions
            features_flat = self.backbone(
                latent.reshape(B * T, 64, d_s),
            )
            features = features_flat.reshape(B, T, 64, d_s)

            # 3. Policy + value on first T-1 positions
            feat_sv = features[:, :Tm1].reshape(B * Tm1, 64, d_s)
            pred_policy = self.policy_head(feat_sv)
            pred_value, _pred_ply = self.value_head(feat_sv)

            target_policy = batch.policies.reshape(B * Tm1, 64, 64)
            target_value = (
                batch.values
                .unsqueeze(1)
                .expand(B, Tm1, 3)
                .reshape(B * Tm1, 3)
            )

            # 4. World model: predict next states + rewards
            wm_states = latent[:, :Tm1]
            pred_next, pred_reward = self.world_model(
                wm_states, batch.actions_from, batch.actions_to,
            )
            actual_next = latent[:, 1:T].detach()

            state_loss = F.mse_loss(pred_next, actual_next)
            reward_loss = F.mse_loss(pred_reward, batch.rewards)

            # 5. Diffusion: noise prediction on random future
            cond = latent[:, 0]
            target_idx = torch.randint(
                1, T, (B,), device=self.device,
            )
            diff_target = torch.stack(
                [latent[b, target_idx[b]] for b in range(B)],
            )
            t = torch.randint(
                0, self._current_max_steps, (B,),
                device=self.device,
            )
            noise = torch.randn_like(diff_target)
            noisy_target = self.schedule.q_sample(
                diff_target, t, noise,
            )
            predicted_noise = self.diffusion(noisy_target, t, cond)
            diffusion_loss = F.mse_loss(predicted_noise, noise)

            # 6. Consistency: SimSiam on predicted vs actual
            pred_next_flat = pred_next.reshape(B * Tm1, 64, d_s)
            actual_next_flat = actual_next.reshape(B * Tm1, 64, d_s)
            proj_pred = self.consistency(pred_next_flat)
            with torch.no_grad():
                proj_actual = self.consistency(actual_next_flat)
            consistency_loss = -F.cosine_similarity(
                proj_pred, proj_actual, dim=-1,
            ).mean()

            # 7. Combine through ChessLossComputer
            total, breakdown = self.loss_fn.compute(
                pred_policy, pred_value,
                target_policy, target_value,
                consistency_loss=consistency_loss,
                diffusion_loss=diffusion_loss,
                reward_loss=reward_loss,
                state_loss=state_loss,
            )

        self.optimizer.zero_grad()
        self.scaler.scale(total).backward()  # type: ignore[no-untyped-call]
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(
            [
                p
                for group in self.optimizer.param_groups
                for p in group["params"]
            ],
            self.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        breakdown["grad_norm"] = total_norm.item()
        return total.item(), breakdown

    @property
    def current_max_steps(self) -> int:
        """Current curriculum diffusion step limit."""
        return self._current_max_steps

    def advance_curriculum(self) -> None:
        """Call once per epoch to grow diffusion step difficulty."""
        self._current_max_steps_f = min(
            float(self._curriculum_max_steps),
            self._current_max_steps_f * self._curriculum_growth,
        )
        self._current_max_steps = int(self._current_max_steps_f)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py -v`

Expected: All tests PASS (TrajectoryBatch + Phase2Trainer)

**Step 5: Commit**

```bash
git add src/denoisr/training/phase2_trainer.py tests/test_training/test_phase2_trainer.py
git commit -m "feat: add Phase2Trainer with 6-loss training step"
```

---

## Task 4: Enriched extract_trajectories

The current `extract_trajectories` in `scripts/train_phase2.py` returns `list[torch.Tensor]` with only board tensors. Enrich it to return a `TrajectoryBatch` with boards, actions, policy targets, value targets, and per-move rewards.

**Files:**
- Modify: `src/denoisr/scripts/train_phase2.py:49-84` (replace extract_trajectories)
- Modify: `tests/test_training/test_phase2_trainer.py` (add extraction test)

**Step 1: Write the failing test**

Add to `tests/test_training/test_phase2_trainer.py`:

```python
from pathlib import Path

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.scripts.train_phase2 import extract_trajectories


class TestExtractTrajectories:
    @pytest.fixture
    def pgn_file(self, tmp_path: Path) -> Path:
        pgn = tmp_path / "test.pgn"
        pgn.write_text(
            '[Event "Test"]\n'
            '[Result "1-0"]\n'
            "\n"
            "1. e2e4 e7e5 2. g1f3 b8c6 3. f1b5 a7a6 "
            "4. b5a4 g8f6 5. e1g1 f8e7 *\n\n"
        )
        return pgn

    def test_returns_trajectory_batch(self, pgn_file: Path) -> None:
        encoder = SimpleBoardEncoder()
        batch = extract_trajectories(
            pgn_file, encoder, seq_len=3, max_trajectories=100,
        )
        assert isinstance(batch, TrajectoryBatch)
        assert batch.boards.shape[1] == 3
        assert batch.actions_from.shape[1] == 2
        assert batch.actions_to.shape[1] == 2
        assert batch.policies.shape[1] == 2
        assert batch.values.shape[1] == 3

    def test_policy_targets_are_one_hot(
        self, pgn_file: Path,
    ) -> None:
        encoder = SimpleBoardEncoder()
        batch = extract_trajectories(
            pgn_file, encoder, seq_len=3, max_trajectories=100,
        )
        for i in range(batch.policies.shape[0]):
            for t in range(batch.policies.shape[1]):
                assert batch.policies[i, t].sum().item() == (
                    pytest.approx(1.0)
                )

    def test_values_are_valid_wdl(self, pgn_file: Path) -> None:
        encoder = SimpleBoardEncoder()
        batch = extract_trajectories(
            pgn_file, encoder, seq_len=3, max_trajectories=100,
        )
        for i in range(batch.values.shape[0]):
            assert batch.values[i].sum().item() == (
                pytest.approx(1.0)
            )

    def test_rewards_match_result(self, pgn_file: Path) -> None:
        encoder = SimpleBoardEncoder()
        batch = extract_trajectories(
            pgn_file, encoder, seq_len=3, max_trajectories=100,
        )
        # Game result is 1-0, so all rewards are +1 or -1
        for r in batch.rewards.flatten():
            assert abs(r.item()) == pytest.approx(1.0)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestExtractTrajectories -v`

Expected: FAIL (extract_trajectories returns list[Tensor], not TrajectoryBatch)

**Step 3: Write minimal implementation**

Replace `extract_trajectories` in `src/denoisr/scripts/train_phase2.py`:

```python
from denoisr.training.phase2_trainer import TrajectoryBatch


def extract_trajectories(
    pgn_path: Path,
    encoder: SimpleBoardEncoder | ExtendedBoardEncoder,
    seq_len: int,
    max_trajectories: int,
) -> TrajectoryBatch:
    """Extract enriched consecutive board-state trajectories from PGN."""
    streamer = SimplePGNStreamer()

    all_boards: list[torch.Tensor] = []
    all_actions_from: list[torch.Tensor] = []
    all_actions_to: list[torch.Tensor] = []
    all_policies: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []

    pbar = tqdm(
        total=max_trajectories, desc="Extracting trajectories",
        unit="traj", smoothing=0.3,
    )

    for record in streamer.stream(pgn_path):
        if len(record.actions) < seq_len:
            continue

        # WDL from game result
        if record.result == 1.0:
            wdl = torch.tensor([1.0, 0.0, 0.0])
        elif record.result == 0.0:
            wdl = torch.tensor([0.0, 0.0, 1.0])
        else:
            wdl = torch.tensor([0.0, 1.0, 0.0])

        # +1 White wins, -1 Black wins, 0 draw
        result_signal = 2.0 * record.result - 1.0

        board = chess.Board()
        boards: list[torch.Tensor] = [encoder.encode(board).data]
        from_sqs: list[int] = []
        to_sqs: list[int] = []

        for action in record.actions:
            from_sqs.append(action.from_square)
            to_sqs.append(action.to_square)
            move = chess.Move(
                action.from_square, action.to_square,
                action.promotion,
            )
            board.push(move)
            boards.append(encoder.encode(board).data)

        for start in range(0, len(boards) - seq_len, seq_len):
            chunk_boards = boards[start : start + seq_len]
            chunk_from = from_sqs[start : start + seq_len - 1]
            chunk_to = to_sqs[start : start + seq_len - 1]

            policies = torch.zeros(seq_len - 1, 64, 64)
            for j in range(seq_len - 1):
                policies[j, chunk_from[j], chunk_to[j]] = 1.0

            rewards = torch.zeros(seq_len - 1)
            for j in range(seq_len - 1):
                game_pos = start + j
                side_sign = 1.0 if game_pos % 2 == 0 else -1.0
                rewards[j] = result_signal * side_sign

            all_boards.append(torch.stack(chunk_boards))
            all_actions_from.append(
                torch.tensor(chunk_from, dtype=torch.long),
            )
            all_actions_to.append(
                torch.tensor(chunk_to, dtype=torch.long),
            )
            all_policies.append(policies)
            all_values.append(wdl)
            all_rewards.append(rewards)

            pbar.update(1)
            if len(all_boards) >= max_trajectories:
                break
        if len(all_boards) >= max_trajectories:
            break

    pbar.close()
    if not all_boards:
        raise ValueError("No valid trajectories extracted from PGN")

    return TrajectoryBatch(
        boards=torch.stack(all_boards),
        actions_from=torch.stack(all_actions_from),
        actions_to=torch.stack(all_actions_to),
        policies=torch.stack(all_policies),
        values=torch.stack(all_values),
        rewards=torch.stack(all_rewards),
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestExtractTrajectories -v`

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/denoisr/scripts/train_phase2.py tests/test_training/test_phase2_trainer.py
git commit -m "feat: enrich extract_trajectories with actions, policies, values, rewards"
```

---

## Task 5: Phase 2 gate function

After training completes, automatically compare diffusion-conditioned vs single-step policy accuracy on holdout data. Delta > 5pp means Phase 2 passes.

**Files:**
- Modify: `src/denoisr/training/phase2_trainer.py` (add `evaluate_phase2_gate`)
- Modify: `tests/test_training/test_phase2_trainer.py` (add gate tests)

**Step 1: Write the failing test**

Add to `tests/test_training/test_phase2_trainer.py`:

```python
from denoisr.training.phase2_trainer import evaluate_phase2_gate


class TestPhase2Gate:
    def test_returns_three_floats(self, device: torch.device) -> None:
        encoder = ChessEncoder(
            num_planes=12, d_s=SMALL_D_S,
        ).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS, ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        diffusion = ChessDiffusionModule(
            d_s=SMALL_D_S, num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        schedule = CosineNoiseSchedule(
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)

        boards = torch.randn(8, 12, 8, 8, device=device)
        target_from = torch.randint(0, 64, (8,), device=device)
        target_to = torch.randint(0, 64, (8,), device=device)

        single_acc, diff_acc, delta = evaluate_phase2_gate(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            diffusion=diffusion,
            schedule=schedule,
            boards=boards,
            target_from=target_from,
            target_to=target_to,
            device=device,
        )
        assert isinstance(single_acc, float)
        assert isinstance(diff_acc, float)
        assert isinstance(delta, float)
        assert 0.0 <= single_acc <= 1.0
        assert 0.0 <= diff_acc <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestPhase2Gate -v`

Expected: FAIL (ImportError: cannot import name 'evaluate_phase2_gate')

**Step 3: Write minimal implementation**

Add to `src/denoisr/training/phase2_trainer.py`:

```python
def evaluate_phase2_gate(
    encoder: nn.Module,
    backbone: nn.Module,
    policy_head: nn.Module,
    diffusion: nn.Module,
    schedule: CosineNoiseSchedule,
    boards: Tensor,
    target_from: Tensor,
    target_to: Tensor,
    device: torch.device,
    num_diff_steps: int = 10,
) -> tuple[float, float, float]:
    """Compare single-step vs diffusion-conditioned accuracy.

    Returns (single_accuracy, diffusion_accuracy, delta_pp).
    """
    encoder.eval()
    backbone.eval()
    policy_head.eval()
    diffusion.eval()

    with torch.no_grad():
        latent = encoder(boards)

        # Single-step accuracy
        features = backbone(latent)
        logits = policy_head(features)
        flat_logits = logits.reshape(logits.shape[0], -1)
        pred_flat = flat_logits.argmax(dim=-1)
        pred_from = pred_flat // 64
        pred_to = pred_flat % 64
        single_correct = (
            (pred_from == target_from) & (pred_to == target_to)
        ).float().mean().item()

        # Diffusion-conditioned accuracy (DDIM denoising)
        x = torch.randn_like(latent)
        num_ts = schedule.num_timesteps
        step_size = max(1, num_ts // num_diff_steps)

        for i in range(num_diff_steps):
            t_val = max(0, num_ts - 1 - i * step_size)
            t = torch.full(
                (boards.shape[0],), t_val, device=device,
            )
            noise_pred = diffusion(x, t, latent)

            ab_t = schedule.alpha_bar[t_val]
            x0_pred = (
                (x - (1 - ab_t).sqrt() * noise_pred) / ab_t.sqrt()
            )

            t_prev = max(0, t_val - step_size)
            if t_prev > 0:
                ab_prev = schedule.alpha_bar[t_prev]
                x = (
                    ab_prev.sqrt() * x0_pred
                    + (1 - ab_prev).sqrt() * noise_pred
                )
            else:
                x = x0_pred

        fused = (latent + x) / 2
        features_diff = backbone(fused)
        logits_diff = policy_head(features_diff)
        flat_diff = logits_diff.reshape(logits_diff.shape[0], -1)
        pred_flat_diff = flat_diff.argmax(dim=-1)
        pred_from_diff = pred_flat_diff // 64
        pred_to_diff = pred_flat_diff % 64
        diff_correct = (
            (pred_from_diff == target_from)
            & (pred_to_diff == target_to)
        ).float().mean().item()

    delta = (diff_correct - single_correct) * 100.0
    return single_correct, diff_correct, delta
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_phase2_trainer.py::TestPhase2Gate -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/denoisr/training/phase2_trainer.py tests/test_training/test_phase2_trainer.py
git commit -m "feat: add evaluate_phase2_gate for automated Phase 2 assessment"
```

---

## Task 6: Wire Phase2Trainer into train_phase2.py

Replace the DiffusionTrainer-based training loop with Phase2Trainer, update data loading for enriched TrajectoryBatch, and add automated gate at the end.

**Files:**
- Modify: `src/denoisr/scripts/train_phase2.py:87-292` (rewrite main)

**Step 1: No isolated test** -- this is a script integration task. Verified by full test suite.

**Step 2: Rewrite main() in `src/denoisr/scripts/train_phase2.py`**

Key changes from original:

1. Replace `DiffusionTrainer` import with `Phase2Trainer, TrajectoryBatch, evaluate_phase2_gate`
2. Add `ChessLossComputer` import
3. Create `ChessLossComputer` with weights from `TrainingConfig`
4. Create `Phase2Trainer` instead of `DiffusionTrainer`
5. `extract_trajectories` now returns `TrajectoryBatch`
6. `TensorDataset` created from all 6 TrajectoryBatch fields
7. Training loop unpacks 6 tensors per batch, reconstructs `TrajectoryBatch`
8. Add 95%/5% train/holdout split before training
9. HarmonyDream coefficients logged when enabled
10. Gate via `evaluate_phase2_gate` at the end

Update imports at top of file:
- Remove: `from denoisr.training.diffusion_trainer import DiffusionTrainer`
- Add: `from denoisr.training.phase2_trainer import Phase2Trainer, TrajectoryBatch, evaluate_phase2_gate`
- Add: `from denoisr.training.loss import ChessLossComputer`

The full rewritten `main()`:

```python
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: World model + diffusion bootstrapping"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Phase 1 checkpoint path"
    )
    parser.add_argument(
        "--pgn", required=True, help="PGN file for trajectory extraction"
    )
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--max-trajectories", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="outputs/phase2.pt")
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="TensorBoard run name (default: timestamp)",
    )
    add_model_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = detect_device()
    tcfg = training_config_from_args(args)
    use_tqdm = args.tqdm
    log.info("device=%s", device)

    # --- Load Phase 1 ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    cfg = resolve_gradient_checkpointing(cfg, args, device)
    log.info(
        "checkpoint loaded  d_s=%d  layers=%d", cfg.d_s, cfg.num_layers,
    )

    encoder = build_encoder(cfg).to(device)
    backbone = build_backbone(cfg).to(device)
    policy_head = build_policy_head(cfg).to(device)
    value_head = build_value_head(cfg).to(device)

    encoder.load_state_dict(state["encoder"])
    backbone.load_state_dict(state["backbone"])
    policy_head.load_state_dict(state["policy_head"])
    value_head.load_state_dict(state["value_head"])

    # --- Build Phase 2 modules ---
    world_model = build_world_model(cfg).to(device)
    diffusion_mod = build_diffusion(cfg).to(device)
    consistency = build_consistency(cfg).to(device)
    schedule = build_schedule(cfg).to(device)

    encoder = maybe_compile(encoder, device)
    backbone = maybe_compile(backbone, device)
    diffusion_mod = maybe_compile(diffusion_mod, device)

    loss_fn = ChessLossComputer(
        policy_weight=tcfg.policy_weight,
        value_weight=tcfg.value_weight,
        consistency_weight=tcfg.consistency_weight,
        diffusion_weight=tcfg.diffusion_weight,
        reward_weight=tcfg.reward_weight,
        ply_weight=tcfg.ply_weight,
        use_harmony_dream=tcfg.use_harmony_dream,
        harmony_ema_decay=tcfg.harmony_ema_decay,
    )

    trainer = Phase2Trainer(
        encoder=encoder,
        backbone=backbone,
        policy_head=policy_head,
        value_head=value_head,
        world_model=world_model,
        diffusion=diffusion_mod,
        consistency=consistency,
        schedule=schedule,
        loss_fn=loss_fn,
        lr=args.lr,
        device=device,
        max_grad_norm=tcfg.max_grad_norm,
        encoder_lr_multiplier=tcfg.encoder_lr_multiplier,
        weight_decay=tcfg.weight_decay,
        curriculum_initial_fraction=tcfg.curriculum_initial_fraction,
        curriculum_growth=tcfg.curriculum_growth,
    )

    # --- Extract enriched trajectories ---
    board_encoder = build_board_encoder(cfg)
    trajectory_data = extract_trajectories(
        Path(args.pgn), board_encoder,
        args.seq_len, args.max_trajectories,
    )
    N = trajectory_data.boards.shape[0]
    log.info("trajectories=%d  seq_len=%d", N, args.seq_len)

    # --- Train/holdout split (95/5) ---
    n_holdout = max(1, int(N * 0.05))
    n_train = N - n_holdout

    train_dataset = TensorDataset(
        trajectory_data.boards[:n_train],
        trajectory_data.actions_from[:n_train],
        trajectory_data.actions_to[:n_train],
        trajectory_data.policies[:n_train],
        trajectory_data.values[:n_train],
        trajectory_data.rewards[:n_train],
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=tcfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    monitor = ResourceMonitor()
    best_loss = float("inf")
    bs = args.batch_size

    with TrainingLogger(Path("logs"), run_name=args.run_name) as logger:
        logger.log_hparams(
            {
                "lr": args.lr,
                "batch_size": bs,
                "epochs": args.epochs,
                "seq_len": args.seq_len,
                "max_trajectories": args.max_trajectories,
                "d_s": cfg.d_s,
                "num_layers": cfg.num_layers,
                "diffusion_layers": cfg.diffusion_layers,
                "num_timesteps": cfg.num_timesteps,
                "max_grad_norm": tcfg.max_grad_norm,
                "harmony_dream": tcfg.use_harmony_dream,
            },
            {"best_total_loss": float("inf")},
        )

        global_step = 0

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.monotonic()
            monitor.reset()
            step_losses: list[float] = []
            step_grad_norms: list[float] = []
            data_time = 0.0
            compute_time = 0.0

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False, smoothing=0.3,
                disable=not use_tqdm,
            )
            data_start = time.monotonic()
            for boards, af, at, pols, vals, rews in pbar:
                data_time += time.monotonic() - data_start

                batch = TrajectoryBatch(
                    boards=boards.to(device, non_blocking=True),
                    actions_from=af.to(device, non_blocking=True),
                    actions_to=at.to(device, non_blocking=True),
                    policies=pols.to(device, non_blocking=True),
                    values=vals.to(device, non_blocking=True),
                    rewards=rews.to(device, non_blocking=True),
                )

                compute_start = time.monotonic()
                loss, breakdown = trainer.train_step(batch)
                compute_time += time.monotonic() - compute_start

                logger.log_train_step(global_step, loss, breakdown)
                step_losses.append(loss)
                step_grad_norms.append(
                    breakdown.get("grad_norm", 0.0),
                )
                if global_step % 100 == 0:
                    logger.log_gpu(global_step)
                    monitor.sample()
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix(loss=f"{loss:.4f}")
                data_start = time.monotonic()
            pbar.close()

            trainer.advance_curriculum()
            epoch_duration = time.monotonic() - epoch_start
            num_samples = len(train_dataset)
            avg_loss = epoch_loss / max(num_batches, 1)

            logger.log_diffusion(
                epoch, avg_loss, trainer.current_max_steps,
            )
            logger.log_epoch_timing(
                epoch, epoch_duration,
                num_samples / epoch_duration,
            )

            resource_metrics = monitor.summarize()
            logger.log_resource_metrics(epoch, resource_metrics)
            logger.log_training_dynamics(
                epoch, step_losses, step_grad_norms,
            )
            logger.log_pipeline_timing(
                epoch, data_time, compute_time,
            )

            total_time = data_time + compute_time
            summary: dict[str, str] = {
                "epoch": f"{epoch+1}/{args.epochs}",
                "total_loss": f"{avg_loss:.4f}",
                "curriculum_steps": str(trainer.current_max_steps),
                "grad_norm_avg": (
                    f"{sum(step_grad_norms)/len(step_grad_norms):.3f}"
                    if step_grad_norms else "n/a"
                ),
                "samples/s": f"{num_samples / epoch_duration:.0f}",
                "epoch_time": f"{epoch_duration:.1f}s",
                "data_pct": (
                    f"{data_time / total_time:.0%}"
                    if total_time > 0 else "0%"
                ),
            }
            if tcfg.use_harmony_dream:
                for k, v in loss_fn.get_coefficients().items():
                    summary[f"hd_{k}"] = f"{v:.3f}"
            logger.log_epoch_summary(summary)

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    Path(args.output), cfg,
                    encoder=encoder.state_dict(),
                    backbone=backbone.state_dict(),
                    policy_head=policy_head.state_dict(),
                    value_head=value_head.state_dict(),
                    world_model=world_model.state_dict(),
                    diffusion=diffusion_mod.state_dict(),
                    consistency=consistency.state_dict(),
                )

        # --- Phase 2 gate ---
        log.info(
            "Phase 2 gate on holdout (%d samples)...", n_holdout,
        )
        holdout_boards = trajectory_data.boards[
            n_train:, 0
        ].to(device)
        holdout_from = trajectory_data.actions_from[
            n_train:, 0
        ].to(device)
        holdout_to = trajectory_data.actions_to[
            n_train:, 0
        ].to(device)

        single_acc, diff_acc, delta = evaluate_phase2_gate(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            diffusion=diffusion_mod,
            schedule=schedule,
            boards=holdout_boards,
            target_from=holdout_from,
            target_to=holdout_to,
            device=device,
        )
        log.info(
            "Phase 2 gate: single=%.1f%%  diffusion=%.1f%%  "
            "delta=%.1fpp  threshold=%.1fpp",
            single_acc * 100, diff_acc * 100,
            delta, tcfg.phase2_gate,
        )
        if delta > tcfg.phase2_gate:
            log.info(
                "Phase 2 gate PASSED (delta %.1fpp > %.1fpp)",
                delta, tcfg.phase2_gate,
            )
        else:
            log.warning(
                "Phase 2 gate NOT PASSED (delta %.1fpp <= %.1fpp)."
                " Checkpoint saved -- user decides.",
                delta, tcfg.phase2_gate,
            )
```

Also update imports at top of file:
- Remove: `from denoisr.training.diffusion_trainer import DiffusionTrainer`
- Add: `from denoisr.training.phase2_trainer import Phase2Trainer, TrajectoryBatch, evaluate_phase2_gate`
- Add: `from denoisr.training.loss import ChessLossComputer`

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -x -q`

Expected: All tests PASS

**Step 4: Run linter and type checker**

```bash
uvx ruff check src/denoisr/scripts/train_phase2.py src/denoisr/training/phase2_trainer.py src/denoisr/training/loss.py
uv run --with mypy mypy --strict src/denoisr/training/phase2_trainer.py src/denoisr/training/loss.py
```

Expected: No errors

**Step 5: Commit**

```bash
git add src/denoisr/scripts/train_phase2.py
git commit -m "feat: wire Phase2Trainer into train_phase2.py with gate"
```

---

## Task 7: Final verification and cleanup

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -x -q
```

Expected: All tests PASS (previous 342 + new tests)

**Step 2: Run linter on all modified files**

```bash
uvx ruff check src/denoisr/training/loss.py src/denoisr/training/phase2_trainer.py src/denoisr/scripts/train_phase2.py
```

Expected: No errors

**Step 3: Run type checker on new files**

```bash
uv run --with mypy mypy --strict src/denoisr/training/phase2_trainer.py src/denoisr/training/loss.py
```

Expected: Success

**Step 4: Verify README alignment**

Check that the README Phase 2 description matches:
- 6-loss training: policy, value, diffusion, world model state, world model reward, consistency
- HarmonyDream balancing via `--harmony-dream`
- Automated gate (diffusion accuracy > single-step by >5pp)

**Step 5: Commit any final cleanup**

```bash
git add -A
git commit -m "chore: final verification for Phase 2 completion"
```

---

## Reference: Key file locations

| File | Purpose |
|---|---|
| `src/denoisr/training/phase2_trainer.py` | **New** -- TrajectoryBatch, Phase2Trainer, evaluate_phase2_gate |
| `src/denoisr/training/loss.py` | Modified -- added state_weight + "state" auxiliary |
| `src/denoisr/scripts/train_phase2.py` | Modified -- enriched extract, Phase2Trainer, gate |
| `tests/test_training/test_phase2_trainer.py` | **New** -- all Phase 2 tests |
| `tests/test_training/test_loss.py` | Modified -- state loss tests |

## Reference: Module interfaces

**ChessWorldModel.forward(states, action_from, action_to)**
- Input: `[B, T, 64, d_s]`, `[B, T]`, `[B, T]`
- Output: `([B, T, 64, d_s], [B, T])` -- predicted next states + rewards

**ChessConsistencyProjector.forward(x)**
- Input: `[B, 64, d_s]`
- Output: `[B, proj_dim]` -- mean-pooled + projected

**CosineNoiseSchedule.q_sample(x, t, noise)**
- Corrupts x at timestep t with given noise
- Uses `alpha_bar[t]` for noise level

**ChessLossComputer.compute(pred_policy, pred_value, target_policy, target_value, **aux)**
- Auxiliary kwargs: `consistency_loss`, `diffusion_loss`, `reward_loss`, `ply_loss`, `state_loss`
- Returns: `(total_loss, breakdown_dict)`
