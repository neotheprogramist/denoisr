from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.amp import autocast  # type: ignore[attr-defined]
from torch.nn import functional as F

from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule, DPMSolverPP
from denoisr.training.loss import ChessLossComputer


@dataclass(frozen=True)
class TrajectoryBatch:
    """Enriched trajectory data for Phase 2 training.

    Each trajectory contains T consecutive board states connected by T-1 actions.
    Boards are raw encoder outputs, not BoardTensor newtypes.
    """

    boards: Tensor  # [N, T, C, 8, 8]
    actions_from: Tensor  # [N, T-1] (int64)
    actions_to: Tensor  # [N, T-1] (int64)
    policies: Tensor  # [N, T-1, 64, 64] one-hot from played move
    values: Tensor  # [N, 3] WDL from game result
    rewards: Tensor  # [N, T-1] per-move reward signal
    legal_masks: Tensor | None = None  # [N, T-1, 64, 64] legal move mask

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
                    f"{name} time dim {actual[1]} != expected {Tm1} (boards T={T})"
                )
        if self.values.shape[0] != B:
            raise ValueError(
                f"values batch dim {self.values.shape[0]} != boards batch dim {B}"
            )
        if self.legal_masks is not None:
            lm = self.legal_masks
            if lm.shape[0] != B:
                raise ValueError(
                    f"legal_masks batch dim {lm.shape[0]} != boards batch dim {B}"
                )
            if lm.shape[1] != Tm1:
                raise ValueError(
                    f"legal_masks time dim {lm.shape[1]} != expected {Tm1} "
                    f"(boards T={T})"
                )


class Phase2Trainer:
    """Unified Phase 2 trainer with all 6 loss terms."""

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
        freeze_encoder: bool = True,
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
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self._autocast_device = (
            self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        )
        self._autocast_enabled = self.device.type == "cuda"

        # Freeze encoder for standard Phase 2 training. Phase 3 auxiliary
        # updates can disable this to avoid interfering with other optimizers.
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad_(False)

        # Single optimizer with differential learning rates
        param_groups = [
            {"params": list(backbone.parameters()), "lr": lr * encoder_lr_multiplier},
            {"params": list(policy_head.parameters()), "lr": lr},
            {"params": list(value_head.parameters()), "lr": lr},
            {"params": list(world_model.parameters()), "lr": lr},
            {"params": list(diffusion.parameters()), "lr": lr},
            {"params": list(consistency.parameters()), "lr": lr},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # Diffusion curriculum
        self._curriculum_max_steps = schedule.num_timesteps
        initial_steps = max(
            1, int(schedule.num_timesteps * curriculum_initial_fraction)
        )
        self._current_max_steps_f = float(initial_steps)
        self._current_max_steps = initial_steps
        self._curriculum_growth = curriculum_growth

    def train_step(self, batch: TrajectoryBatch) -> tuple[float, dict[str, float]]:
        B, T = batch.boards.shape[:2]
        Tm1 = T - 1

        self.encoder.eval()
        self.backbone.train()
        self.policy_head.train()
        self.value_head.train()
        self.world_model.train()
        self.diffusion.train()
        self.consistency.train()

        with autocast(self._autocast_device, enabled=self._autocast_enabled):
            # 1. Encode all boards (frozen encoder)
            with torch.no_grad():
                flat_boards = batch.boards.reshape(B * T, *batch.boards.shape[2:])
                latent_flat = self.encoder(flat_boards)
                latent = latent_flat.reshape(B, T, 64, -1)

            d_s = latent.shape[-1]

            # 2. Backbone processes all T positions
            features_flat = self.backbone(latent.reshape(B * T, 64, d_s))
            features = features_flat.reshape(B, T, 64, d_s)

            # 3. Policy + value on first T-1 positions
            feat_sv = features[:, :Tm1].reshape(B * Tm1, 64, d_s)
            pred_policy = self.policy_head(feat_sv)
            pred_value, _pred_ply = self.value_head(feat_sv)

            target_policy = batch.policies.reshape(B * Tm1, 64, 64)
            target_legal_mask = (
                batch.legal_masks.reshape(B * Tm1, 64, 64)
                if batch.legal_masks is not None
                else None
            )
            target_value = (
                batch.values.unsqueeze(1).expand(B, Tm1, 3).reshape(B * Tm1, 3)
            )

            # 4. World model: predict next states + rewards
            wm_states = latent[:, :Tm1]
            pred_next, pred_reward = self.world_model(
                wm_states,
                batch.actions_from,
                batch.actions_to,
            )
            actual_next = latent[:, 1:T].detach()

            state_loss = F.mse_loss(pred_next, actual_next)
            reward_loss = F.mse_loss(pred_reward, batch.rewards)

            # 5. Diffusion: v-prediction on random future
            cond = latent[:, 0]
            target_idx = torch.randint(1, T, (B,), device=self.device)
            diff_target = torch.stack([latent[b, target_idx[b]] for b in range(B)])
            t = torch.randint(0, self._current_max_steps, (B,), device=self.device)
            noise = torch.randn_like(diff_target)
            noisy_target = self.schedule.q_sample(diff_target, t, noise)
            v_target = self.schedule.compute_v_target(diff_target, noise, t)
            v_pred = self.diffusion(noisy_target, t, cond)
            diffusion_loss = F.mse_loss(v_pred, v_target)

            # 6. Consistency: SimSiam on predicted vs actual
            pred_next_flat = pred_next.reshape(B * Tm1, 64, d_s)
            actual_next_flat = actual_next.reshape(B * Tm1, 64, d_s)
            proj_pred = self.consistency(pred_next_flat)
            with torch.no_grad():
                proj_actual = self.consistency(actual_next_flat)
            consistency_loss = -F.cosine_similarity(
                proj_pred,
                proj_actual,
                dim=-1,
            ).mean()

            # 7. Combine through ChessLossComputer
            total, breakdown = self.loss_fn.compute(
                pred_policy,
                pred_value,
                target_policy,
                target_value,
                policy_legal_mask=target_legal_mask,
                consistency_loss=consistency_loss,
                diffusion_loss=diffusion_loss,
                reward_loss=reward_loss,
                state_loss=state_loss,
            )

        self.optimizer.zero_grad()
        self.scaler.scale(total).backward()  # type: ignore[no-untyped-call]
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        breakdown["grad_norm"] = total_norm.item()
        return total.item(), breakdown

    @property
    def current_max_steps(self) -> int:
        return self._current_max_steps

    def advance_curriculum(self) -> None:
        self._current_max_steps_f = min(
            float(self._curriculum_max_steps),
            self._current_max_steps_f * self._curriculum_growth,
        )
        self._current_max_steps = int(self._current_max_steps_f)


def evaluate_phase2_gate(
    encoder: nn.Module,
    backbone: nn.Module,
    policy_head: nn.Module,
    diffusion: ChessDiffusionModule,
    schedule: CosineNoiseSchedule,
    boards: Tensor,
    target_from: Tensor,
    target_to: Tensor,
    device: torch.device,
    num_diff_steps: int = 10,
    legal_mask: Tensor | None = None,
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
        if legal_mask is not None:
            flat_legal = legal_mask.reshape(logits.shape[0], -1).to(
                device=device, dtype=torch.bool
            )
            has_legal = flat_legal.any(dim=-1, keepdim=True)
            flat_logits = flat_logits.masked_fill(~flat_legal, float("-inf"))
            # Guard all-zero legal rows.
            flat_logits = flat_logits.masked_fill(
                ~has_legal.expand_as(flat_logits), 0.0
            )
        pred_flat = flat_logits.argmax(dim=-1)
        pred_from = pred_flat // 64
        pred_to = pred_flat % 64
        single_correct = (
            ((pred_from == target_from) & (pred_to == target_to)).float().mean().item()
        )

        # Diffusion-conditioned accuracy (DPM-Solver++)
        solver = DPMSolverPP(schedule, num_steps=num_diff_steps)
        x = solver.sample(diffusion, latent.shape, latent, device)
        fused = diffusion.fuse(latent, x)
        features_diff = backbone(fused)
        logits_diff = policy_head(features_diff)
        flat_diff = logits_diff.reshape(logits_diff.shape[0], -1)
        if legal_mask is not None:
            flat_legal = legal_mask.reshape(logits_diff.shape[0], -1).to(
                device=device, dtype=torch.bool
            )
            has_legal = flat_legal.any(dim=-1, keepdim=True)
            flat_diff = flat_diff.masked_fill(~flat_legal, float("-inf"))
            flat_diff = flat_diff.masked_fill(~has_legal.expand_as(flat_diff), 0.0)
        pred_flat_diff = flat_diff.argmax(dim=-1)
        pred_from_diff = pred_flat_diff // 64
        pred_to_diff = pred_flat_diff % 64
        diff_correct = (
            ((pred_from_diff == target_from) & (pred_to_diff == target_to))
            .float()
            .mean()
            .item()
        )

    delta = (diff_correct - single_correct) * 100.0
    return single_correct, diff_correct, delta
