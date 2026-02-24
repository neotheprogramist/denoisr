import math
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.amp import autocast  # type: ignore[attr-defined]

from denoisr.training.grokfast import GrokfastFilter
from denoisr.training.loss import ChessLossComputer
from denoisr.types import TrainingExample


class SupervisedTrainer:
    """Supervised training loop for Phase 1.

    Takes batches of TrainingExamples (board tensor + policy/value targets)
    and updates the encoder, backbone, policy head, and value head.
    """

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
        max_grad_norm: float = 1.0,
        weight_decay: float = 1e-4,
        encoder_lr_multiplier: float = 0.3,
        min_lr: float = 1e-6,
        grokfast_filter: GrokfastFilter | None = None,
        use_warm_restarts: bool = False,
    ) -> None:
        self.encoder = encoder
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head
        self.loss_fn = loss_fn
        self.device = device or torch.device("cpu")
        self._grokfast_filter = grokfast_filter
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self._autocast_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        self._autocast_enabled = self.device.type == "cuda"

        param_groups = [
            {"params": list(encoder.parameters()), "lr": lr * encoder_lr_multiplier},
            {"params": list(backbone.parameters()), "lr": lr * encoder_lr_multiplier},
            {"params": list(policy_head.parameters()), "lr": lr},
            {"params": list(value_head.parameters()), "lr": lr},
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=weight_decay
        )
        self._params: list[torch.nn.Parameter] = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
        ]

        self._warmup_epochs = warmup_epochs
        self._base_lrs: list[float] = [float(g["lr"]) for g in param_groups]  # type: ignore[arg-type]
        # Start at 1/N of peak LR; warmup will ramp up from here
        for g, base_lr in zip(param_groups, self._base_lrs):
            g["lr"] = base_lr / max(self._warmup_epochs, 1)
        if use_warm_restarts:
            self._scheduler: torch.optim.lr_scheduler.LRScheduler = (
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=20, T_mult=2, eta_min=min_lr,
                )
            )
        else:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr,
            )
        self._scheduler.base_lrs = list(self._base_lrs)
        self._epoch = 0

    def _has_nonfinite_gradients(self) -> bool:
        for param in self._params:
            grad = param.grad
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                return True
        return False

    def _handle_overflow(self, breakdown: dict[str, float | bool], batch_top1: float) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            new_scale = max(float(self.scaler.get_scale()) / 2.0, 1.0)
            self.scaler.update(new_scale)
        if self._grokfast_filter is not None:
            self._grokfast_filter.reset()
        breakdown["grad_norm"] = float("nan")
        breakdown["overflow"] = True
        breakdown["batch_top1"] = batch_top1

    def _forward_backward(
        self,
        boards: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> tuple[float, dict[str, float | bool]]:
        """Core training step on device-resident tensors."""
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

        with torch.no_grad():
            B = boards.shape[0]
            pf = pred_policy.detach().reshape(B, -1)
            tf = target_policies.reshape(B, -1)
            mask = tf > 0
            masked = pf.masked_fill(~mask, float("-inf"))
            batch_top1 = (masked.argmax(-1) == tf.argmax(-1)).float().mean().item()

        if not torch.isfinite(total_loss):
            self._handle_overflow(breakdown, batch_top1)
            return float("nan"), breakdown

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()  # type: ignore[no-untyped-call]
        self.scaler.unscale_(self.optimizer)
        if self._has_nonfinite_gradients():
            self._handle_overflow(breakdown, batch_top1)
            return total_loss.item(), breakdown
        if self._grokfast_filter is not None:
            for module in [
                self.encoder,
                self.backbone,
                self.policy_head,
                self.value_head,
            ]:
                self._grokfast_filter.apply(module)
            if self._has_nonfinite_gradients():
                self._handle_overflow(breakdown, batch_top1)
                return total_loss.item(), breakdown
        total_norm = torch.nn.utils.clip_grad_norm_(
            self._params,
            self.max_grad_norm,
        )
        grad_norm = total_norm.item()
        if not math.isfinite(grad_norm):
            self._handle_overflow(breakdown, batch_top1)
            return total_loss.item(), breakdown

        self.scaler.step(self.optimizer)
        self.scaler.update()

        breakdown["grad_norm"] = grad_norm
        breakdown["overflow"] = False
        breakdown["batch_top1"] = batch_top1
        return total_loss.item(), breakdown

    def train_step(
        self, batch: list[TrainingExample]
    ) -> tuple[float, dict[str, float | bool]]:
        boards = torch.stack([ex.board.data for ex in batch]).to(self.device)
        target_policies = torch.stack([ex.policy.data for ex in batch]).to(
            self.device
        )
        target_values = torch.tensor(
            [[ex.value.win, ex.value.draw, ex.value.loss] for ex in batch],
            dtype=torch.float32,
            device=self.device,
        )
        return self._forward_backward(boards, target_policies, target_values)

    def train_step_tensors(
        self,
        boards: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> tuple[float, dict[str, float | bool]]:
        """Train step accepting pre-stacked tensors (from DataLoader)."""
        return self._forward_backward(
            boards.to(self.device, non_blocking=True),
            target_policies.to(self.device, non_blocking=True),
            target_values.to(self.device, non_blocking=True),
        )

    def scheduler_step(self) -> None:
        self._epoch += 1
        if self._epoch <= self._warmup_epochs:
            # Linear warmup
            frac = self._epoch / self._warmup_epochs
            for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                group["lr"] = base_lr * frac
        else:
            self._scheduler.step()

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "backbone": self.backbone.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "scheduler": self._scheduler.state_dict(),
                "scheduler_epoch": self._epoch,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.backbone.load_state_dict(checkpoint["backbone"])
        self.policy_head.load_state_dict(checkpoint["policy_head"])
        self.value_head.load_state_dict(checkpoint["value_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if "scheduler" in checkpoint:
            self._scheduler.load_state_dict(checkpoint["scheduler"])
            self._epoch = checkpoint["scheduler_epoch"]
