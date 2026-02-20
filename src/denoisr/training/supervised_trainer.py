from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.amp import autocast  # type: ignore[attr-defined]

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
    ) -> None:
        self.encoder = encoder
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head
        self.loss_fn = loss_fn
        self.device = device or torch.device("cpu")
        self.max_grad_norm = 1.0
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self._autocast_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        self._autocast_enabled = self.device.type == "cuda"

        param_groups = [
            {"params": list(encoder.parameters()), "lr": lr * 0.3},
            {"params": list(backbone.parameters()), "lr": lr * 0.3},
            {"params": list(policy_head.parameters()), "lr": lr},
            {"params": list(value_head.parameters()), "lr": lr},
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=1e-4
        )

        self._warmup_epochs = warmup_epochs
        self._base_lrs: list[float] = [float(g["lr"]) for g in param_groups]  # type: ignore[arg-type]
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6
        )
        self._epoch = 0

    def _forward_backward(
        self,
        boards: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> tuple[float, dict[str, float]]:
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

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()  # type: ignore[no-untyped-call]
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss.item(), breakdown

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
        return self._forward_backward(boards, target_policies, target_values)

    def train_step_tensors(
        self,
        boards: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> tuple[float, dict[str, float]]:
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
