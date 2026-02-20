import torch
from torch import Tensor, nn
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.amp import autocast  # type: ignore[attr-defined]

from denoisr.nn.diffusion import CosineNoiseSchedule


class DiffusionTrainer:
    """Trains the diffusion module to denoise future latent trajectories.

    Given a trajectory of board tensors, encodes them into latent space,
    corrupts future states with DDPM noise, and trains the diffusion
    model to predict the noise. The current state serves as the condition.
    """

    def __init__(
        self,
        encoder: nn.Module,
        diffusion: nn.Module,
        schedule: CosineNoiseSchedule,
        lr: float = 1e-4,
        device: torch.device | None = None,
    ) -> None:
        self.encoder = encoder
        self.diffusion = diffusion
        self.schedule = schedule
        self.device = device or torch.device("cpu")

        params = list(diffusion.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.max_grad_norm = 1.0
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self._autocast_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        self._autocast_enabled = self.device.type == "cuda"

        self._curriculum_max_steps = schedule.num_timesteps
        self._current_max_steps_f = float(max(1, schedule.num_timesteps // 4))
        self._current_max_steps = int(self._current_max_steps_f)
        self._curriculum_growth = 1.02

    def train_step(self, trajectories: Tensor) -> float:
        """Train on a batch of board tensor trajectories.

        trajectories: [B, T, C, 8, 8] where T is consecutive board states.
        Uses position 0 as condition, a random later position as target.
        """
        B, T, C, H, W = trajectories.shape

        self.encoder.eval()
        self.diffusion.train()

        with autocast(self._autocast_device, enabled=self._autocast_enabled):
            with torch.no_grad():
                flat = trajectories.reshape(B * T, C, H, W)
                latent_flat = self.encoder(flat)
                latent = latent_flat.reshape(B, T, 64, -1)

            cond = latent[:, 0]

            target_idx = torch.randint(1, T, (B,), device=self.device)
            target = torch.stack(
                [latent[b, target_idx[b]] for b in range(B)]
            )

            t = torch.randint(
                0, self._current_max_steps, (B,), device=self.device
            )
            noise = torch.randn_like(target)
            noisy_target = self.schedule.q_sample(target, t, noise)

            predicted_noise = self.diffusion(noisy_target, t, cond)

            loss = nn.functional.mse_loss(predicted_noise, noise)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def advance_curriculum(self) -> None:
        """Call once per epoch to increase diffusion step difficulty."""
        self._current_max_steps_f = min(
            float(self._curriculum_max_steps),
            self._current_max_steps_f * self._curriculum_growth,
        )
        self._current_max_steps = int(self._current_max_steps_f)
