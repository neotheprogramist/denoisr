import torch
from torch import Tensor, nn
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.amp import autocast  # type: ignore[attr-defined]

from denoisr_chess.nn.diffusion import CosineNoiseSchedule


class DiffusionTrainer:
    """Trains the diffusion module to denoise future latent trajectories.

    Given a trajectory of board tensors, encodes them into latent space,
    corrupts future states with Gaussian noise, and trains the diffusion
    model to predict the velocity (v-prediction). The current state
    serves as the condition.
    """

    def __init__(
        self,
        encoder: nn.Module,
        diffusion: nn.Module,
        schedule: CosineNoiseSchedule,
        lr: float = 1e-4,
        device: torch.device | None = None,
        max_grad_norm: float = 1.0,
        curriculum_initial_fraction: float = 0.25,
        curriculum_growth: float = 1.02,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.encoder = encoder
        self.diffusion = diffusion
        self.schedule = schedule
        self.device = device or torch.device("cpu")

        params = list(diffusion.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.max_grad_norm = max_grad_norm
        use_scaler = amp_dtype == torch.float16 and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=use_scaler)
        self._autocast_device = (
            self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        )
        self._autocast_enabled = amp_dtype is not None and self.device.type == "cuda"
        self._amp_dtype = amp_dtype

        self._curriculum_max_steps = schedule.num_timesteps
        initial_steps = max(
            1, int(schedule.num_timesteps * curriculum_initial_fraction)
        )
        self._current_max_steps_f = float(initial_steps)
        self._current_max_steps = initial_steps
        self._curriculum_growth = curriculum_growth

    def train_step(self, trajectories: Tensor) -> tuple[float, dict[str, float]]:
        """Train on a batch of board tensor trajectories.

        trajectories: [B, T, C, 8, 8] where T is consecutive board states.
        Uses position 0 as condition, a random later position as target.
        """
        B, T, C, H, W = trajectories.shape

        self.encoder.eval()
        self.diffusion.train()

        with autocast(self._autocast_device, enabled=self._autocast_enabled, dtype=self._amp_dtype):
            with torch.no_grad():
                flat = trajectories.reshape(B * T, C, H, W)
                latent_flat = self.encoder(flat)
                latent = latent_flat.reshape(B, T, 64, -1)

            cond = latent[:, 0]

            target_idx = torch.randint(1, T, (B,), device=self.device)
            batch_idx = torch.arange(B, device=self.device)
            target = latent[batch_idx, target_idx]

            t = torch.randint(0, self._current_max_steps, (B,), device=self.device)
            noise = torch.randn_like(target)
            noisy_target = self.schedule.q_sample(target, t, noise)

            v_target = self.schedule.compute_v_target(target, noise, t)
            v_pred = self.diffusion(noisy_target, t, cond)

            loss = nn.functional.mse_loss(v_pred, v_target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        breakdown = {"grad_norm": total_norm.item()}
        return loss.item(), breakdown

    @property
    def current_max_steps(self) -> int:
        """Current curriculum diffusion step limit."""
        return self._current_max_steps

    def advance_curriculum(self) -> None:
        """Call once per epoch to increase diffusion step difficulty."""
        self._current_max_steps_f = min(
            float(self._curriculum_max_steps),
            self._current_max_steps_f * self._curriculum_growth,
        )
        self._current_max_steps = int(self._current_max_steps_f)
