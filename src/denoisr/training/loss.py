import torch
from torch import Tensor
from torch.nn import functional as F


class ChessLossComputer:
    """6-term loss computer with optional HarmonyDream balancing.

    Core losses (always active):
    1. Policy: cross-entropy between predicted logits and target distribution
    2. Value: cross-entropy between predicted WDL and target WDL

    Auxiliary losses (Phase 2/3, passed via kwargs):
    3. Consistency: SimSiam negative cosine similarity
    4. Diffusion: MSE between predicted and actual noise
    5. Reward: MSE between predicted and actual reward
    6. Ply: Huber loss on game-length prediction

    HarmonyDream (optional): tracks EMA of per-loss magnitudes and
    adjusts coefficients inversely proportional to balance contributions.
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        consistency_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        reward_weight: float = 1.0,
        ply_weight: float = 0.1,
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
        }
        self._use_harmony = use_harmony_dream
        self._ema_decay = harmony_ema_decay
        self._ema_norms: dict[str, float] = {}
        self._coefficients: dict[str, float] = dict(self._base_weights)

    def compute(
        self,
        pred_policy: Tensor,
        pred_value: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
        **auxiliary_losses: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        B = pred_policy.shape[0]
        pred_flat = pred_policy.reshape(B, -1)
        target_flat = target_policy.reshape(B, -1)
        # Mask illegal moves: set logits to -inf where target is zero
        legal_mask = target_flat > 0
        masked_logits = pred_flat.masked_fill(~legal_mask, float("-inf"))
        log_probs = F.log_softmax(masked_logits, dim=-1)
        # Replace -inf with 0 at illegal positions to avoid 0 * -inf = NaN
        log_probs = log_probs.masked_fill(~legal_mask, 0.0)
        policy_loss = -(target_flat * log_probs).sum(dim=-1).mean()

        pred_log = torch.log(pred_value.clamp(min=1e-8))
        value_loss = -(target_value * pred_log).sum(dim=-1).mean()

        losses = {"policy": policy_loss, "value": value_loss}

        for name in ("consistency", "diffusion", "reward", "ply"):
            key = f"{name}_loss"
            if key in auxiliary_losses:
                losses[name] = auxiliary_losses[key]

        if self._use_harmony:
            self._update_harmony(losses)

        total = torch.tensor(0.0, device=pred_policy.device)
        for name, loss in losses.items():
            coeff = self._coefficients.get(name, self._base_weights.get(name, 1.0))
            total = total + coeff * loss

        breakdown: dict[str, float] = {
            name: loss.item() for name, loss in losses.items()
        }
        breakdown["total"] = total.item()
        return total, breakdown

    def _update_harmony(self, losses: dict[str, Tensor]) -> None:
        for name, loss in losses.items():
            norm = loss.detach().abs().item()
            if name not in self._ema_norms:
                self._ema_norms[name] = norm
            else:
                self._ema_norms[name] = (
                    self._ema_decay * self._ema_norms[name]
                    + (1 - self._ema_decay) * norm
                )

        if self._ema_norms:
            mean_norm = sum(self._ema_norms.values()) / len(self._ema_norms)
            if mean_norm > 0:
                for name in self._ema_norms:
                    ratio = mean_norm / max(self._ema_norms[name], 1e-8)
                    self._coefficients[name] = (
                        self._base_weights.get(name, 1.0) * ratio
                    )

    def get_coefficients(self) -> dict[str, float]:
        return dict(self._coefficients)
