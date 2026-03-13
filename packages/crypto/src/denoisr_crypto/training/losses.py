"""Loss functions for entry-quality supervision."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    name: str = "p6"
    alpha: float = 0.45
    beta: float = 0.30
    gamma_sparse: float = 0.15
    delta_denoise: float = 0.10
    drawdown_lambda: float = 1.0
    gamma_time_discount: float = 0.98
    epsilon: float = 1e-4
    r_min: float = 0.03
    sparsity_target: float = 0.05
    tau: float = 0.50
    path_scale: float = 0.02
    miss_policy: str = "opportunity_cost"
    opportunity_weight: float = 0.10
    opportunity_q_min: float = 1.0
    use_volatility_scaled_targets: bool = True


@dataclass(frozen=True)
class LossArtifacts:
    total: torch.Tensor
    components: dict[str, float]


def _soft_buy_weight(score: torch.Tensor, tau: float, temp: float) -> torch.Tensor:
    return torch.sigmoid((score - tau) / max(temp, 1e-6))


def _effective_returns(batch: dict[str, torch.Tensor], config: LossConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.use_volatility_scaled_targets:
        return batch["r_up_vol_scaled"], batch["r_dd_vol_scaled"]
    return batch["r_up"], batch["r_dd"]


def _discounted_quality(batch: dict[str, torch.Tensor], config: LossConfig) -> torch.Tensor:
    r_up, r_dd = _effective_returns(batch, config)
    discounted = r_up * torch.pow(
        torch.full_like(batch["t_max_hours"], config.gamma_time_discount),
        batch["t_max_hours"],
    )
    denominator = r_dd * (1.0 + config.drawdown_lambda * batch["f_dd"]) + config.epsilon
    return discounted / denominator


def _minimum_gain_gate(batch: dict[str, torch.Tensor], config: LossConfig) -> torch.Tensor:
    threshold = batch["r_min_scaled"] if config.use_volatility_scaled_targets else torch.full_like(batch["r_up"], config.r_min)
    return torch.sigmoid((batch["r_up"] - threshold) / threshold.clamp_min(config.epsilon))


def _weighted_quality_loss(
    per_item_loss: torch.Tensor,
    *,
    weight: torch.Tensor,
) -> torch.Tensor:
    weight_sum = weight.sum().clamp_min(1e-6)
    return (per_item_loss * weight).sum() / weight_sum


def _kl_bernoulli(p_actual: torch.Tensor, target: float, epsilon: float) -> torch.Tensor:
    q = torch.full_like(p_actual, target).clamp(epsilon, 1.0 - epsilon)
    p = p_actual.clamp(epsilon, 1.0 - epsilon)
    return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))


def _opportunity_penalty(
    *,
    quality: torch.Tensor,
    gain_gate: torch.Tensor,
    weight: torch.Tensor,
    config: LossConfig,
) -> torch.Tensor:
    if config.miss_policy != "opportunity_cost":
        return torch.zeros((), device=quality.device, dtype=quality.dtype)
    opportunity = torch.sigmoid(quality - config.opportunity_q_min) * gain_gate
    return ((1.0 - weight) * opportunity).mean()


def compute_entry_loss(
    *,
    score: torch.Tensor,
    temp: float,
    batch: dict[str, torch.Tensor],
    config: LossConfig,
    denoise_loss: torch.Tensor | None = None,
) -> LossArtifacts:
    weight = _soft_buy_weight(score, config.tau, temp)
    quality = _discounted_quality(batch, config)
    gain_gate = _minimum_gain_gate(batch, config)
    ratio_loss = 1.0 / (1.0 + (quality * gain_gate))
    additive_gain = -torch.log1p(batch["r_up"])
    additive_drawdown = batch["r_dd"] * (1.0 + config.drawdown_lambda * batch["f_dd"])
    additive_loss = config.alpha * additive_gain + config.beta * additive_drawdown
    regret_loss = config.alpha * torch.log1p(batch["regret_up"]) + config.beta * additive_drawdown
    path_loss = torch.tanh(batch["a_pain_15m"] / config.path_scale)
    sparse_loss = _kl_bernoulli(weight.mean(), config.sparsity_target, config.epsilon)
    miss_loss = _opportunity_penalty(
        quality=quality,
        gain_gate=gain_gate,
        weight=weight,
        config=config,
    )

    if config.name == "p1":
        quality_term = _weighted_quality_loss(additive_loss, weight=weight)
        total = quality_term + miss_loss * config.opportunity_weight
    elif config.name == "p2":
        quality_term = _weighted_quality_loss(ratio_loss, weight=weight)
        total = quality_term + miss_loss * config.opportunity_weight
    elif config.name == "p3":
        quality_term = _weighted_quality_loss(regret_loss, weight=weight)
        total = quality_term + miss_loss * config.opportunity_weight
    elif config.name == "p5":
        quality_term = _weighted_quality_loss(ratio_loss, weight=weight)
        total = quality_term + config.gamma_sparse * sparse_loss + miss_loss * config.opportunity_weight
    elif config.name == "p6":
        quality_term = _weighted_quality_loss(ratio_loss, weight=weight)
        denoise_term = denoise_loss if denoise_loss is not None else torch.zeros((), device=score.device)
        total = (
            config.alpha * quality_term
            + config.beta * path_loss.mean()
            + config.gamma_sparse * sparse_loss
            + config.delta_denoise * denoise_term
            + config.opportunity_weight * miss_loss
        )
    else:
        raise ValueError(f"Unsupported loss name: {config.name}")

    return LossArtifacts(
        total=total,
        components={
            "weight_mean": float(weight.mean().item()),
            "quality_ratio_mean": float(quality.mean().item()),
            "ratio_loss_mean": float(ratio_loss.mean().item()),
            "path_loss_mean": float(path_loss.mean().item()),
            "sparse_loss": float(sparse_loss.item()),
            "miss_loss": float(miss_loss.item()),
            "temperature": float(temp),
            "loss": float(total.item()),
        },
    )


def temperature_for_epoch(epoch: int) -> float:
    if epoch <= 10:
        return 2.0
    if epoch <= 30:
        progress = (epoch - 10) / 20.0
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.2 + (2.0 - 0.2) * cosine
    return 0.1
