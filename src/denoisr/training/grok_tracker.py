"""Grokking detection tracker for Phase 1 supervised training.

Registers forward hooks on backbone transformer layers to capture activations.
Computes Tier 1 (weight norms, loss gap) and Tier 2 (effective rank, spectral
norms, HTSR alpha) metrics at configurable frequencies. Contains a 4-state
machine for adaptive evaluation frequency and console alerts.

References:
- Power et al. (2022) — grokking in modular arithmetic
- Nanda et al. (2023) — mechanistic interpretability of grokking (ICLR Oral)
- Zunkovic & Ilievski (2024) — effective dimensionality (JMLR)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from enum import IntEnum, auto

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from denoisr.nn.policy_backbone import ChessPolicyBackbone, TransformerBlock

log = logging.getLogger(__name__)


class GrokState(IntEnum):
    BASELINE = 0
    ONSET_DETECTED = auto()
    TRANSITIONING = auto()
    GROKKED = auto()


class GrokTracker:
    """Grokking detection and metric computation for training instrumentation."""

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        erank_freq: int = 1000,
        spectral_freq: int = 5000,
        onset_threshold: float = 0.95,
        on_state_transition: Callable[[int, str, str, str], None] | None = None,
    ) -> None:
        self._encoder = encoder
        self._backbone = backbone
        self._policy_head = policy_head
        self._value_head = value_head
        self._erank_freq = erank_freq
        self._spectral_freq = spectral_freq
        self._onset_threshold = onset_threshold
        self._on_state_transition = on_state_transition

        # State machine
        self._state = GrokState.BASELINE
        self._weight_norm_history: list[float] = []
        self._holdout_accuracy_history: list[float] = []
        self._erank_history: list[float] = []
        self._train_saturation_step: int | None = None

        # Adaptive frequency multiplier
        self._freq_multiplier = 1

        # Forward hooks on backbone layers
        self._activations: dict[int, Tensor] = {}
        self._hooks: list[RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        if not isinstance(self._backbone, ChessPolicyBackbone):
            return
        for i, layer in enumerate(self._backbone.layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook_fn(module: nn.Module, input: object, output: Tensor) -> None:
            self._activations[layer_idx] = output.detach()

        return hook_fn

    def close(self) -> None:
        """Remove all forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # -- Tier 1: Weight norms (every step) -----------------------------------

    def compute_weight_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}
        for name, module in [
            ("encoder", self._encoder),
            ("backbone", self._backbone),
            ("policy_head", self._policy_head),
            ("value_head", self._value_head),
        ]:
            sq_sum = sum(p.data.norm(2).item() ** 2 for p in module.parameters())
            norms[name] = sq_sum**0.5
        norms["total"] = sum(v**2 for v in norms.values()) ** 0.5
        return norms

    # -- Tier 2: Effective rank (every N steps) ------------------------------

    def compute_effective_rank(self) -> dict[int, float]:
        eranks: dict[int, float] = {}
        for layer_idx, act in self._activations.items():
            # act shape: [B, 64, d_s]
            flat = act.reshape(-1, act.shape[-1]).float().cpu()
            flat = flat - flat.mean(dim=0)
            svs = torch.linalg.svdvals(flat)
            # Normalize to probability distribution
            p = svs / svs.sum()
            p = p[p > 1e-12]
            # Shannon entropy -> effective rank
            entropy = -(p * p.log()).sum()
            eranks[layer_idx] = math.exp(entropy.item())
        return eranks

    # -- Tier 2: Spectral norms (every N steps) -----------------------------

    def compute_spectral_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}
        if not isinstance(self._backbone, ChessPolicyBackbone):
            return norms
        for i, layer in enumerate(self._backbone.layers):
            if not isinstance(layer, TransformerBlock):
                continue
            # Attention QKV weight (CPU for SVD compatibility)
            w_qkv = layer.qkv.weight.data.float().cpu()
            norms[f"layer_{i}/attn"] = torch.linalg.svdvals(w_qkv)[0].item()
            # FFN first linear weight
            w_ffn = layer.ffn[0].weight.data.float().cpu()
            norms[f"layer_{i}/ffn"] = torch.linalg.svdvals(w_ffn)[0].item()
        return norms

    # -- Tier 2: HTSR alpha (every N steps) ----------------------------------

    def compute_htsr_alpha(self) -> dict[int, float]:
        alphas: dict[int, float] = {}
        if not isinstance(self._backbone, ChessPolicyBackbone):
            return alphas
        for i, layer in enumerate(self._backbone.layers):
            if not isinstance(layer, TransformerBlock):
                continue
            w = layer.qkv.weight.data.float().cpu()
            eigenvalues = torch.linalg.svdvals(w) ** 2
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) < 10:
                continue
            # Fit power-law tail (top 50% of eigenvalues)
            n = len(eigenvalues)
            tail = eigenvalues[: n // 2]
            log_rank = torch.log(torch.arange(1, len(tail) + 1, dtype=torch.float32))
            log_vals = torch.log(tail.cpu())
            # Linear regression: log_vals = -alpha * log_rank + const
            x = log_rank
            y = log_vals
            x_mean = x.mean()
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            alphas[i] = -slope.item()
        return alphas

    # -- Step and epoch hooks ------------------------------------------------

    def step(
        self,
        global_step: int,
        loss_breakdown: dict[str, float],
        grad_norm: float,
    ) -> dict[str, float]:
        """Called after every training step. Returns metrics dict for logging."""
        metrics: dict[str, float] = {}

        # Tier 1: weight norms (every step)
        norms = self.compute_weight_norms()
        metrics["grok/weight_norm_total"] = norms["total"]
        for name, val in norms.items():
            if name != "total":
                metrics[f"grok/weight_norm/{name}"] = val
        self._weight_norm_history.append(norms["total"])

        # Tier 2: effective rank (every N steps, adaptive)
        erank_freq = max(1, self._erank_freq // self._freq_multiplier)
        if global_step % erank_freq == 0 and self._activations:
            eranks = self.compute_effective_rank()
            for layer_idx, erank in eranks.items():
                metrics[f"grok/erank/layer_{layer_idx}"] = erank
            if eranks:
                mean_erank = sum(eranks.values()) / len(eranks)
                self._erank_history.append(mean_erank)

        # Tier 2: spectral norms + HTSR alpha (every N steps, adaptive)
        spectral_freq = max(1, self._spectral_freq // self._freq_multiplier)
        if global_step % spectral_freq == 0:
            spectral = self.compute_spectral_norms()
            for key, val in spectral.items():
                metrics[f"grok/spectral_norm/{key}"] = val
            alphas = self.compute_htsr_alpha()
            for layer_idx, alpha in alphas.items():
                metrics[f"grok/alpha/layer_{layer_idx}"] = alpha

        # State machine transition check
        self._check_step_transitions(global_step)
        metrics["grok/state"] = float(self._state)

        return metrics

    def epoch(
        self,
        epoch: int,
        train_loss: float,
        holdout_metrics: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """Called after every epoch.

        holdout_metrics: mapping of split_name -> (accuracy, loss)
        """
        metrics: dict[str, float] = {}
        for split_name, (acc, loss) in holdout_metrics.items():
            metrics[f"grok/holdout/{split_name}/accuracy"] = acc
            metrics[f"grok/holdout/{split_name}/loss"] = loss

        # Compute loss gap (train - best holdout)
        if holdout_metrics:
            best_holdout_loss = min(loss for _, loss in holdout_metrics.values())
            metrics["grok/loss_gap"] = train_loss - best_holdout_loss

        # Track accuracy for state machine
        if "random" in holdout_metrics:
            self._holdout_accuracy_history.append(holdout_metrics["random"][0])

        self._check_epoch_transitions(epoch, holdout_metrics)
        metrics["grok/state"] = float(self._state)

        return metrics

    # -- State machine -------------------------------------------------------

    def _check_step_transitions(self, global_step: int) -> None:
        if self._state >= GrokState.ONSET_DETECTED:
            return
        # Check weight norm sustained decrease
        history = self._weight_norm_history
        if len(history) >= 100:
            recent = sum(history[-50:]) / 50
            earlier = sum(history[-100:-50]) / 50
            if earlier > 0 and recent < self._onset_threshold * earlier:
                self._transition_to(
                    GrokState.ONSET_DETECTED,
                    global_step,
                    f"weight_norm decreased {(1 - recent / earlier) * 100:.1f}%"
                    " (50-step window)",
                )

        # Check effective rank drop
        if len(self._erank_history) >= 10:
            recent_er = sum(self._erank_history[-5:]) / 5
            earlier_er = sum(self._erank_history[-10:-5]) / 5
            if earlier_er > 0 and recent_er < 0.9 * earlier_er:
                self._transition_to(
                    GrokState.ONSET_DETECTED,
                    global_step,
                    f"effective_rank decreased"
                    f" {(1 - recent_er / earlier_er) * 100:.1f}%",
                )

    def _check_epoch_transitions(
        self,
        epoch: int,
        holdout_metrics: dict[str, tuple[float, float]],
    ) -> None:
        acc_history = self._holdout_accuracy_history

        # ONSET -> TRANSITIONING
        if self._state == GrokState.ONSET_DETECTED and len(acc_history) >= 20:
            recent = sum(acc_history[-10:]) / 10
            earlier = sum(acc_history[-20:-10]) / 10
            if recent - earlier > 0.05:  # >5pp improvement
                self._transition_to(
                    GrokState.TRANSITIONING,
                    epoch,
                    f"holdout accuracy improved"
                    f" {(recent - earlier) * 100:.1f}pp over 20 epochs",
                )

        # TRANSITIONING -> GROKKED
        if self._state == GrokState.TRANSITIONING and acc_history:
            if acc_history[-1] > 0.25:
                grok_gap = epoch - (self._train_saturation_step or 0)
                self._transition_to(
                    GrokState.GROKKED,
                    epoch,
                    f"holdout accuracy {acc_history[-1] * 100:.1f}%"
                    f" > 25% threshold, gap={grok_gap} epochs",
                )

        # BASELINE -> ONSET via holdout accuracy jump
        if self._state == GrokState.BASELINE and len(acc_history) >= 10:
            recent = sum(acc_history[-5:]) / 5
            earlier = sum(acc_history[-10:-5]) / 5
            if recent - earlier > 0.02:  # >2pp improvement
                self._transition_to(
                    GrokState.ONSET_DETECTED,
                    epoch,
                    f"holdout accuracy improved"
                    f" {(recent - earlier) * 100:.1f}pp over 10 epochs",
                )

    def _transition_to(
        self, new_state: GrokState, step_or_epoch: int, trigger: str
    ) -> None:
        old_state = self._state
        self._state = new_state

        # Adaptive frequency
        if new_state == GrokState.ONSET_DETECTED:
            self._freq_multiplier = 5
        elif new_state == GrokState.TRANSITIONING:
            self._freq_multiplier = 10

        log.warning("GROKKING %s (step/epoch %d)", new_state.name, step_or_epoch)
        log.warning("  Trigger: %s", trigger)
        if self._freq_multiplier > 1:
            log.warning(
                "  Action: eval frequency %dx (erank: %d steps, spectral: %d steps)",
                self._freq_multiplier,
                max(1, self._erank_freq // self._freq_multiplier),
                max(1, self._spectral_freq // self._freq_multiplier),
            )

        if self._on_state_transition is not None:
            self._on_state_transition(
                step_or_epoch, old_state.name, new_state.name, trigger
            )

    @property
    def state(self) -> GrokState:
        return self._state
