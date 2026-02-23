"""Shared model and training configuration with construction helpers.

All tunable hyperparameters live here in two frozen dataclasses:
- ModelConfig: architecture params, saved in checkpoints, needed at inference
- TrainingConfig: optimization params, used only during training
"""

import argparse
import logging
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.nn.consistency import ChessConsistencyProjector
from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.nn.world_model import ChessWorldModel

log = logging.getLogger(__name__)
DEFAULT_AUTO_WORKERS = 64


# ---------------------------------------------------------------------------
# Model architecture config (saved in checkpoints)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    # Number of input feature planes from ExtendedBoardEncoder (122 planes:
    # 12 piece placement + 84 history + 14 metadata + 12 tactical).
    num_planes: int = 122

    # Latent dimension per square token. All transformer layers, heads, and
    # projections use this width. Larger = more capacity but quadratic in
    # attention memory. 256 balances strength vs. training speed.
    d_s: int = 256

    # Number of attention heads in the policy backbone. Must divide d_s.
    # More heads = finer-grained attention patterns. 8 is standard for d=256.
    num_heads: int = 8

    # Depth of the main policy backbone transformer. More layers = deeper
    # positional reasoning. 15 matches Lc0 BT4 architecture.
    num_layers: int = 15

    # Feed-forward hidden dimension inside transformer blocks.
    # Typically 4× d_s. Wider FFN = more per-position processing capacity.
    ffn_dim: int = 1024

    # Number of discrete diffusion timesteps. More steps = finer noise
    # schedule = better sample quality, but slower generation. 100 is standard.
    num_timesteps: int = 100

    # Depth of the world model transformer (UniZero-style latent dynamics).
    # Fewer layers than backbone since it predicts one-step transitions.
    world_model_layers: int = 12

    # Depth of the DiT diffusion denoiser. Fewer layers since it operates
    # in latent space (already compressed by the encoder).
    diffusion_layers: int = 6

    # Projection dimension for the consistency loss (SimSiam-style).
    # Prevents latent-space collapse by enforcing representation diversity.
    proj_dim: int = 256

    # Trade VRAM for compute by recomputing activations during backward pass.
    # Enables training larger models on limited GPU memory at ~30% speed cost.
    gradient_checkpointing: bool = False


# ---------------------------------------------------------------------------
# Training hyperparameter config (NOT saved in checkpoints)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """All tunable training hyperparameters across all phases.

    Optimization, loss weighting, curriculum, data loading, and phase
    transition gates are centralized here so nothing is hardcoded in
    trainer classes or training scripts.
    """

    # -- Optimization -------------------------------------------------------

    # Maximum L2 norm for gradient clipping. Prevents training instability
    # from occasional large gradient spikes (e.g. unusual positions).
    # 5.0 is permissive for from-scratch training; decrease to 1.0 for
    # fine-tuning (visible in TensorBoard under gradients/norm).
    max_grad_norm: float = 5.0

    # AdamW weight decay coefficient. Acts as L2 regularization to prevent
    # overfitting. 1e-4 is mild — increase to 1e-2 for smaller datasets,
    # decrease to 0 if underfitting.
    weight_decay: float = 1e-4

    # Learning rate multiplier for the encoder and backbone relative to
    # task-specific heads (policy, value). Lower values preserve pretrained
    # representations when fine-tuning. 1.0 means encoder/backbone train
    # at the same LR as heads (appropriate for from-scratch training).
    encoder_lr_multiplier: float = 1.0

    # Minimum learning rate at the end of cosine annealing. Prevents LR
    # from reaching exactly 0, which would completely stall learning.
    # Should be 10-100× smaller than the base LR.
    min_lr: float = 1e-6

    # Number of initial epochs with linearly increasing LR (0 → target).
    # Prevents destructively large parameter updates when weights are
    # still random. 5 epochs compensates for higher peak LR (3e-4).
    warmup_epochs: int = 5

    # Enable cosine annealing with warm restarts (T_0=20, T_mult=2) instead
    # of plain CosineAnnealingLR. Periodically resets LR to help escape local
    # basins when training stalls.
    use_warm_restarts: bool = True

    # -- Loss weights -------------------------------------------------------
    # These control the relative importance of each training objective.
    # Higher weight = model prioritizes that objective more.

    # Weight for policy (move prediction) cross-entropy loss.
    # Set higher than value_weight because learning correct moves is the
    # primary objective — value estimation is secondary supervision.
    policy_weight: float = 2.0

    # Weight for value (win/draw/loss) cross-entropy loss.
    # Lower than policy because position evaluation is less critical than
    # move selection in early training phases.
    value_weight: float = 0.5

    # Weight for consistency loss (SimSiam negative cosine similarity).
    # Prevents the world model's latent space from collapsing to a
    # constant — ensures distinct positions map to distinct latents.
    consistency_weight: float = 1.0

    # Weight for diffusion denoising loss (MSE between predicted/actual noise).
    # Trains the imagination module to generate plausible future trajectories.
    diffusion_weight: float = 1.0

    # Weight for reward prediction loss (MSE between predicted/actual reward).
    # Teaches the world model to predict game outcomes from latent states.
    reward_weight: float = 1.0

    # Weight for ply (game length) prediction loss (Huber loss).
    # Auxiliary signal — lower weight since game length is less important
    # than move quality or position evaluation.
    ply_weight: float = 0.1

    # Weight for illegal-move logit L2 penalty. Encourages the model to
    # output low logits at illegal positions, improving accuracy evaluation
    # robustness. Small values (0.01) prevent interference with policy loss.
    illegal_penalty_weight: float = 0.01

    # Weight for threat prediction auxiliary loss. Forces intermediate
    # representations to encode threat information, improving defensive play.
    threat_weight: float = 0.1

    # -- HarmonyDream loss balancing -----------------------------------------

    # Enable HarmonyDream dynamic loss balancing (Ma et al., ICML 2024).
    # When active, loss coefficients auto-adjust inversely proportional
    # to each loss's EMA magnitude, keeping all objectives balanced.
    # Recommended for Phase 2+ when all 6 losses are active.
    use_harmony_dream: bool = True

    # EMA decay for tracking per-loss magnitudes in HarmonyDream.
    # Higher = smoother, slower adaptation. 0.99 means ~100 steps to
    # converge to a new loss scale after a distribution shift.
    harmony_ema_decay: float = 0.99

    # -- Diffusion curriculum -----------------------------------------------

    # Fraction of total diffusion timesteps used at the start of training.
    # Training begins with easier denoising tasks (fewer corruption steps)
    # and gradually increases. 0.25 means start with T/4 steps.
    curriculum_initial_fraction: float = 0.25

    # Per-epoch multiplier for the curriculum step count. Each epoch,
    # max_steps *= curriculum_growth until reaching num_timesteps.
    # 1.02 = ~2% growth per epoch, reaching full difficulty in ~70 epochs.
    curriculum_growth: float = 1.02

    # -- Data loading -------------------------------------------------------

    # DataLoader worker processes for parallel data loading (0 = auto: 64).
    # Set to 1 for debugging (single-process).
    workers: int = 0

    # -- Phase gates --------------------------------------------------------

    # Top-1 policy accuracy threshold to advance from Phase 1 to Phase 2.
    # The model must predict the best move at least this often before
    # diffusion training is worthwhile. 50% ensures strong move-ranking
    # ability and lets the model train through most/all epochs.
    phase1_gate: float = 0.50

    # Percentage-point improvement in accuracy from diffusion-conditioned
    # vs single-step inference required to advance from Phase 2 to Phase 3.
    # Ensures the diffusion module actually helps before starting RL.
    phase2_gate: float = 5.0

    # -- Phase 3: MCTS → diffusion transition -------------------------------

    # Number of generations over which the MCTS→diffusion alpha mixing
    # transitions from 0 (pure MCTS) to 1 (pure diffusion).
    # Shorter = faster transition but risk of instability.
    alpha_generations: int = 50

    # UCB exploration constant for MCTS. Controls exploration vs exploitation
    # in tree search. Higher = more exploration of less-visited nodes.
    # 1.4 is standard (√2 ≈ 1.414 from UCB1 theory).
    c_puct: float = 1.4

    # Dirichlet noise alpha for MCTS root exploration. Smaller values
    # concentrate noise on fewer moves (sharper). 0.3 is standard for chess
    # (vs 0.03 for Go which has more moves per position).
    dirichlet_alpha: float = 0.3

    # Fraction of root prior replaced by Dirichlet noise. Controls how much
    # random exploration overrides the policy network at the root.
    # 0.25 means 75% policy, 25% noise.
    dirichlet_epsilon: float = 0.25

    # -- Phase 3: Temperature schedule --------------------------------------

    # Base temperature for self-play move selection. Higher = more random
    # exploration. Decays across generations by temperature_generation_decay.
    temperature_base: float = 1.0

    # Number of moves at the start of each game that use the full temperature.
    # After this many moves, temperature drops to 0 (greedy selection).
    # 30 covers most opening theory for diverse opening repertoire.
    temperature_explore_moves: int = 30

    # Per-generation decay for the base temperature. As training progresses,
    # self-play becomes more exploitative (lower temperature).
    # 0.97 = ~50% temperature after 23 generations.
    temperature_generation_decay: float = 0.97

    # Maximum moves per self-play game before declaring a draw.
    # Prevents infinite games in positions the model can't resolve.
    max_moves: int = 300

    # Number of MCTS simulations for MuZero Reanalyse.
    # Lower than main MCTS sims since reanalyse runs on many old games.
    # Trades search depth for broader coverage of the replay buffer.
    reanalyse_simulations: int = 100

    # -- Grokking detection ---------------------------------------------------

    grok_tracking: bool = True
    grok_erank_freq: int = 1000
    grok_spectral_freq: int = 5000
    grok_onset_threshold: float = 0.95

    # -- Grokfast acceleration ------------------------------------------------

    grokfast: bool = True
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0

    # -- EMA shadow model evaluation ------------------------------------------

    # EMA decay for shadow model evaluation. 0 = disabled.
    # 0.999 for large datasets, 0.9999 for very long training. Enabled by
    # default — the shadow model provides smoother evaluation metrics.
    ema_decay: float = 0.999


def _detect_available_cpus() -> int:
    """Best-effort count of logical CPUs available to this process."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:  # noqa: BLE001
            pass
    return max(1, os.cpu_count() or 1)


def resolve_workers(workers: int) -> int:
    """Resolve process workers: 0 means auto (hardcoded default 64)."""
    if workers > 0:
        return workers
    return DEFAULT_AUTO_WORKERS


def resolve_dataloader_workers(workers: int) -> int:
    """Resolve DataLoader workers with hardcoded auto default and safety clamp."""
    max_workers = _detect_available_cpus()
    if workers <= 0:
        return min(DEFAULT_AUTO_WORKERS, max_workers)
    if workers > max_workers:
        log.warning(
            "Requested DataLoader workers=%d exceeds available CPUs=%d; clamping to %d",
            workers,
            max_workers,
            max_workers,
        )
        return max_workers
    return workers


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_M = TypeVar("_M", bound=nn.Module)


def maybe_compile(module: _M, device: torch.device) -> _M:
    """Compile module with torch.compile on CUDA, return as-is otherwise."""
    if device.type == "cuda":
        return torch.compile(module)  # type: ignore[return-value]
    return module


def build_encoder(cfg: ModelConfig) -> ChessEncoder:
    return ChessEncoder(num_planes=cfg.num_planes, d_s=cfg.d_s)


def build_board_encoder(cfg: ModelConfig) -> ExtendedBoardEncoder:
    """Return the board encoder (always ExtendedBoardEncoder, 122 planes)."""
    return ExtendedBoardEncoder()


def build_backbone(cfg: ModelConfig) -> ChessPolicyBackbone:
    return ChessPolicyBackbone(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_dim=cfg.ffn_dim,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )


def build_policy_head(cfg: ModelConfig) -> ChessPolicyHead:
    return ChessPolicyHead(d_s=cfg.d_s)


def build_value_head(cfg: ModelConfig) -> ChessValueHead:
    return ChessValueHead(d_s=cfg.d_s)


def build_world_model(cfg: ModelConfig) -> ChessWorldModel:
    return ChessWorldModel(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.world_model_layers,
        ffn_dim=cfg.ffn_dim,
    )


def build_diffusion(cfg: ModelConfig) -> ChessDiffusionModule:
    return ChessDiffusionModule(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.diffusion_layers,
        num_timesteps=cfg.num_timesteps,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )


def build_consistency(cfg: ModelConfig) -> ChessConsistencyProjector:
    return ChessConsistencyProjector(d_s=cfg.d_s, proj_dim=cfg.proj_dim)


def build_schedule(cfg: ModelConfig) -> CosineNoiseSchedule:
    return CosineNoiseSchedule(num_timesteps=cfg.num_timesteps)


def add_model_args(parser: ArgumentParser) -> None:
    """Register CLI flags for model architecture hyperparameters."""
    g = parser.add_argument_group("model")
    g.add_argument(
        "--d-s", type=int, default=256,
        help="latent dimension per square token (default: 256)",
    )
    g.add_argument(
        "--num-heads", type=int, default=8,
        help="attention heads in backbone (default: 8, must divide d_s)",
    )
    g.add_argument(
        "--num-layers", type=int, default=15,
        help="policy backbone transformer depth (default: 15)",
    )
    g.add_argument(
        "--ffn-dim", type=int, default=1024,
        help="feed-forward hidden dim, typically 4×d_s (default: 1024)",
    )
    g.add_argument(
        "--num-timesteps", type=int, default=100,
        help="Diffusion timesteps (default: 100)",
    )
    g.add_argument(
        "--world-model-layers", type=int, default=12,
        help="world model transformer depth (default: 12)",
    )
    g.add_argument(
        "--diffusion-layers", type=int, default=6,
        help="DiT diffusion denoiser depth (default: 6)",
    )
    g.add_argument(
        "--proj-dim", type=int, default=256,
        help="consistency projector dimension (default: 256)",
    )
    g.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable gradient checkpointing (saves VRAM, ~30%% slower; default: off)",
    )


def add_training_args(parser: ArgumentParser) -> None:
    """Register CLI flags for all training hyperparameters."""
    g = parser.add_argument_group("training")
    g.add_argument(
        "--max-grad-norm", type=float, default=5.0,
        help="gradient clipping L2 norm threshold (default: 5.0)",
    )
    g.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    g.add_argument(
        "--encoder-lr-multiplier", type=float, default=1.0,
        help="LR multiplier for encoder/backbone vs heads (default: 1.0)",
    )
    g.add_argument(
        "--min-lr", type=float, default=1e-6,
        help="minimum LR at end of cosine annealing (default: 1e-6)",
    )
    g.add_argument(
        "--warmup-epochs", type=int, default=5,
        help="linear warmup epochs before cosine decay (default: 5)",
    )
    g.add_argument(
        "--warm-restarts",
        action=argparse.BooleanOptionalAction, default=True,
        help="use cosine annealing with warm restarts (default: on)",
    )
    g.add_argument(
        "--threat-weight", type=float, default=0.1,
        help="loss weight for threat prediction auxiliary head (default: 0.1)",
    )
    g.add_argument(
        "--policy-weight", type=float, default=2.0,
        help="loss weight for policy cross-entropy (default: 2.0)",
    )
    g.add_argument(
        "--value-weight", type=float, default=0.5,
        help="loss weight for value cross-entropy (default: 0.5)",
    )
    g.add_argument(
        "--consistency-weight", type=float, default=1.0,
        help="loss weight for consistency (default: 1.0)",
    )
    g.add_argument(
        "--diffusion-weight", type=float, default=1.0,
        help="loss weight for diffusion denoising (default: 1.0)",
    )
    g.add_argument(
        "--reward-weight", type=float, default=1.0,
        help="loss weight for reward prediction (default: 1.0)",
    )
    g.add_argument(
        "--ply-weight", type=float, default=0.1,
        help="loss weight for game-length prediction (default: 0.1)",
    )
    g.add_argument(
        "--illegal-penalty-weight", type=float, default=0.01,
        help="L2 penalty weight on illegal-move logits (default: 0.01)",
    )
    g.add_argument(
        "--harmony-dream",
        action=argparse.BooleanOptionalAction, default=True,
        help="enable HarmonyDream dynamic loss balancing (default: on)",
    )
    g.add_argument(
        "--harmony-ema-decay", type=float, default=0.99,
        help="EMA decay for HarmonyDream loss tracking (default: 0.99)",
    )
    g.add_argument(
        "--curriculum-initial-fraction", type=float, default=0.25,
        help="fraction of diffusion steps at curriculum start (default: 0.25)",
    )
    g.add_argument(
        "--curriculum-growth", type=float, default=1.02,
        help="per-epoch multiplier for curriculum steps (default: 1.02)",
    )
    g.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader worker processes (0 = auto: 64)",
    )
    g.add_argument(
        "--tqdm", action="store_true", default=False,
        help="show tqdm progress bars (default: off, structured log lines instead)",
    )
    g.add_argument(
        "--phase1-gate", type=float, default=0.50,
        help="top-1 accuracy to pass Phase 1 gate (default: 0.50)",
    )
    g.add_argument(
        "--phase2-gate", type=float, default=5.0,
        help="diffusion improvement pp to pass Phase 2 gate (default: 5.0)",
    )
    # Grokking detection
    g.add_argument(
        "--grok-tracking",
        action=argparse.BooleanOptionalAction, default=True,
        help="enable grokking detection metrics (default: on)",
    )
    g.add_argument(
        "--grok-erank-freq", type=int, default=1000,
        help="effective rank computation frequency in steps (default: 1000)",
    )
    g.add_argument(
        "--grok-spectral-freq", type=int, default=5000,
        help="spectral norm / HTSR alpha frequency in steps (default: 5000)",
    )
    g.add_argument(
        "--grok-onset-threshold", type=float, default=0.95,
        help="weight norm ratio for onset detection (default: 0.95)",
    )
    g.add_argument(
        "--grokfast",
        action=argparse.BooleanOptionalAction, default=True,
        help="enable Grokfast EMA gradient filtering (default: on)",
    )
    g.add_argument(
        "--grokfast-alpha", type=float, default=0.98,
        help="Grokfast EMA decay rate (default: 0.98)",
    )
    g.add_argument(
        "--grokfast-lamb", type=float, default=2.0,
        help="Grokfast amplification factor (default: 2.0)",
    )
    # EMA
    g.add_argument(
        "--ema-decay", type=float, default=0.999,
        help="EMA decay for shadow model evaluation (0=disabled, default: 0.999)",
    )


def add_phase3_args(parser: ArgumentParser) -> None:
    """Register Phase 3-specific CLI flags for self-play and MCTS."""
    g = parser.add_argument_group("phase3")
    g.add_argument(
        "--c-puct", type=float, default=1.4,
        help="MCTS UCB exploration constant (default: 1.4)",
    )
    g.add_argument(
        "--dirichlet-alpha", type=float, default=0.3,
        help="Dirichlet noise alpha for root exploration (default: 0.3)",
    )
    g.add_argument(
        "--dirichlet-epsilon", type=float, default=0.25,
        help="fraction of root prior replaced by Dirichlet noise (default: 0.25)",
    )
    g.add_argument(
        "--temperature-base", type=float, default=1.0,
        help="base temperature for move selection (default: 1.0)",
    )
    g.add_argument(
        "--temperature-explore-moves", type=int, default=30,
        help="moves per game using full temperature (default: 30)",
    )
    g.add_argument(
        "--temperature-generation-decay", type=float, default=0.97,
        help="per-generation temperature decay (default: 0.97)",
    )
    g.add_argument(
        "--max-moves", type=int, default=300,
        help="maximum moves per self-play game (default: 300)",
    )
    g.add_argument(
        "--reanalyse-simulations", type=int, default=100,
        help="MCTS sims for MuZero Reanalyse (default: 100)",
    )


def training_config_from_args(args: Namespace) -> TrainingConfig:
    """Build TrainingConfig from parsed CLI arguments."""
    return TrainingConfig(
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        encoder_lr_multiplier=args.encoder_lr_multiplier,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        use_warm_restarts=args.warm_restarts,
        threat_weight=args.threat_weight,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        consistency_weight=args.consistency_weight,
        diffusion_weight=args.diffusion_weight,
        reward_weight=args.reward_weight,
        ply_weight=args.ply_weight,
        illegal_penalty_weight=args.illegal_penalty_weight,
        use_harmony_dream=args.harmony_dream,
        harmony_ema_decay=args.harmony_ema_decay,
        curriculum_initial_fraction=args.curriculum_initial_fraction,
        curriculum_growth=args.curriculum_growth,
        workers=args.workers,
        phase1_gate=args.phase1_gate,
        phase2_gate=args.phase2_gate,
        grok_tracking=args.grok_tracking,
        grok_erank_freq=args.grok_erank_freq,
        grok_spectral_freq=args.grok_spectral_freq,
        grok_onset_threshold=args.grok_onset_threshold,
        grokfast=args.grokfast,
        grokfast_alpha=args.grokfast_alpha,
        grokfast_lamb=args.grokfast_lamb,
        ema_decay=args.ema_decay,
    )


def full_training_config_from_args(args: Namespace) -> TrainingConfig:
    """Build TrainingConfig including Phase 3 args from parsed CLI arguments."""
    base = training_config_from_args(args)
    return replace(
        base,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        temperature_base=args.temperature_base,
        temperature_explore_moves=args.temperature_explore_moves,
        temperature_generation_decay=args.temperature_generation_decay,
        max_moves=args.max_moves,
        reanalyse_simulations=args.reanalyse_simulations,
    )


def resolve_gradient_checkpointing(
    cfg: ModelConfig, args: Namespace, device: torch.device,
) -> ModelConfig:
    """Override checkpoint config's gradient_checkpointing with CLI flag.

    --gradient-checkpointing → True (saves VRAM, ~30% slower)
    --no-gradient-checkpointing or default → False (optimize training speed)
    """
    gc = args.gradient_checkpointing
    if gc != cfg.gradient_checkpointing:
        return replace(cfg, gradient_checkpointing=gc)
    return cfg


def config_from_args(args: Namespace) -> ModelConfig:
    return ModelConfig(
        d_s=args.d_s,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        num_timesteps=args.num_timesteps,
        world_model_layers=args.world_model_layers,
        diffusion_layers=args.diffusion_layers,
        proj_dim=args.proj_dim,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def save_checkpoint(
    path: Path,
    cfg: ModelConfig,
    **state_dicts: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"config": cfg.__dict__}
    data.update(state_dicts)
    torch.save(data, path)
    log.info("Checkpoint saved to %s", path)


def _strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip ``_orig_mod.`` prefix that ``torch.compile`` adds to state dict keys."""
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[ModelConfig, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**data.pop("config"))
    # Checkpoints saved from torch.compile()-wrapped models have an
    # ``_orig_mod.`` prefix on every key.  Strip it so state dicts load
    # correctly into both compiled and uncompiled modules.
    return cfg, {
        k: _strip_compile_prefix(v) if isinstance(v, dict) else v
        for k, v in data.items()
    }
