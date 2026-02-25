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
from typing import Any, Callable, TypeVar

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
from denoisr.scripts.runtime import add_env_argument

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
    # fine-tuning (visible in gradients/norm structured event lines).
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
    # still random. 10 epochs is safer for large-batch training.
    warmup_epochs: int = 10

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
        # Favor throughput for fixed-shape training workloads.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


_M = TypeVar("_M", bound=nn.Module)


def maybe_compile(
    module: _M,
    device: torch.device,
) -> _M:
    """Compile module on CUDA; no optional fallback policy."""
    if device.type != "cuda":
        return module
    return torch.compile(module)  # type: ignore[return-value]


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
    """Register model flags; all values must come from env or explicit CLI."""
    add_env_argument(
        parser,
        "--d-s",
        env_var="DENOISR_MODEL_D_S",
        type=int,
        help="latent dimension per square token",
    )
    add_env_argument(
        parser,
        "--num-heads",
        env_var="DENOISR_MODEL_NUM_HEADS",
        type=int,
        help="attention heads in backbone (must divide d_s)",
    )
    add_env_argument(
        parser,
        "--num-layers",
        env_var="DENOISR_MODEL_NUM_LAYERS",
        type=int,
        help="policy backbone transformer depth",
    )
    add_env_argument(
        parser,
        "--ffn-dim",
        env_var="DENOISR_MODEL_FFN_DIM",
        type=int,
        help="feed-forward hidden dim, typically 4×d_s",
    )
    add_env_argument(
        parser,
        "--num-timesteps",
        env_var="DENOISR_MODEL_NUM_TIMESTEPS",
        type=int,
        help="diffusion timesteps",
    )
    add_env_argument(
        parser,
        "--world-model-layers",
        env_var="DENOISR_MODEL_WORLD_MODEL_LAYERS",
        type=int,
        help="world model transformer depth",
    )
    add_env_argument(
        parser,
        "--diffusion-layers",
        env_var="DENOISR_MODEL_DIFFUSION_LAYERS",
        type=int,
        help="DiT diffusion denoiser depth",
    )
    add_env_argument(
        parser,
        "--proj-dim",
        env_var="DENOISR_MODEL_PROJ_DIM",
        type=int,
        help="consistency projector dimension",
    )
    add_env_argument(
        parser,
        "--gradient-checkpointing",
        env_var="DENOISR_MODEL_GRADIENT_CHECKPOINTING",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable gradient checkpointing (saves VRAM, ~30%% slower)",
    )


def add_training_args(parser: ArgumentParser) -> None:
    """Register training flags; all values must come from env or explicit CLI."""
    add_env_argument(
        parser,
        "--max-grad-norm",
        env_var="DENOISR_TRAIN_MAX_GRAD_NORM",
        type=float,
        help="gradient clipping L2 norm threshold",
    )
    add_env_argument(
        parser,
        "--weight-decay",
        env_var="DENOISR_TRAIN_WEIGHT_DECAY",
        type=float,
        help="AdamW weight decay",
    )
    add_env_argument(
        parser,
        "--encoder-lr-multiplier",
        env_var="DENOISR_TRAIN_ENCODER_LR_MULTIPLIER",
        type=float,
        help="LR multiplier for encoder/backbone vs heads",
    )
    add_env_argument(
        parser,
        "--min-lr",
        env_var="DENOISR_TRAIN_MIN_LR",
        type=float,
        help="minimum LR at end of cosine annealing",
    )
    add_env_argument(
        parser,
        "--warmup-epochs",
        env_var="DENOISR_TRAIN_WARMUP_EPOCHS",
        type=int,
        help="linear warmup epochs before cosine decay",
    )
    add_env_argument(
        parser,
        "--warm-restarts",
        env_var="DENOISR_TRAIN_WARM_RESTARTS",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use cosine annealing with warm restarts",
    )
    add_env_argument(
        parser,
        "--threat-weight",
        env_var="DENOISR_TRAIN_THREAT_WEIGHT",
        type=float,
        help="loss weight for threat prediction auxiliary head",
    )
    add_env_argument(
        parser,
        "--policy-weight",
        env_var="DENOISR_TRAIN_POLICY_WEIGHT",
        type=float,
        help="loss weight for policy cross-entropy",
    )
    add_env_argument(
        parser,
        "--value-weight",
        env_var="DENOISR_TRAIN_VALUE_WEIGHT",
        type=float,
        help="loss weight for value cross-entropy",
    )
    add_env_argument(
        parser,
        "--consistency-weight",
        env_var="DENOISR_TRAIN_CONSISTENCY_WEIGHT",
        type=float,
        help="loss weight for consistency",
    )
    add_env_argument(
        parser,
        "--diffusion-weight",
        env_var="DENOISR_TRAIN_DIFFUSION_WEIGHT",
        type=float,
        help="loss weight for diffusion denoising",
    )
    add_env_argument(
        parser,
        "--reward-weight",
        env_var="DENOISR_TRAIN_REWARD_WEIGHT",
        type=float,
        help="loss weight for reward prediction",
    )
    add_env_argument(
        parser,
        "--ply-weight",
        env_var="DENOISR_TRAIN_PLY_WEIGHT",
        type=float,
        help="loss weight for game-length prediction",
    )
    add_env_argument(
        parser,
        "--illegal-penalty-weight",
        env_var="DENOISR_TRAIN_ILLEGAL_PENALTY_WEIGHT",
        type=float,
        help="L2 penalty weight on illegal-move logits",
    )
    add_env_argument(
        parser,
        "--harmony-dream",
        env_var="DENOISR_TRAIN_HARMONY_DREAM",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable HarmonyDream dynamic loss balancing",
    )
    add_env_argument(
        parser,
        "--harmony-ema-decay",
        env_var="DENOISR_TRAIN_HARMONY_EMA_DECAY",
        type=float,
        help="EMA decay for HarmonyDream loss tracking",
    )
    add_env_argument(
        parser,
        "--curriculum-initial-fraction",
        env_var="DENOISR_TRAIN_CURRICULUM_INITIAL_FRACTION",
        type=float,
        help="fraction of diffusion steps at curriculum start",
    )
    add_env_argument(
        parser,
        "--curriculum-growth",
        env_var="DENOISR_TRAIN_CURRICULUM_GROWTH",
        type=float,
        help="per-epoch multiplier for curriculum steps",
    )
    add_env_argument(
        parser,
        "--workers",
        env_var="DENOISR_WORKERS",
        type=int,
        help="DataLoader worker processes",
    )
    add_env_argument(
        parser,
        "--tqdm",
        env_var="DENOISR_TQDM",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="show tqdm progress bars",
    )
    add_env_argument(
        parser,
        "--phase1-gate",
        env_var="DENOISR_PHASE1_GATE",
        type=float,
        help="top-1 accuracy to pass Phase 1 gate",
    )
    add_env_argument(
        parser,
        "--phase2-gate",
        env_var="DENOISR_PHASE2_GATE",
        type=float,
        help="diffusion improvement pp to pass Phase 2 gate",
    )
    add_env_argument(
        parser,
        "--grok-tracking",
        env_var="DENOISR_GROK_TRACKING",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable grokking detection metrics",
    )
    add_env_argument(
        parser,
        "--grok-erank-freq",
        env_var="DENOISR_GROK_ERANK_FREQ",
        type=int,
        help="effective rank computation frequency in steps",
    )
    add_env_argument(
        parser,
        "--grok-spectral-freq",
        env_var="DENOISR_GROK_SPECTRAL_FREQ",
        type=int,
        help="spectral norm / HTSR alpha frequency",
    )
    add_env_argument(
        parser,
        "--grok-onset-threshold",
        env_var="DENOISR_GROK_ONSET_THRESHOLD",
        type=float,
        help="weight norm ratio for onset detection",
    )
    add_env_argument(
        parser,
        "--grokfast",
        env_var="DENOISR_GROKFAST",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable Grokfast EMA gradient filtering",
    )
    add_env_argument(
        parser,
        "--grokfast-alpha",
        env_var="DENOISR_GROKFAST_ALPHA",
        type=float,
        help="Grokfast EMA decay rate",
    )
    add_env_argument(
        parser,
        "--grokfast-lamb",
        env_var="DENOISR_GROKFAST_LAMB",
        type=float,
        help="Grokfast amplification factor",
    )
    add_env_argument(
        parser,
        "--ema-decay",
        env_var="DENOISR_EMA_DECAY",
        type=float,
        help="EMA decay for shadow model evaluation",
    )


def add_phase3_args(parser: ArgumentParser) -> None:
    """Register Phase 3 flags; all values must come from env or explicit CLI."""
    add_env_argument(
        parser,
        "--c-puct",
        env_var="DENOISR_PHASE3_C_PUCT",
        type=float,
        help="MCTS UCB exploration constant",
    )
    add_env_argument(
        parser,
        "--dirichlet-alpha",
        env_var="DENOISR_PHASE3_DIRICHLET_ALPHA",
        type=float,
        help="Dirichlet noise alpha for root exploration",
    )
    add_env_argument(
        parser,
        "--dirichlet-epsilon",
        env_var="DENOISR_PHASE3_DIRICHLET_EPSILON",
        type=float,
        help="fraction of root prior replaced by Dirichlet noise",
    )
    add_env_argument(
        parser,
        "--temperature-base",
        env_var="DENOISR_PHASE3_TEMPERATURE_BASE",
        type=float,
        help="base temperature for move selection",
    )
    add_env_argument(
        parser,
        "--temperature-explore-moves",
        env_var="DENOISR_PHASE3_TEMPERATURE_EXPLORE_MOVES",
        type=int,
        help="moves per game using full temperature",
    )
    add_env_argument(
        parser,
        "--temperature-generation-decay",
        env_var="DENOISR_PHASE3_TEMPERATURE_GENERATION_DECAY",
        type=float,
        help="per-generation temperature decay",
    )
    add_env_argument(
        parser,
        "--max-moves",
        env_var="DENOISR_PHASE3_MAX_MOVES",
        type=int,
        help="maximum moves per self-play game",
    )
    add_env_argument(
        parser,
        "--reanalyse-simulations",
        env_var="DENOISR_PHASE3_REANALYSE_SIMULATIONS",
        type=int,
        help="MCTS sims for MuZero Reanalyse",
    )


_EnvValueParser = Callable[[str], Any]
_EnvSpec = tuple[str, _EnvValueParser]


def _parse_env_int(raw: str) -> int:
    return int(raw.replace("_", ""))


def _parse_env_float(raw: str) -> float:
    return float(raw.replace("_", ""))


def _parse_env_bool(raw: str) -> bool:
    norm = raw.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"invalid boolean value {raw!r} (expected one of 1/0,true/false,yes/no,on/off)"
    )


_MODEL_REQUIRED_ENV_SPECS: tuple[_EnvSpec, ...] = (
    ("DENOISR_MODEL_D_S", _parse_env_int),
    ("DENOISR_MODEL_NUM_HEADS", _parse_env_int),
    ("DENOISR_MODEL_NUM_LAYERS", _parse_env_int),
    ("DENOISR_MODEL_FFN_DIM", _parse_env_int),
    ("DENOISR_MODEL_NUM_TIMESTEPS", _parse_env_int),
    ("DENOISR_MODEL_WORLD_MODEL_LAYERS", _parse_env_int),
    ("DENOISR_MODEL_DIFFUSION_LAYERS", _parse_env_int),
    ("DENOISR_MODEL_PROJ_DIM", _parse_env_int),
    ("DENOISR_MODEL_GRADIENT_CHECKPOINTING", _parse_env_bool),
)

_TRAINING_REQUIRED_ENV_SPECS: tuple[_EnvSpec, ...] = (
    ("DENOISR_TRAIN_MAX_GRAD_NORM", _parse_env_float),
    ("DENOISR_TRAIN_WEIGHT_DECAY", _parse_env_float),
    ("DENOISR_TRAIN_ENCODER_LR_MULTIPLIER", _parse_env_float),
    ("DENOISR_TRAIN_MIN_LR", _parse_env_float),
    ("DENOISR_TRAIN_WARMUP_EPOCHS", _parse_env_int),
    ("DENOISR_TRAIN_WARM_RESTARTS", _parse_env_bool),
    ("DENOISR_TRAIN_THREAT_WEIGHT", _parse_env_float),
    ("DENOISR_TRAIN_POLICY_WEIGHT", _parse_env_float),
    ("DENOISR_TRAIN_VALUE_WEIGHT", _parse_env_float),
    ("DENOISR_TRAIN_CONSISTENCY_WEIGHT", _parse_env_float),
    ("DENOISR_TRAIN_DIFFUSION_WEIGHT", _parse_env_float),
    ("DENOISR_TRAIN_REWARD_WEIGHT", _parse_env_float),
    ("DENOISR_TRAIN_PLY_WEIGHT", _parse_env_float),
    ("DENOISR_TRAIN_ILLEGAL_PENALTY_WEIGHT", _parse_env_float),
    ("DENOISR_TRAIN_HARMONY_DREAM", _parse_env_bool),
    ("DENOISR_TRAIN_HARMONY_EMA_DECAY", _parse_env_float),
    ("DENOISR_TRAIN_CURRICULUM_INITIAL_FRACTION", _parse_env_float),
    ("DENOISR_TRAIN_CURRICULUM_GROWTH", _parse_env_float),
    ("DENOISR_WORKERS", _parse_env_int),
    ("DENOISR_TQDM", _parse_env_bool),
    ("DENOISR_PHASE1_GATE", _parse_env_float),
    ("DENOISR_PHASE2_GATE", _parse_env_float),
    ("DENOISR_GROK_TRACKING", _parse_env_bool),
    ("DENOISR_GROK_ERANK_FREQ", _parse_env_int),
    ("DENOISR_GROK_SPECTRAL_FREQ", _parse_env_int),
    ("DENOISR_GROK_ONSET_THRESHOLD", _parse_env_float),
    ("DENOISR_GROKFAST", _parse_env_bool),
    ("DENOISR_GROKFAST_ALPHA", _parse_env_float),
    ("DENOISR_GROKFAST_LAMB", _parse_env_float),
    ("DENOISR_EMA_DECAY", _parse_env_float),
)

_PHASE3_REQUIRED_ENV_SPECS: tuple[_EnvSpec, ...] = (
    ("DENOISR_PHASE3_C_PUCT", _parse_env_float),
    ("DENOISR_PHASE3_DIRICHLET_ALPHA", _parse_env_float),
    ("DENOISR_PHASE3_DIRICHLET_EPSILON", _parse_env_float),
    ("DENOISR_PHASE3_TEMPERATURE_BASE", _parse_env_float),
    ("DENOISR_PHASE3_TEMPERATURE_EXPLORE_MOVES", _parse_env_int),
    ("DENOISR_PHASE3_TEMPERATURE_GENERATION_DECAY", _parse_env_float),
    ("DENOISR_PHASE3_MAX_MOVES", _parse_env_int),
    ("DENOISR_PHASE3_REANALYSE_SIMULATIONS", _parse_env_int),
)


def required_env_vars(*, include_phase3: bool = True) -> tuple[str, ...]:
    """Required env vars for model/training configuration."""
    specs: list[_EnvSpec] = list(_MODEL_REQUIRED_ENV_SPECS) + list(
        _TRAINING_REQUIRED_ENV_SPECS
    )
    if include_phase3:
        specs.extend(_PHASE3_REQUIRED_ENV_SPECS)
    return tuple(name for name, _parser in specs)


def validate_required_env(*, include_phase3: bool = True) -> None:
    """Fail fast if required env vars are missing or malformed."""
    specs: list[_EnvSpec] = list(_MODEL_REQUIRED_ENV_SPECS) + list(
        _TRAINING_REQUIRED_ENV_SPECS
    )
    if include_phase3:
        specs.extend(_PHASE3_REQUIRED_ENV_SPECS)

    missing: list[str] = []
    invalid: list[str] = []
    for env_name, parser in specs:
        raw = os.environ.get(env_name)
        if raw is None or raw.strip() == "":
            missing.append(env_name)
            continue
        try:
            parser(raw)
        except ValueError as exc:
            invalid.append(f"{env_name}={raw!r}: {exc}")

    if missing or invalid:
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {', '.join(sorted(missing))}")
        if invalid:
            parts.append(f"invalid: {'; '.join(invalid)}")
        raise ValueError(f"Environment validation failed for training config: {' | '.join(parts)}")


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
    cfg: ModelConfig,
    args: Namespace,
    device: torch.device,
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
