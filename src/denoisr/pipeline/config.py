"""Pipeline configuration: frozen dataclasses + TOML loader."""

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DataConfig:
    pgn_url: str = (
        "https://database.lichess.org/standard/"
        "lichess_db_standard_rated_2025-01.pgn.zst"
    )
    pgn_path: str = "data/lichess.pgn.zst"
    stockfish_path: str = ""
    stockfish_depth: int = 10
    max_examples: int = 1_000_000
    workers: int = 48
    scratch_dir: str = "outputs/scratch"
    chunk_examples: int = 250_000


@dataclass(frozen=True)
class ModelSectionConfig:
    d_s: int = 256
    num_heads: int = 8
    num_layers: int = 15
    ffn_dim: int = 1024
    num_timesteps: int = 100


@dataclass(frozen=True)
class Phase1Config:
    epochs: int = 80
    lr: float = 3e-4
    batch_size: int = 128
    holdout_frac: float = 0.05
    warmup_epochs: int = 5
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class Phase2Config:
    epochs: int = 120
    lr: float = 3e-4
    batch_size: int = 64
    seq_len: int = 10
    max_trajectories: int = 30_000


@dataclass(frozen=True)
class Phase3Config:
    generations: int = 400
    games_per_gen: int = 64
    reanalyse_per_gen: int = 32
    mcts_sims: int = 400
    buffer_capacity: int = 50_000
    alpha_generations: int = 40
    lr: float = 1e-4
    train_batch_size: int = 128
    diffusion_steps: int = 8
    aux_updates_per_gen: int = 1
    aux_batch_size: int = 64
    aux_seq_len: int = 10
    aux_lr: float | None = None
    self_play_workers: int = 0
    reanalyse_workers: int = 0
    save_every: int = 10


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "outputs/"
    run_name: str = ""


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelSectionConfig = field(default_factory=ModelSectionConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    output: OutputConfig = field(default_factory=OutputConfig)


def _env_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value if value != "" else None


def _parse_int(raw: str) -> int:
    return int(raw.replace("_", ""))


def _parse_float(raw: str) -> float:
    return float(raw.replace("_", ""))


def _parse_str(raw: str) -> str:
    return raw


_EnvFieldSpec = tuple[str, str, Callable[[str], Any]]

_DATA_ENV_SPECS: tuple[_EnvFieldSpec, ...] = (
    ("DENOISR_PGN_URL", "pgn_url", _parse_str),
    ("DENOISR_PGN_PATH", "pgn_path", _parse_str),
    ("DENOISR_STOCKFISH_PATH", "stockfish_path", _parse_str),
    ("DENOISR_STOCKFISH_DEPTH", "stockfish_depth", _parse_int),
    ("DENOISR_MAX_EXAMPLES", "max_examples", _parse_int),
    ("DENOISR_WORKERS", "workers", _parse_int),
    ("DENOISR_SCRATCH_DIR", "scratch_dir", _parse_str),
    ("DENOISR_CHUNK_EXAMPLES", "chunk_examples", _parse_int),
)
_MODEL_ENV_SPECS: tuple[_EnvFieldSpec, ...] = (
    ("DENOISR_MODEL_D_S", "d_s", _parse_int),
    ("DENOISR_MODEL_NUM_HEADS", "num_heads", _parse_int),
    ("DENOISR_MODEL_NUM_LAYERS", "num_layers", _parse_int),
    ("DENOISR_MODEL_FFN_DIM", "ffn_dim", _parse_int),
    ("DENOISR_MODEL_NUM_TIMESTEPS", "num_timesteps", _parse_int),
)
_PHASE1_ENV_SPECS: tuple[_EnvFieldSpec, ...] = (
    ("DENOISR_PHASE1_EPOCHS", "epochs", _parse_int),
    ("DENOISR_PHASE1_LR", "lr", _parse_float),
    ("DENOISR_PHASE1_BATCH_SIZE", "batch_size", _parse_int),
    ("DENOISR_PHASE1_HOLDOUT_FRAC", "holdout_frac", _parse_float),
    ("DENOISR_PHASE1_WARMUP_EPOCHS", "warmup_epochs", _parse_int),
    ("DENOISR_PHASE1_WEIGHT_DECAY", "weight_decay", _parse_float),
)
_PHASE2_ENV_SPECS: tuple[_EnvFieldSpec, ...] = (
    ("DENOISR_PHASE2_EPOCHS", "epochs", _parse_int),
    ("DENOISR_PHASE2_LR", "lr", _parse_float),
    ("DENOISR_PHASE2_BATCH_SIZE", "batch_size", _parse_int),
    ("DENOISR_PHASE2_SEQ_LEN", "seq_len", _parse_int),
    ("DENOISR_PHASE2_MAX_TRAJECTORIES", "max_trajectories", _parse_int),
)
_PHASE3_ENV_SPECS: tuple[_EnvFieldSpec, ...] = (
    ("DENOISR_PHASE3_GENERATIONS", "generations", _parse_int),
    ("DENOISR_PHASE3_GAMES_PER_GEN", "games_per_gen", _parse_int),
    ("DENOISR_PHASE3_REANALYSE_PER_GEN", "reanalyse_per_gen", _parse_int),
    ("DENOISR_PHASE3_MCTS_SIMS", "mcts_sims", _parse_int),
    ("DENOISR_PHASE3_BUFFER_CAPACITY", "buffer_capacity", _parse_int),
    ("DENOISR_PHASE3_ALPHA_GENERATIONS", "alpha_generations", _parse_int),
    ("DENOISR_PHASE3_LR", "lr", _parse_float),
    ("DENOISR_PHASE3_TRAIN_BATCH_SIZE", "train_batch_size", _parse_int),
    ("DENOISR_PHASE3_DIFFUSION_STEPS", "diffusion_steps", _parse_int),
    ("DENOISR_PHASE3_AUX_UPDATES_PER_GEN", "aux_updates_per_gen", _parse_int),
    ("DENOISR_PHASE3_AUX_BATCH_SIZE", "aux_batch_size", _parse_int),
    ("DENOISR_PHASE3_AUX_SEQ_LEN", "aux_seq_len", _parse_int),
    ("DENOISR_PHASE3_AUX_LR", "aux_lr", _parse_float),
    ("DENOISR_PHASE3_SELF_PLAY_WORKERS", "self_play_workers", _parse_int),
    ("DENOISR_PHASE3_REANALYSE_WORKERS", "reanalyse_workers", _parse_int),
    ("DENOISR_PHASE3_SAVE_EVERY", "save_every", _parse_int),
)
_OUTPUT_ENV_SPECS: tuple[_EnvFieldSpec, ...] = (
    ("DENOISR_OUTPUT_DIR", "dir", _parse_str),
    ("DENOISR_RUN_NAME", "run_name", _parse_str),
)


def _apply_section_overrides(section: Any, specs: tuple[_EnvFieldSpec, ...]) -> Any:
    updates: dict[str, Any] = {}
    for env_name, field_name, parser in specs:
        raw = _env_str(env_name)
        if raw is None:
            continue
        try:
            updates[field_name] = parser(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid value in {env_name}={raw!r}") from exc
    return replace(section, **updates) if updates else section


def _apply_env_overrides(cfg: PipelineConfig) -> PipelineConfig:
    return PipelineConfig(
        data=_apply_section_overrides(cfg.data, _DATA_ENV_SPECS),
        model=_apply_section_overrides(cfg.model, _MODEL_ENV_SPECS),
        phase1=_apply_section_overrides(cfg.phase1, _PHASE1_ENV_SPECS),
        phase2=_apply_section_overrides(cfg.phase2, _PHASE2_ENV_SPECS),
        phase3=_apply_section_overrides(cfg.phase3, _PHASE3_ENV_SPECS),
        output=_apply_section_overrides(cfg.output, _OUTPUT_ENV_SPECS),
    )


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load pipeline config from optional TOML and env overrides."""
    raw: dict[str, dict[str, object]] = {}
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    cfg = PipelineConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelSectionConfig(**raw.get("model", {})),
        phase1=Phase1Config(**raw.get("phase1", {})),
        phase2=Phase2Config(**raw.get("phase2", {})),
        phase3=Phase3Config(**raw.get("phase3", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
    return _apply_env_overrides(cfg)
