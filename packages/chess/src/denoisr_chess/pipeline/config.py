"""Pipeline configuration loaded strictly from environment variables."""

import os
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DataConfig:
    pgn_url: str
    pgn_path: str
    stockfish_path: str
    stockfish_depth: int
    max_examples: int
    workers: int
    chunksize: int
    chunk_examples: int


@dataclass(frozen=True)
class ModelSectionConfig:
    d_s: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    num_timesteps: int


@dataclass(frozen=True)
class Phase1Config:
    epochs: int
    lr: float
    batch_size: int
    holdout_frac: float
    warmup_epochs: int
    weight_decay: float


@dataclass(frozen=True)
class Phase2Config:
    epochs: int
    lr: float
    batch_size: int
    seq_len: int
    max_trajectories: int


@dataclass(frozen=True)
class Phase3Config:
    generations: int
    games_per_gen: int
    reanalyse_per_gen: int
    mcts_sims: int
    buffer_capacity: int
    alpha_generations: int
    lr: float
    train_batch_size: int
    diffusion_steps: int
    aux_updates_per_gen: int
    aux_batch_size: int
    aux_seq_len: int
    aux_lr: float | None
    self_play_workers: int
    reanalyse_workers: int
    save_every: int


@dataclass(frozen=True)
class OutputConfig:
    dir: str
    run_name: str


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    model: ModelSectionConfig
    phase1: Phase1Config
    phase2: Phase2Config
    phase3: Phase3Config
    output: OutputConfig


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
    ("DENOISR_CHUNKSIZE", "chunksize", _parse_int),
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
_OUTPUT_REQUIRED_ENV_SPECS: tuple[_EnvFieldSpec, ...] = (
    ("DENOISR_OUTPUT_DIR", "dir", _parse_str),
    ("DENOISR_RUN_NAME", "run_name", _parse_str),
)
_OUTPUT_OPTIONAL_ENV_SPECS: tuple[_EnvFieldSpec, ...] = ()


def _load_required_section(
    section_name: str,
    section_type: type[Any],
    specs: tuple[_EnvFieldSpec, ...],
) -> Any:
    values: dict[str, Any] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for env_name, field_name, parser in specs:
        raw = _env_str(env_name)
        if raw is None:
            missing.append(env_name)
            continue
        try:
            values[field_name] = parser(raw)
        except ValueError as exc:
            invalid.append(f"{env_name}={raw!r}: {exc}")
    if missing or invalid:
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {', '.join(sorted(missing))}")
        if invalid:
            parts.append(f"invalid: {'; '.join(invalid)}")
        details = " | ".join(parts)
        raise ValueError(f"Environment validation failed for {section_name}: {details}")
    return section_type(**values)


def _apply_optional_overrides(
    section: Any,
    specs: tuple[_EnvFieldSpec, ...],
) -> Any:
    updates: dict[str, Any] = {}
    for env_name, field_name, parser in specs:
        raw = _env_str(env_name)
        if raw is None:
            continue
        try:
            updates[field_name] = parser(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid value in {env_name}={raw!r}") from exc
    if not updates:
        return section
    return type(section)(**{**section.__dict__, **updates})


def required_env_vars() -> tuple[str, ...]:
    """All env vars required to load PipelineConfig."""
    sections = (
        _DATA_ENV_SPECS,
        _MODEL_ENV_SPECS,
        _PHASE1_ENV_SPECS,
        _PHASE2_ENV_SPECS,
        _PHASE3_ENV_SPECS,
        _OUTPUT_REQUIRED_ENV_SPECS,
    )
    return tuple(env_name for specs in sections for env_name, _field, _parser in specs)


def load_config() -> PipelineConfig:
    """Load PipelineConfig strictly from env; fail fast on missing/invalid values."""
    return PipelineConfig(
        data=_load_required_section("data", DataConfig, _DATA_ENV_SPECS),
        model=_load_required_section("model", ModelSectionConfig, _MODEL_ENV_SPECS),
        phase1=_load_required_section("phase1", Phase1Config, _PHASE1_ENV_SPECS),
        phase2=_load_required_section("phase2", Phase2Config, _PHASE2_ENV_SPECS),
        phase3=_load_required_section("phase3", Phase3Config, _PHASE3_ENV_SPECS),
        output=_apply_optional_overrides(
            _load_required_section(
                "output",
                OutputConfig,
                _OUTPUT_REQUIRED_ENV_SPECS,
            ),
            _OUTPUT_OPTIONAL_ENV_SPECS,
        ),
    )

