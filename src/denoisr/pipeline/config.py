"""Pipeline configuration: frozen dataclasses + TOML loader."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    pgn_url: str = (
        "https://database.lichess.org/standard/"
        "lichess_db_standard_rated_2025-01.pgn.zst"
    )
    pgn_path: str = "data/lichess.pgn.zst"
    stockfish_path: str = ""
    stockfish_depth: int = 10
    max_examples: int = 2_000_000
    tactical_fraction: float = 0.25
    workers: int = 0



@dataclass(frozen=True)
class ModelSectionConfig:
    d_s: int = 256
    num_heads: int = 8
    num_layers: int = 15
    ffn_dim: int = 1024
    num_timesteps: int = 100


@dataclass(frozen=True)
class Phase1Config:
    lr: float = 3e-4
    batch_size: int = 1024
    warmup_epochs: int = 5
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class Phase2Config:
    epochs: int = 200
    lr: float = 3e-4
    batch_size: int = 1024
    seq_len: int = 10
    max_trajectories: int = 50_000


@dataclass(frozen=True)
class Phase3Config:
    generations: int = 1000
    games_per_gen: int = 100
    mcts_sims: int = 800


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


def load_config(path: Path) -> PipelineConfig:
    """Load pipeline config from TOML, using defaults for missing fields."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return PipelineConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelSectionConfig(**raw.get("model", {})),
        phase1=Phase1Config(**raw.get("phase1", {})),
        phase2=Phase2Config(**raw.get("phase2", {})),
        phase3=Phase3Config(**raw.get("phase3", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
