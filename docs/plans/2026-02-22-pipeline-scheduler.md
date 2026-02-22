# Pipeline Scheduler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A single `denoisr-train` command that orchestrates the entire training pipeline from PGN download through Phase 3, with Elo curriculum learning and TOML-based configuration.

**Architecture:** A `src/denoisr/pipeline/` package with TOML config parsing, JSON state persistence, step functions, and a runner. Existing training functions are called directly (not via subprocess). SimpleBoardEncoder and in-memory generation paths are removed as part of simplification.

**Tech Stack:** Python 3.14 `tomllib` (stdlib), existing torch/chess infrastructure, frozen dataclasses for config.

---

## Part 1: Simplifications

### Task 1: Remove SimpleBoardEncoder

Replace all 13 references to `SimpleBoardEncoder` with `ExtendedBoardEncoder`. Delete the file. Remove the `--num-planes` CLI flag. Hardcode `num_planes = 122` as a constant.

**Files:**
- Delete: `src/denoisr/data/board_encoder.py`
- Modify: `src/denoisr/scripts/config.py` (lines 17, 38, 295-299, 350-353, 623)
- Modify: `src/denoisr/scripts/train_phase2.py` (line 21)
- Modify: `src/denoisr/training/reanalyse.py` (line 6)
- Modify: `src/denoisr/training/self_play.py` (line 8)
- Modify: `src/denoisr/inference/mlx_engine.py` (line 284)
- Modify: `tests/test_data/test_board_encoder.py`
- Modify: `tests/test_data/test_dataset.py` (line 5)
- Modify: `tests/test_inference/test_engine.py` (line 5)
- Modify: `tests/test_inference/test_diffusion_engine.py` (line 5)
- Modify: `tests/test_training/test_augmentation.py` (line 4)
- Modify: `tests/test_training/test_reanalyse.py` (line 5)
- Modify: `tests/test_training/test_self_play.py` (line 4)
- Modify: `tests/test_training/test_phase2_trainer.py` (line 13)

**Step 1: Update config.py**

In `src/denoisr/scripts/config.py`:
- Remove `from denoisr.data.board_encoder import SimpleBoardEncoder` (line 17)
- Change `ModelConfig.num_planes` from `int = 122` to a constant: remove the field, add `NUM_PLANES = 122` module constant
- Actually, keep the field for checkpoint compatibility but remove the CLI flag
- In `build_board_encoder()` (line 295-299), simplify to always return `ExtendedBoardEncoder()`
- Remove `--num-planes` from `add_model_args()` (lines 350-353)
- Remove `num_planes=args.num_planes` from `config_from_args()` (line 623)

**Step 2: Update all imports**

In every file that imports `SimpleBoardEncoder`:
- Replace `SimpleBoardEncoder` with `ExtendedBoardEncoder`
- Update `from denoisr.data.board_encoder import SimpleBoardEncoder` to `from denoisr.data.extended_board_encoder import ExtendedBoardEncoder`
- In type hints like `SimpleBoardEncoder | ExtendedBoardEncoder`, simplify to just `ExtendedBoardEncoder`

**Step 3: Update tests**

- `tests/test_data/test_board_encoder.py`: Rewrite to test `ExtendedBoardEncoder` only (or delete if `test_extended_board_encoder.py` already covers it)
- All other test files: swap imports

**Step 4: Delete board_encoder.py**

```bash
git rm src/denoisr/data/board_encoder.py
```

**Step 5: Run tests and commit**

```bash
uv run pytest tests/ -x -q
git add -u && git add tests/
git commit -m "refactor: remove SimpleBoardEncoder, hardcode 122-plane ExtendedBoardEncoder"
```

---

### Task 2: Remove in-memory generation path

Delete `generate_examples()`, `stack_examples()`, and `unstack_examples()` from `generate_data.py`. Update callers.

**Files:**
- Modify: `src/denoisr/scripts/generate_data.py` (delete lines 375-479)
- Modify: `src/denoisr/scripts/train_phase1.py` (line 35, uses `unstack_examples`)
- Modify: `tests/test_types/test_training_types.py` (lines 68, 83, 98)

**Step 1: Check train_phase1.py usage**

Read `src/denoisr/scripts/train_phase1.py` to understand how `unstack_examples` is used. If it only converts loaded .pt data to TrainingExample list and the training loop just needs the raw tensors, we can inline the tensor loading.

**Step 2: Update train_phase1.py**

If `unstack_examples` is used just to get the list for grokking holdout splits, refactor to work with the raw tensor dict directly:
```python
# Instead of: examples = unstack_examples(data)
# Just use: boards = data["boards"], policies = data["policies"], values = data["values"]
```

**Step 3: Delete the three functions from generate_data.py**

Remove `generate_examples()`, `stack_examples()`, `unstack_examples()`, and the `_extract_positions()` helper they depend on.

**Step 4: Update tests**

Remove or rewrite tests in `test_training_types.py` that use `stack_examples`/`unstack_examples`.

**Step 5: Run tests and commit**

```bash
uv run pytest tests/ -x -q
git add -u
git commit -m "refactor: remove in-memory data generation path, keep disk-backed memmap only"
```

---

### Task 3: Consolidate streaming functions

Merge `_stream_positions_elo_stratified()` into `_stream_positions()` by adding optional `elo_dir` and `tactical_fraction` parameters.

**Files:**
- Modify: `src/denoisr/scripts/generate_data.py`

**Step 1: Refactor _stream_positions()**

Add parameters to `_stream_positions()`:
```python
def _stream_positions(
    pgn_path: Path,
    max_positions: int,
    min_elo: int | None = None,
    sample_rate: float = 1.0,
    elo_dir: Path | None = None,
    tactical_fraction: float = 0.0,
    seed: int | None = None,
) -> Iterator[_PositionMeta]:
```

When `elo_dir` is provided, iterate over sorted bucket files round-robin with tactical enrichment. Otherwise, stream from `pgn_path` as before.

**Step 2: Delete _stream_positions_elo_stratified()**

Remove the standalone function. Update `generate_to_file()` to call the unified function.

**Step 3: Run tests and commit**

```bash
uv run pytest tests/ -x -q
git add -u
git commit -m "refactor: consolidate position streaming into single function"
```

---

## Part 2: Pipeline Package

### Task 4: PipelineConfig TOML parser

**Files:**
- Create: `src/denoisr/pipeline/__init__.py`
- Create: `src/denoisr/pipeline/config.py`
- Create: `tests/test_pipeline/__init__.py`
- Create: `tests/test_pipeline/test_config.py`

**Step 1: Write failing tests**

```python
# tests/test_pipeline/test_config.py
import tomllib
from denoisr.pipeline.config import PipelineConfig, load_config

def test_load_minimal_config(tmp_path):
    """Empty TOML uses all defaults."""
    cfg_path = tmp_path / "pipeline.toml"
    cfg_path.write_text("")
    cfg = load_config(cfg_path)
    assert cfg.data.stockfish_depth == 10
    assert cfg.elo_curriculum.tiers == [800, 1200, 1600, 2000, 2400]
    assert cfg.phase1.lr == 3e-4

def test_load_partial_config(tmp_path):
    """Partial TOML overrides only specified fields."""
    cfg_path = tmp_path / "pipeline.toml"
    cfg_path.write_text('[phase1]\nlr = 1e-3\n')
    cfg = load_config(cfg_path)
    assert cfg.phase1.lr == 1e-3
    assert cfg.phase1.batch_size == 1024  # default preserved

def test_elo_tiers_custom(tmp_path):
    cfg_path = tmp_path / "pipeline.toml"
    cfg_path.write_text('[elo_curriculum]\ntiers = [1000, 1500, 2000]\n')
    cfg = load_config(cfg_path)
    assert cfg.elo_curriculum.tiers == [1000, 1500, 2000]
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_pipeline/test_config.py -v
```

**Step 3: Implement PipelineConfig**

```python
# src/denoisr/pipeline/config.py
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class DataConfig:
    pgn_url: str = "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
    pgn_path: str = "data/lichess.pgn.zst"
    sorted_dir: str = "data/sorted/"
    stockfish_path: str = ""
    stockfish_depth: int = 10
    examples_per_tier: int = 200_000
    tactical_fraction: float = 0.25
    workers: int = 0

@dataclass(frozen=True)
class EloCurriculumConfig:
    tiers: list[int] = field(default_factory=lambda: [800, 1200, 1600, 2000, 2400])
    gate_accuracy: float = 0.50
    max_epochs_per_tier: int = 100

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
    batch_size: int = 128
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
    elo_curriculum: EloCurriculumConfig = field(default_factory=EloCurriculumConfig)
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
        elo_curriculum=EloCurriculumConfig(**raw.get("elo_curriculum", {})),
        model=ModelSectionConfig(**raw.get("model", {})),
        phase1=Phase1Config(**raw.get("phase1", {})),
        phase2=Phase2Config(**raw.get("phase2", {})),
        phase3=Phase3Config(**raw.get("phase3", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
```

**Step 4: Run tests and commit**

```bash
uv run pytest tests/test_pipeline/test_config.py -v
git add src/denoisr/pipeline/ tests/test_pipeline/
git commit -m "feat: add PipelineConfig TOML parser for unified training pipeline"
```

---

### Task 5: PipelineState persistence

**Files:**
- Create: `src/denoisr/pipeline/state.py`
- Create: `tests/test_pipeline/test_state.py`

**Step 1: Write failing tests**

```python
# tests/test_pipeline/test_state.py
from denoisr.pipeline.state import PipelineState

def test_fresh_state():
    state = PipelineState()
    assert state.phase == "init"
    assert state.elo_tier_index == 0

def test_save_and_load(tmp_path):
    state = PipelineState(phase="elo_curriculum", elo_tier_index=2)
    path = tmp_path / "state.json"
    state.save(path)
    loaded = PipelineState.load(path)
    assert loaded.phase == "elo_curriculum"
    assert loaded.elo_tier_index == 2

def test_load_missing_returns_fresh(tmp_path):
    path = tmp_path / "nonexistent.json"
    state = PipelineState.load(path)
    assert state.phase == "init"
```

**Step 2: Implement PipelineState**

```python
# src/denoisr/pipeline/state.py
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

@dataclass
class PipelineState:
    phase: str = "init"
    elo_tier_index: int = 0
    last_checkpoint: str = ""
    last_data: str = ""
    tier_accuracies: dict[str, float] = field(default_factory=dict)
    started_at: str = ""
    updated_at: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
```

**Step 3: Run tests and commit**

```bash
uv run pytest tests/test_pipeline/test_state.py -v
git add src/denoisr/pipeline/state.py tests/test_pipeline/test_state.py
git commit -m "feat: add PipelineState JSON persistence for resume support"
```

---

### Task 6: Pipeline step functions

**Files:**
- Create: `src/denoisr/pipeline/steps.py`
- Create: `tests/test_pipeline/test_steps.py`

**Step 1: Implement step functions**

Each step function takes `(PipelineConfig, PipelineState)` and returns an updated `PipelineState`. Steps call existing library functions directly (not subprocesses).

```python
# src/denoisr/pipeline/steps.py
import logging
import shutil
import subprocess
from pathlib import Path

from denoisr.pipeline.config import PipelineConfig
from denoisr.pipeline.state import PipelineState

log = logging.getLogger(__name__)

def step_fetch_data(cfg: PipelineConfig, state: PipelineState) -> PipelineState:
    """Download PGN if not already present."""
    pgn_path = Path(cfg.data.pgn_path)
    if pgn_path.exists():
        log.info("PGN already exists at %s, skipping download", pgn_path)
        return state
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s ...", cfg.data.pgn_url)
    subprocess.run(
        ["wget", "-q", "-O", str(pgn_path), cfg.data.pgn_url],
        check=True,
    )
    state.phase = "fetched"
    return state

def step_sort_pgn(cfg: PipelineConfig, state: PipelineState) -> PipelineState:
    """Sort PGN into Elo-stratified bucket files."""
    sorted_dir = Path(cfg.data.sorted_dir)
    if sorted_dir.exists() and any(sorted_dir.glob("*.pgn.zst")):
        log.info("Sorted PGN files already exist in %s, skipping", sorted_dir)
        return state
    from denoisr.scripts.sort_pgn import main as sort_main
    # Call sort_pgn programmatically by constructing args
    import sys
    tiers = cfg.elo_curriculum.tiers
    ranges = ",".join(
        f"{tiers[i]}-{tiers[i+1]}" if i < len(tiers) - 1 else f"{tiers[i]}+"
        for i in range(len(tiers))
    )
    # Prepend 0-lowest tier
    ranges = f"0-{tiers[0]}," + ranges
    old_argv = sys.argv
    sys.argv = [
        "denoisr-sort-pgn",
        "--input", cfg.data.pgn_path,
        "--output", cfg.data.sorted_dir,
        "--ranges", ranges,
    ]
    try:
        sort_main()
    finally:
        sys.argv = old_argv
    state.phase = "sorted"
    return state

def step_init_model(cfg: PipelineConfig, state: PipelineState) -> PipelineState:
    """Create random model checkpoint."""
    output_dir = Path(cfg.output.dir)
    model_path = output_dir / "random_model.pt"
    if model_path.exists():
        log.info("Random model already exists at %s, skipping", model_path)
        state.last_checkpoint = str(model_path)
        return state
    from denoisr.scripts.config import ModelConfig, save_checkpoint, build_encoder, build_backbone, build_policy_head, build_value_head, build_world_model, build_diffusion, build_consistency
    mc = ModelConfig(
        d_s=cfg.model.d_s,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        ffn_dim=cfg.model.ffn_dim,
        num_timesteps=cfg.model.num_timesteps,
    )
    encoder = build_encoder(mc)
    backbone = build_backbone(mc)
    policy_head = build_policy_head(mc)
    value_head = build_value_head(mc)
    save_checkpoint(
        model_path, mc,
        encoder=encoder.state_dict(),
        backbone=backbone.state_dict(),
        policy_head=policy_head.state_dict(),
        value_head=value_head.state_dict(),
    )
    log.info("Created random model at %s", model_path)
    state.last_checkpoint = str(model_path)
    state.phase = "initialized"
    return state

def step_generate_tier_data(
    cfg: PipelineConfig,
    state: PipelineState,
    min_elo: int,
    tier_index: int,
) -> PipelineState:
    """Generate training data for a specific Elo tier."""
    output_dir = Path(cfg.output.dir)
    data_path = output_dir / f"tier_{min_elo}_data.pt"
    if data_path.exists():
        log.info("Tier %d data already exists at %s, skipping", min_elo, data_path)
        state.last_data = str(data_path)
        return state
    from denoisr.scripts.generate_data import generate_to_file
    stockfish_path = cfg.data.stockfish_path or shutil.which("stockfish") or ""
    workers = cfg.data.workers or (((__import__("os").cpu_count() or 1) * 2) + 1)
    generate_to_file(
        pgn_path=Path(cfg.data.pgn_path),
        output_path=data_path,
        stockfish_path=stockfish_path,
        stockfish_depth=cfg.data.stockfish_depth,
        max_examples=cfg.data.examples_per_tier,
        num_workers=workers,
        min_elo=min_elo,
        tactical_fraction=cfg.data.tactical_fraction,
    )
    state.last_data = str(data_path)
    return state

def step_train_phase1_tier(
    cfg: PipelineConfig,
    state: PipelineState,
    tier_index: int,
    min_elo: int,
) -> PipelineState:
    """Train Phase 1 on a single Elo tier until gate or max epochs."""
    # This calls the core training loop from supervised_trainer
    # Uses state.last_checkpoint as input, saves to phase1_tier{N}.pt
    output_dir = Path(cfg.output.dir)
    output_path = output_dir / f"phase1_tier{tier_index}.pt"
    # Import and run phase 1 training programmatically
    # (Details in implementation — calls SupervisedTrainer directly)
    state.last_checkpoint = str(output_path)
    state.elo_tier_index = tier_index
    return state

def step_train_phase2(cfg: PipelineConfig, state: PipelineState) -> PipelineState:
    """Train Phase 2 (diffusion bootstrapping)."""
    output_dir = Path(cfg.output.dir)
    output_path = output_dir / "phase2.pt"
    # Import and run phase 2 training programmatically
    state.last_checkpoint = str(output_path)
    state.phase = "phase2_complete"
    return state

def step_train_phase3(cfg: PipelineConfig, state: PipelineState) -> PipelineState:
    """Train Phase 3 (RL self-play)."""
    output_dir = Path(cfg.output.dir)
    output_path = output_dir / "phase3.pt"
    # Import and run phase 3 training programmatically
    state.last_checkpoint = str(output_path)
    state.phase = "complete"
    return state
```

**Step 2: Write basic tests**

```python
# tests/test_pipeline/test_steps.py
from denoisr.pipeline.config import PipelineConfig
from denoisr.pipeline.state import PipelineState
from denoisr.pipeline.steps import step_init_model

def test_step_init_model_creates_checkpoint(tmp_path):
    cfg = PipelineConfig(output=OutputConfig(dir=str(tmp_path)))
    state = PipelineState()
    state = step_init_model(cfg, state)
    assert (tmp_path / "random_model.pt").exists()
    assert state.last_checkpoint == str(tmp_path / "random_model.pt")
```

**Step 3: Run tests and commit**

```bash
uv run pytest tests/test_pipeline/ -v
git add src/denoisr/pipeline/steps.py tests/test_pipeline/test_steps.py
git commit -m "feat: add pipeline step functions for each training stage"
```

---

### Task 7: Pipeline runner and CLI entry point

**Files:**
- Create: `src/denoisr/pipeline/runner.py`
- Create: `src/denoisr/scripts/train.py`
- Modify: `pyproject.toml` (add entry point)
- Create: `tests/test_pipeline/test_runner.py`

**Step 1: Implement PipelineRunner**

```python
# src/denoisr/pipeline/runner.py
import logging
from datetime import datetime, timezone
from pathlib import Path

from denoisr.pipeline.config import PipelineConfig
from denoisr.pipeline.state import PipelineState
from denoisr.pipeline.steps import (
    step_fetch_data,
    step_sort_pgn,
    step_init_model,
    step_generate_tier_data,
    step_train_phase1_tier,
    step_train_phase2,
    step_train_phase3,
)

log = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, cfg: PipelineConfig, restart: bool = False) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.output.dir)
        self.state_path = self.output_dir / "pipeline_state.json"
        if restart or not self.state_path.exists():
            self.state = PipelineState(started_at=datetime.now(timezone.utc).isoformat())
        else:
            self.state = PipelineState.load(self.state_path)
            log.info("Resumed from state: phase=%s, tier=%d", self.state.phase, self.state.elo_tier_index)

    def run(self) -> None:
        self._save_state()

        # 1. Fetch data
        if self.state.phase in ("init", ""):
            log.info("=== Step 1: Fetch data ===")
            self.state = step_fetch_data(self.cfg, self.state)
            self._save_state()

        # 2. Sort by Elo
        if self.state.phase in ("fetched", "init", ""):
            log.info("=== Step 2: Sort PGN by Elo ===")
            self.state = step_sort_pgn(self.cfg, self.state)
            self._save_state()

        # 3. Init model
        if self.state.phase in ("sorted", "fetched", "init", ""):
            log.info("=== Step 3: Initialize model ===")
            self.state = step_init_model(self.cfg, self.state)
            self._save_state()

        # 4. Elo curriculum (Phase 1)
        tiers = self.cfg.elo_curriculum.tiers
        start_tier = self.state.elo_tier_index
        for i, min_elo in enumerate(tiers):
            if i < start_tier:
                continue
            log.info("=== Phase 1, Tier %d: Elo >= %d ===", i, min_elo)
            self.state = step_generate_tier_data(self.cfg, self.state, min_elo, i)
            self._save_state()
            self.state = step_train_phase1_tier(self.cfg, self.state, i, min_elo)
            self.state.phase = "elo_curriculum"
            self._save_state()

        # 5. Phase 2
        if self.state.phase != "phase2_complete" and self.state.phase != "complete":
            log.info("=== Phase 2: Diffusion bootstrapping ===")
            self.state = step_train_phase2(self.cfg, self.state)
            self._save_state()

        # 6. Phase 3
        if self.state.phase != "complete":
            log.info("=== Phase 3: RL self-play ===")
            self.state = step_train_phase3(self.cfg, self.state)
            self._save_state()

        log.info("Pipeline complete! Final checkpoint: %s", self.state.last_checkpoint)

    def _save_state(self) -> None:
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state.save(self.state_path)
```

**Step 2: Implement CLI entry point**

```python
# src/denoisr/scripts/train.py
"""Unified training pipeline: PGN → Phase 3 in one command."""
import argparse
import logging
from pathlib import Path

from denoisr.pipeline.config import load_config
from denoisr.pipeline.runner import PipelineRunner

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Denoisr training pipeline from a TOML config"
    )
    parser.add_argument(
        "--config", type=str, default="pipeline.toml",
        help="Path to pipeline TOML config (default: pipeline.toml)",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Ignore saved state and start fresh",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(Path(args.config))
    runner = PipelineRunner(cfg, restart=args.restart)
    runner.run()

if __name__ == "__main__":
    main()
```

**Step 3: Register entry point**

In `pyproject.toml`, add to `[project.scripts]`:
```toml
denoisr-train = "denoisr.scripts.train:main"
```

**Step 4: Run tests and commit**

```bash
uv run pytest tests/test_pipeline/ -v
git add src/denoisr/pipeline/runner.py src/denoisr/scripts/train.py pyproject.toml tests/test_pipeline/
git commit -m "feat: add denoisr-train unified pipeline command with TOML config"
```

---

### Task 8: Update README.md

**Files:**
- Modify: `README.md`

Update:
- Data source recommendation: standard Lichess instead of Elite
- Add `denoisr-train` as the primary command
- Document `pipeline.toml` format
- Update the commands table
- Simplify the "Full training pipeline" section to show the single-command path first, with per-phase commands as "advanced usage"

**Step 1: Update and commit**

```bash
git add README.md
git commit -m "docs: update README for unified pipeline and standard Lichess data"
```

---

### Task 9: Integration test

**Files:**
- Create: `tests/test_pipeline/test_integration.py`

Write a mini integration test that runs the pipeline with a tiny config (1 Elo tier, 10 examples, 1 epoch) against a fixture PGN. This verifies the full flow works end-to-end without requiring Stockfish or large data.

```python
# tests/test_pipeline/test_integration.py
import pytest
from pathlib import Path
from denoisr.pipeline.config import PipelineConfig, DataConfig, EloCurriculumConfig, Phase1Config, OutputConfig
from denoisr.pipeline.runner import PipelineRunner

@pytest.mark.skipif(not shutil.which("stockfish"), reason="Stockfish not installed")
def test_mini_pipeline(tmp_path, fixture_pgn):
    """End-to-end pipeline with minimal config."""
    cfg = PipelineConfig(
        data=DataConfig(pgn_path=str(fixture_pgn), examples_per_tier=10),
        elo_curriculum=EloCurriculumConfig(tiers=[0], max_epochs_per_tier=1),
        phase1=Phase1Config(batch_size=4),
        output=OutputConfig(dir=str(tmp_path)),
    )
    runner = PipelineRunner(cfg)
    # Run only through init + data gen (skip training for speed)
    # Full integration with training requires Stockfish + GPU
```

**Step 1: Write and commit**

```bash
uv run pytest tests/test_pipeline/test_integration.py -v
git add tests/test_pipeline/test_integration.py
git commit -m "test: add mini pipeline integration test"
```

---

## Task Summary

| Task | Description | Est. Size |
|------|-------------|-----------|
| 1 | Remove SimpleBoardEncoder (13 files) | Large |
| 2 | Remove in-memory generation path | Medium |
| 3 | Consolidate streaming functions | Small |
| 4 | PipelineConfig TOML parser | Medium |
| 5 | PipelineState JSON persistence | Small |
| 6 | Pipeline step functions | Large |
| 7 | Runner + CLI entry point | Medium |
| 8 | Update README.md | Medium |
| 9 | Integration test | Small |

**Total: 9 tasks, ~33 files touched/created.**

Commit after every task. Run `uv run pytest tests/ -x -q` after every change.
