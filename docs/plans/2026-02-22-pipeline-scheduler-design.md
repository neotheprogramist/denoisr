# Pipeline Scheduler & Simplification Design

## Problem

Training a Denoisr model requires manually running 6+ CLI commands in sequence: download PGN, sort by Elo, generate data, initialize model, train Phase 1, train Phase 2, train Phase 3. Each step has its own flags, and the user must manually check phase gates and advance. The codebase also carries two encoder paths (12 vs 122 planes) and two data source strategies (elite vs standard) — unnecessary complexity now that the audit has settled on the advanced configuration.

## Solution

A single `denoisr-train` command driven by a TOML config file that orchestrates the entire pipeline end-to-end, with Elo curriculum learning and automatic phase progression.

## Pipeline Flow

```
1. FETCH DATA
   Download standard Lichess rated games (.pgn.zst) from database.lichess.org

2. SORT BY ELO
   Sort PGN into 5 cumulative tier files:
   800+, 1200+, 1600+, 2000+, 2400+

3. INIT MODEL
   Create random 122-plane checkpoint

4. ELO CURRICULUM (Phase 1)
   For each tier [800, 1200, 1600, 2000, 2400]:
     a. Generate training data from games ≥ tier Elo
        (Stockfish evaluation, tactical enrichment, random sampling)
     b. Train Phase 1 (supervised) until gate OR max epochs
     c. If top-1 accuracy ≥ gate → advance to next tier
     d. If max epochs hit → warn and advance anyway
   Output: phase1.pt (checkpoint from final tier)

5. PHASE 2 — Diffusion bootstrapping
   Extract trajectories from 2400+ tier PGN
   Train all 6 loss terms until Phase 2 gate
   Output: phase2.pt

6. PHASE 3 — RL self-play
   Self-play with MCTS→diffusion transition
   Output: phase3.pt
```

### Elo curriculum rationale

Standard Lichess games (database.lichess.org) include all skill levels, from beginners to titled players. Training on everything at once is suboptimal — the model wastes capacity learning beginner blunders. Progressive training raises the floor:

- **Tier 800+**: Learn basic piece movement, captures, simple tactics
- **Tier 1200+**: Learn opening principles, basic endgames
- **Tier 1600+**: Learn positional play, tactical combinations
- **Tier 2000+**: Learn deeper strategy, pawn structures
- **Tier 2400+**: Learn master-level subtleties

Each tier generates fresh training data via Stockfish. The model checkpoint carries forward — fine-tuned on progressively harder games. Cumulative tiers (≥ threshold) ensure higher tiers still contain lower-rated games for diversity, but the average game quality rises.

## TOML Config Format

```toml
# pipeline.toml — Denoisr training pipeline configuration
# All fields are optional. Shown values are defaults.

[data]
# Standard Lichess rated games (all Elos, .pgn.zst format)
pgn_url = "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
pgn_path = "data/lichess.pgn.zst"
sorted_dir = "data/sorted/"
stockfish_path = ""                        # auto-detect from PATH
stockfish_depth = 10
examples_per_tier = 200_000
tactical_fraction = 0.25
workers = 0                                # 0 = auto (cpu_count*2+1)

[elo_curriculum]
tiers = [800, 1200, 1600, 2000, 2400]
gate_accuracy = 0.50                       # top-1 to advance
max_epochs_per_tier = 100                  # safety cap

[model]
d_s = 256
num_heads = 8
num_layers = 15
ffn_dim = 1024
num_timesteps = 100

[phase1]
lr = 3e-4
batch_size = 1024
warmup_epochs = 5
weight_decay = 1e-4

[phase2]
epochs = 200
lr = 3e-4
batch_size = 128
seq_len = 10
max_trajectories = 50_000

[phase3]
generations = 1000
games_per_gen = 100
mcts_sims = 800

[output]
dir = "outputs/"
run_name = ""                              # auto-timestamp if empty
```

### CLI interface

```bash
# Full pipeline from scratch
denoisr-train --config pipeline.toml

# Resume interrupted training
denoisr-train --config pipeline.toml

# Start fresh, ignoring saved state
denoisr-train --config pipeline.toml --restart

# Override specific values
denoisr-train --config pipeline.toml --phase1.lr 1e-3

# Run only specific steps
denoisr-train --config pipeline.toml --only fetch,sort,phase1
```

### Config precedence

1. TOML config values
2. CLI flag overrides
3. Defaults (hardcoded in PipelineConfig dataclass)

## State Management & Resumption

A `pipeline_state.json` file tracks progress:

```json
{
  "phase": "elo_curriculum",
  "elo_tier_index": 2,
  "elo_tier_min": 1600,
  "last_checkpoint": "outputs/phase1_tier2.pt",
  "last_data": "outputs/tier_1600_data.pt",
  "tier_accuracies": {
    "800": 0.52,
    "1200": 0.51
  },
  "started_at": "2026-02-22T14:30:00",
  "updated_at": "2026-02-22T16:45:00"
}
```

Resume logic:
1. On start, check for `pipeline_state.json` in output dir
2. If found, load and skip completed steps
3. If not found, start from scratch
4. `--restart` ignores saved state

### Artifact naming

```
outputs/
├── pipeline_state.json
├── random_model.pt
├── tier_800_data.pt
├── phase1_tier0.pt
├── tier_1200_data.pt
├── phase1_tier1.pt
├── tier_1600_data.pt
├── phase1_tier2.pt
├── tier_2000_data.pt
├── phase1_tier3.pt
├── tier_2400_data.pt
├── phase1_tier4.pt           # final Phase 1
├── phase2.pt
├── phase3.pt
└── logs/<run-name>/
```

## Simplifications

### Remove SimpleBoardEncoder

Delete `SimpleBoardEncoder` and the `--num-planes` flag entirely. Hardcode `ExtendedBoardEncoder` (122 planes) as the only encoder. `ModelConfig.num_planes` becomes a constant `122`.

**Files affected**: `board_encoder.py` (delete), `config.py` (remove flag), all references to `SimpleBoardEncoder`.

### Remove in-memory data generation path

Delete `generate_examples()` and `stack_examples()` — only keep `generate_to_file()` (disk-backed memmap). For 200K+ examples per tier, in-memory is never appropriate.

### Consolidate streaming functions

Merge `_stream_positions()` and `_stream_positions_elo_stratified()` into one function with optional parameters. The Elo floor parameter handles both cases.

### Default to standard Lichess games

README updated to reference `database.lichess.org` standard rated games instead of the Elite database. The Elo curriculum handles quality filtering — no need for a pre-curated dataset.

## Architecture

### New files

- `src/denoisr/scripts/train.py` — Main `denoisr-train` entry point
- `src/denoisr/pipeline/config.py` — `PipelineConfig` TOML parser
- `src/denoisr/pipeline/state.py` — `PipelineState` persistence
- `src/denoisr/pipeline/runner.py` — `PipelineRunner` orchestration
- `src/denoisr/pipeline/steps.py` — Individual step functions (fetch, sort, generate, train)

### Internal step functions

Each step is a plain function:

```python
def step_fetch_data(cfg: PipelineConfig, state: PipelineState) -> PipelineState:
    ...

def step_sort_pgn(cfg: PipelineConfig, state: PipelineState) -> PipelineState:
    ...

def step_generate_data(cfg: PipelineConfig, state: PipelineState, min_elo: int) -> PipelineState:
    ...

def step_train_phase1(cfg: PipelineConfig, state: PipelineState) -> PipelineState:
    ...
```

Each returns an updated `PipelineState` that gets persisted to disk. The runner calls steps in sequence, skipping completed ones based on saved state.

### Existing scripts unchanged

`denoisr-train-phase1`, `denoisr-train-phase2`, etc. remain for power users. The new `denoisr-train` calls the same underlying training functions, not the CLI scripts.

## Testing

- Unit tests for TOML parsing with missing/partial sections
- Unit tests for state persistence (save, load, resume)
- Unit tests for step skip logic (completed steps skipped)
- Integration test: mini pipeline with tiny config (1 tier, 10 examples, 1 epoch)
