# Denoisr

A transformer-diffusion chess engine that learns to play chess by imagining futures rather than calculating move scores.

## What this is

Traditional chess engines evaluate positions with explicit scores ("e4 is worth +0.3 pawns"). Denoisr takes a fundamentally different approach: it learns to **dream up plausible future game continuations** using diffusion, then picks the move that leads to the best imagined futures. A single transformer forward pass is limited to constant-depth reasoning (TC^0), which cannot express minimax search. Diffusion's T denoising steps provide effective depth O(L\*T), breaking this barrier and enabling adjustable-depth reasoning at inference time.

The architecture combines ideas from several recent results:

| Result                                 | Source                  | What it provides                                      |
| -------------------------------------- | ----------------------- | ----------------------------------------------------- |
| BT4 +270 Elo over CNN                  | Lc0 project             | Transformer backbone (attention subsumes convolution) |
| AlphaVile +180 Elo from input features | Czech et al., ECAI 2024 | Extended board encoder with 110 feature planes        |
| DiffuSearch +540 Elo over searchless   | Ye et al., ICLR 2025    | Diffusion-based iterative refinement for chess        |
| HarmonyDream 10-69% improvement        | Ma et al., ICML 2024    | Dynamic loss balancing across 6 training objectives   |
| EfficientZero consistency loss         | Yu et al.               | Prevents latent-space collapse in world model         |

## Quick start: play against an untrained engine

You can play against Denoisr immediately — no training data, no Stockfish, no GPU required. The untrained model plays random-looking moves, which makes a great baseline to compare against after training.

### 1. Install

```bash
git clone <repo-url> && cd denoisr
uv sync
```

### 2. Initialize a random model

```bash
uv run denoisr-init --output outputs/random_model.pt
```

This creates a checkpoint with random weights (~340M parameters). The engine will make legal moves, but they'll be essentially random.

### 3. Install a chess GUI

Denoisr speaks the UCI protocol, so it works with any UCI-compatible chess GUI. We recommend **CuteChess**:

**macOS (Homebrew):**

```bash
brew install --cask cutechess
```

**Linux (Flatpak):**

```bash
flatpak install flathub io.github.cutechess.cutechess
```

**Linux (build from source):**

```bash
sudo apt install qt6-base-dev cmake g++
git clone https://github.com/cutechess/cutechess.git
cd cutechess && cmake . && make -j$(nproc)
sudo make install
```

**Windows:**
Download the latest release from [github.com/cutechess/cutechess/releases](https://github.com/cutechess/cutechess/releases).

Other UCI-compatible GUIs also work: Arena, Banksia, Lucas Chess, PyChess, or any GUI that supports adding custom UCI engines.

### 4. Connect the engine to the GUI

In CuteChess:

1. Go to **Tools > Settings > Engines**
2. Click **+** (Add)
3. Set **Name** to `Denoisr (random)`
4. Set **Command** to the full path of the engine launcher, for example:
   ```
   /path/to/denoisr/.venv/bin/denoisr-play
   ```
5. Set **Arguments** to:
   ```
   --checkpoint outputs/random_model.pt --mode single
   ```
6. Set **Protocol** to **UCI**
7. Set **Working Directory** to your denoisr project root
8. Click **OK**

Now start a new game (**Game > New**) and select `Denoisr (random)` as one of the players. Play a few games to see how the untrained engine behaves — it makes legal moves but has no chess understanding.

> **Tip:** If you prefer the terminal, you can also interact with the engine directly via UCI:
>
> ```bash
> uv run denoisr-play --checkpoint outputs/random_model.pt --mode single
> ```
>
> Then type `uci`, `isready`, `position startpos`, `go movetime 1000`, etc.

## Full training pipeline

After seeing how the random model plays, train it through all three phases to produce a strong chess engine.

### Prerequisites

- [Stockfish](https://stockfishchess.org/) for Phase 1 supervised targets (install via `brew install stockfish`, `sudo apt install stockfish`, or `sudo snap install stockfish`)
- A GPU is recommended (MPS on Apple Silicon, CUDA on Linux) but CPU works for small runs
- ~2 GB disk space for training data

### Step 1: Download training data

Phase 1 needs a PGN file of chess games. The [Lichess Elite Database](https://database.nikonoel.fr/) provides curated high-quality games (2400+ vs 2200+ rated players, no bullet):

```bash
mkdir -p data

# Download a month of elite games (~60-120 MB .zip, extracts to .pgn)
wget -P data/ https://database.nikonoel.fr/lichess_elite_2025-01.zip
unzip data/lichess_elite_2025-01.zip -d data/
```

For larger-scale training, the [full Lichess database](https://database.lichess.org/) provides all rated games in `.pgn.zst` format (natively supported by the streamer, no decompression needed):

```bash
# Full month of rated games (~20-50 GB compressed, streams directly)
wget -P data/ https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst
```

### Step 2: Initialize the model

Create a random model checkpoint that Phase 1 will train from:

```bash
uv run denoisr-init --output outputs/random_model.pt
```

This is the same random model from the quick start — if you already created it, skip this step.

### Step 3: Generate training examples

Data generation with Stockfish is slow (~22 pos/s), so it runs as a separate step. Generate once, then iterate on training without re-generating:

```bash
uv run denoisr-generate-data \
    --pgn data/lichess_elite_2025-01.pgn \
    --max-examples 100000 \
    --output outputs/training_data.pt
```

Stockfish is auto-detected from PATH. Pass `--stockfish /path/to/stockfish` to override.

**What you'll see:**

```
Generating examples: 45%|████████▌          | 45000/100000 [12:30<15:20, 59.7pos/s]
Generated 100000 training examples
Saved 100000 examples to outputs/training_data.pt
```

| Flag                | Default                    | Description                                |
| ------------------- | -------------------------- | ------------------------------------------ |
| `--pgn`             | (required)                 | Path to `.pgn` or `.pgn.zst` file          |
| `--stockfish`       | auto-detect PATH           | Path to Stockfish binary                   |
| `--stockfish-depth` | `10`                       | Stockfish analysis depth (higher = better) |
| `--max-examples`    | `100000`                   | Training examples to generate              |
| `--output`          | `outputs/training_data.pt` | Output path for generated data             |

### Step 4: Phase 1 — Supervised learning

The network learns basic chess from the pre-generated training data:

```bash
uv run denoisr-train-phase1 \
    --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt \
    --batch-size 64 \
    --epochs 100 \
    --output outputs/phase1.pt
```

**What you'll see:**

```
Loaded checkpoint: d_s=256, heads=8, layers=15
Loaded 100000 training examples from outputs/training_data.pt
Epoch 12/100:  68%|█████████████▌      | 1088/1600 [00:45<00:21] loss=2.1234 policy=1.8901 value=0.2333
Epoch 12/100: avg_loss=2.0891 top1_accuracy=28.5%
```

Training automatically stops when top-1 accuracy exceeds **30%** (Phase 1 gate).

| Flag             | Default             | Description                                                    |
| ---------------- | ------------------- | -------------------------------------------------------------- |
| `--checkpoint`   | (required)          | Checkpoint to load (create with `denoisr-init`)                |
| `--data`         | (required)          | Training data `.pt` file (create with `denoisr-generate-data`) |
| `--holdout-frac` | `0.05`              | Fraction for accuracy evaluation                               |
| `--batch-size`   | `64`                | Batch size                                                     |
| `--epochs`       | `100`               | Maximum epochs                                                 |
| `--lr`           | `1e-4`              | Learning rate                                                  |
| `--output`       | `outputs/phase1.pt` | Checkpoint path                                                |

### Step 5: Phase 2 — Diffusion bootstrapping

Trains the diffusion module to denoise future trajectories, with the Phase 1 encoder frozen:

```bash
uv run denoisr-train-phase2 \
    --checkpoint outputs/phase1.pt \
    --pgn data/lichess_elite_2025-01.pgn \
    --max-trajectories 50000 \
    --epochs 200 \
    --output outputs/phase2.pt
```

**What you'll see:**

```
Extracting trajectories: 72%|██████████████▍     | 36000/50000 [02:15<00:52, 267traj/s]
Epoch 45/200:  55%|███████████         | 860/1562 [00:32<00:26] loss=0.0234
Epoch 45/200: avg_diffusion_loss=0.0218 curriculum_steps=32
```

Gate to Phase 3: diffusion-conditioned accuracy must exceed single-step by >5 percentage points.

| Flag                 | Default             | Description                        |
| -------------------- | ------------------- | ---------------------------------- |
| `--checkpoint`       | (required)          | Phase 1 checkpoint                 |
| `--pgn`              | (required)          | PGN file for trajectory extraction |
| `--seq-len`          | `5`                 | Board states per trajectory        |
| `--max-trajectories` | `50000`             | Trajectories to extract            |
| `--batch-size`       | `32`                | Batch size                         |
| `--epochs`           | `200`               | Training epochs                    |
| `--lr`               | `1e-4`              | Learning rate                      |
| `--output`           | `outputs/phase2.pt` | Checkpoint path                    |

### Step 6: Phase 3 — RL self-play

The engine improves beyond human/Stockfish supervision by playing against itself:

```bash
uv run denoisr-train-phase3 \
    --checkpoint outputs/phase2.pt \
    --generations 1000 \
    --games-per-gen 100 \
    --save-every 10 \
    --output outputs/phase3.pt
```

**What you'll see:**

```
Generations:  5%|█                   | 50/1000 [4:12:30<79:30:00]
Gen 51 self-play:  34%|██████▊             | 34/100 [08:12<15:55] W=12 D=8 L=14
Gen 51/1000: buffer=5100 alpha=0.00 temp=0.220 W/D/L=48/21/31 reanalysed=450
```

| Flag                  | Default             | Description                              |
| --------------------- | ------------------- | ---------------------------------------- |
| `--checkpoint`        | (required)          | Phase 2 checkpoint                       |
| `--generations`       | `1000`              | Self-play generations                    |
| `--games-per-gen`     | `100`               | Games per generation                     |
| `--reanalyse-per-gen` | `50`                | Old games reanalysed per generation      |
| `--mcts-sims`         | `800`               | MCTS simulations per move                |
| `--buffer-capacity`   | `100000`            | Replay buffer capacity                   |
| `--alpha-generations` | `50`                | Generations to transition MCTS→diffusion |
| `--save-every`        | `10`                | Checkpoint every N generations           |
| `--output`            | `outputs/phase3.pt` | Checkpoint path                          |

## Play with the trained model

After training, add the trained engine to your chess GUI and compare it against the random model:

### Single-pass mode (fastest)

Direct encoder → backbone → policy head. No search, no diffusion:

```bash
uv run denoisr-play \
    --checkpoint outputs/phase3.pt \
    --mode single
```

### Diffusion-enhanced mode (stronger)

Adds diffusion imagination before the policy backbone. More denoising steps = stronger play:

```bash
uv run denoisr-play \
    --checkpoint outputs/phase3.pt \
    --mode diffusion \
    --denoising-steps 20
```

### Compare random vs trained

Add both engines to CuteChess to see the difference training makes:

| Engine in CuteChess   | Command args                                         |
| --------------------- | ---------------------------------------------------- |
| `Denoisr (random)`    | `--checkpoint outputs/random_model.pt --mode single` |
| `Denoisr (trained)`   | `--checkpoint outputs/phase3.pt --mode single`       |
| `Denoisr (diffusion)` | `--checkpoint outputs/phase3.pt --mode diffusion`    |

Set up a match between them (**Game > New**, select both engines) to see the Elo improvement from training.

| Flag                | Default    | Description                                    |
| ------------------- | ---------- | ---------------------------------------------- |
| `--checkpoint`      | (required) | Path to any phase checkpoint                   |
| `--mode`            | `single`   | `single` (fast) or `diffusion` (stronger)      |
| `--denoising-steps` | `20`       | Denoising iterations (more = stronger, slower) |

## Benchmarking with cutechess-cli

Measure Elo against a reference engine using SPRT for statistical confidence:

```bash
# Basic benchmark (100 games)
uv run denoisr-benchmark \
    --engine-cmd "uv run denoisr-play --checkpoint outputs/phase3.pt --mode diffusion" \
    --opponent-cmd stockfish \
    --games 100 \
    --time-control "10+0.1"

# With SPRT (stops early when statistically significant)
uv run denoisr-benchmark \
    --engine-cmd "uv run denoisr-play --checkpoint outputs/phase3.pt --mode diffusion" \
    --opponent-cmd stockfish \
    --games 1000 \
    --sprt-elo0 0 \
    --sprt-elo1 50 \
    --concurrency 4

# Dry run (print the cutechess-cli command without executing)
uv run denoisr-benchmark \
    --engine-cmd "./run_denoisr.sh" \
    --opponent-cmd stockfish \
    --dry-run
```

| Flag             | Default     | Description                                |
| ---------------- | ----------- | ------------------------------------------ |
| `--engine-cmd`   | (required)  | Command to run the Denoisr UCI engine      |
| `--opponent-cmd` | `stockfish` | Command to run the opponent UCI engine     |
| `--games`        | `100`       | Number of games to play                    |
| `--time-control` | `10+0.1`    | Time control (seconds + increment)         |
| `--sprt-elo0`    | (none)      | SPRT null hypothesis Elo difference        |
| `--sprt-elo1`    | (none)      | SPRT alternative hypothesis Elo difference |
| `--concurrency`  | `1`         | Parallel games                             |
| `--dry-run`      | `false`     | Print command without running              |

## Architecture deep dive

### Board encoding → latent space → policy/value

```
Board position (chess.Board)
    |
    v
BoardEncoder  ──>  BoardTensor [C, 8, 8]
    |                   (12 planes simple, 110 planes extended)
    v
ChessEncoder  ──>  LatentState [64, d_s]
    |                   (one token per square)
    v
PolicyBackbone  ──>  LatentState [64, d_s]
    |                   (15-layer transformer with smolgen + Shaw relative PE)
    |
    ├──> PolicyHead  ──>  move_logits [64, 64]
    |                       (source-destination grid)
    └──> ValueHead   ──>  (wdl_probs [3], ply [1])
                            (win/draw/loss + game length)
```

### Diffusion imagination

The diffusion module adds an imagination step before the backbone:

```
LatentState  ──>  DiffusionModule (T denoising steps)  ──>  enriched LatentState
                      (iteratively refines noise into plausible future trajectories,
                       conditioned on the current position)
```

### World model for latent MCTS

```
(LatentState, Action)  ──>  WorldModel  ──>  (next LatentState, predicted reward)
                                (causal transformer, UniZero-style)
```

Total: ~340M parameters across all modules.

### Training phases explained

Training proceeds in three phases, each gated on measurable quality thresholds to prevent premature advancement.

**Phase 1: Supervised learning** — The cheapest way to bootstrap. Millions of Lichess games provide positional patterns, and Stockfish provides policy targets (move distributions) and value targets (win/draw/loss probabilities) far stronger than human labels alone. Stockfish gives full probability distributions over legal moves rather than one-hot human labels — "e4 is best at 45%, d4 is close at 40%" is dramatically more informative than "e4 was played."

**Phase 2: World model + diffusion** — Two new modules train on top of Phase 1 representations. The world model learns latent-space dynamics (given position + move, predict next state). The diffusion module learns to denoise corrupted future trajectories. The 6-term HarmonyDream loss balances policy, value, consistency, diffusion, reward, and ply objectives. A curriculum gradually increases diffusion timesteps from 25% to 100%.

**Phase 3: RL self-play** — MCTS in latent space generates self-play data (Phase 3a), then alpha mixing gradually transitions from MCTS to diffusion guidance (Phase 3b). MuZero Reanalyse replays old positions through the improved network for sample efficiency. Temperature scheduling diversifies openings (high temp for first 30 moves, then greedy).

## All available commands

| Command                        | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| `uv run denoisr-init`          | Initialize a random (untrained) model checkpoint    |
| `uv run denoisr-generate-data` | Generate training data from PGN + Stockfish         |
| `uv run denoisr-train-phase1`  | Phase 1: Supervised learning from generated data    |
| `uv run denoisr-train-phase2`  | Phase 2: Diffusion bootstrapping on trajectories    |
| `uv run denoisr-train-phase3`  | Phase 3: RL self-play with MCTS-to-diffusion mixing |
| `uv run denoisr-play`          | UCI chess engine (single-pass or diffusion)         |
| `uv run denoisr-benchmark`     | Elo benchmarking via cutechess-cli                  |

All commands support `--help` for full flag documentation:

```bash
uv run denoisr-init --help
uv run denoisr-generate-data --help
uv run denoisr-train-phase1 --help
uv run denoisr-play --help
```

## Running tests

```bash
uv run pytest tests/ -v                  # all tests
uv run pytest tests/ -x                  # stop at first failure
uv run pytest tests/ -n auto             # parallel execution
uv run pytest tests/ -k "not stockfish"  # skip Stockfish-dependent tests
uv run pytest tests/test_nn/ -v          # just neural network tests
```

## Project structure

```
denoisr/
├── src/denoisr/
│   ├── types/          # Domain types: BoardTensor, Action, LatentState, TrainingExample
│   ├── game/           # ChessGame wrapping python-chess
│   ├── data/           # Encoders, PGN streaming, Stockfish oracle, dataset
│   ├── nn/             # Neural network modules (encoder, backbone, heads, world model, diffusion)
│   ├── training/       # Loss, trainers, MCTS, self-play, replay buffer, reanalyse, orchestrator
│   ├── inference/      # Chess engines (single-pass, diffusion-enhanced), UCI protocol
│   ├── evaluation/     # cutechess-cli benchmarking harness
│   └── scripts/        # CLI entry points for all phases + inference + benchmarking
├── tests/              # 197 tests mirroring src/ structure
├── fixtures/           # Sample PGN files for testing
├── docs/plans/         # Architecture design and implementation plans
└── outputs/            # Training artifacts (gitignored)
```
