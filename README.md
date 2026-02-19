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

Training proceeds in three phases, each gated on measurable quality thresholds to prevent premature advancement.

## Prerequisites

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management (never use pip directly)
- [Stockfish](https://stockfishchess.org/) for supervised targets in Phase 1
- [cutechess-cli](https://github.com/cutechess/cutechess) for Elo benchmarking (optional)
- Apple Silicon (MPS) for development, CUDA for scale training

```bash
# Clone and sync
git clone <repo-url> && cd denoisr
uv sync

# Verify all 197 tests pass
uv run pytest tests/ -v

# Skip Stockfish-dependent tests if not installed
uv run pytest tests/ -v -k "not stockfish"
```

## Architecture overview

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

The diffusion module adds an imagination step before the backbone:

```
LatentState  ──>  DiffusionModule (T denoising steps)  ──>  enriched LatentState
                      (iteratively refines noise into plausible future trajectories,
                       conditioned on the current position)
```

The world model enables latent-space MCTS:

```
(LatentState, Action)  ──>  WorldModel  ──>  (next LatentState, predicted reward)
                                (causal transformer, UniZero-style)
```

Total: ~340M parameters across all modules.

## Phase 1: Supervised learning (Lichess + Stockfish)

### What happens

The network learns basic chess knowledge from human games annotated with Stockfish evaluations. This is the cheapest way to bootstrap: millions of Lichess games provide positional patterns, and Stockfish provides policy targets (move distributions) and value targets (win/draw/loss probabilities) that are far stronger than human labels alone.

### Why Stockfish targets instead of human move labels

Human games provide one-hot move labels (the move that was played), but Stockfish analysis provides full probability distributions over all legal moves. This is dramatically more informative: a one-hot label says "e4 was played" while Stockfish says "e4 is best at 45%, d4 is close at 40%, c4 is reasonable at 10%..." The richer signal accelerates learning and teaches the network to distinguish between good alternatives rather than memorizing a single move.

### Why separate learning rates per component

The encoder learns general board representations that should change slowly (stability matters), while the policy and value heads need to quickly adapt their output mappings. Using a single learning rate forces a compromise that's too slow for the heads or too aggressive for the encoder. Tiered rates (encoder at 0.1x, backbone at 0.3x, heads at 1.0x base LR) let each component train at its natural pace.

### Data pipeline

```
lichess_games.pgn.zst
    |
    v
SimplePGNStreamer  ──>  stream of GameRecord
    |
    v
For each position in each game:
    |
    ├──> SimpleBoardEncoder.encode(board)  ──>  BoardTensor [12, 8, 8]
    └──> StockfishOracle.evaluate(board)   ──>  (PolicyTarget [64,64], ValueTarget, centipawns)
    |
    v
TrainingExample batches  ──>  SupervisedTrainer
```

### How to run

```bash
uv run denoisr-train-phase1 \
    --pgn data/lichess_2026-01.pgn.zst \
    --stockfish /usr/local/bin/stockfish \
    --stockfish-depth 12 \
    --max-examples 100000 \
    --batch-size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --output outputs/phase1.pt
```

Key flags:

| Flag                | Default             | Description                                                |
| ------------------- | ------------------- | ---------------------------------------------------------- |
| `--pgn`             | (required)          | Path to `.pgn` or `.pgn.zst` file                          |
| `--stockfish`       | (required)          | Path to Stockfish binary                                   |
| `--stockfish-depth` | `10`                | Stockfish analysis depth (higher = better targets)         |
| `--max-examples`    | `100000`            | Maximum training examples to generate                      |
| `--holdout-frac`    | `0.05`              | Fraction reserved for accuracy evaluation                  |
| `--batch-size`      | `64`                | Training batch size                                        |
| `--epochs`          | `100`               | Maximum training epochs                                    |
| `--lr`              | `1e-4`              | Base learning rate (heads 1x, backbone 0.3x, encoder 0.1x) |
| `--output`          | `outputs/phase1.pt` | Checkpoint output path                                     |
| `--log-every`       | `10`                | Log every N batches                                        |
| `--d-s`             | `256`               | Latent dimension                                           |
| `--num-heads`       | `8`                 | Attention heads                                            |
| `--num-layers`      | `15`                | Transformer backbone layers                                |
| `--ffn-dim`         | `1024`              | Feedforward dimension                                      |

The loss function uses HarmonyDream for dynamic coefficient balancing across policy, value, and auxiliary objectives. Gradient clipping (max norm 1.0) prevents training instability.

### Gate to Phase 2

Training automatically checks whether policy accuracy exceeds **30% top-1** on the held-out set after each epoch. When the gate passes, training stops and the checkpoint is ready for Phase 2.

## Phase 2: World model + diffusion bootstrapping

### What happens

Two new modules are trained on top of the Phase 1 representations:

1. **World model** learns latent-space dynamics: given a position and a move, predict the next latent state and reward. This enables MCTS to search entirely in latent space without needing game rules.
2. **Diffusion module** learns to denoise corrupted future trajectories. Given the current position, it learns to reconstruct plausible future continuations from noise.

### Why train the world model on Lichess trajectories first

The world model needs consistent latent dynamics before MCTS can search effectively. Training on supervised Lichess trajectories (where positions evolve according to real games) gives the model a stable foundation. If trained directly from self-play with a weak policy, the trajectories would be near-random, and the world model would learn chaotic dynamics that MCTS cannot exploit.

### Why diffusion in latent space instead of FEN tokens

Operating on latent tensors [64, d_s] rather than FEN strings means the diffusion process works with rich continuous representations where nearby points in latent space correspond to similar board positions. This makes the corruption-and-recovery training signal much smoother than discrete token prediction. It also means denoising steps are cheap (small tensor operations) rather than expensive (autoregressive token generation).

### Why the consistency projector matters

Without the consistency loss, the world model's latent dynamics can "collapse" — the predicted next state might satisfy the value equivalence constraint by mapping everything to a small subspace, losing information needed for planning. The SimSiam-style consistency projector, with stop-gradient on the target branch, prevents this by ensuring predicted latent states remain structurally similar to encoded real states.

### The 6-term loss function

Phase 2 activates all 6 loss terms, balanced by HarmonyDream:

| Term        | What it measures                                                             | Why it matters                |
| ----------- | ---------------------------------------------------------------------------- | ----------------------------- |
| Policy      | Cross-entropy between predicted and target move distributions                | Core chess skill              |
| Value       | Cross-entropy between predicted and target WDL                               | Position evaluation           |
| Consistency | SimSiam negative cosine similarity between predicted and encoded next states | Prevents latent collapse      |
| Diffusion   | MSE between predicted and actual noise (DDPM)                                | Future trajectory imagination |
| Reward      | MSE between predicted and actual game reward                                 | Outcome-relevant dynamics     |
| Ply         | Huber loss on predicted vs actual game length                                | Time horizon awareness        |

### Diffusion step curriculum

The diffusion trainer starts with only 25% of the maximum timesteps and gradually increases by 2% per epoch. This curriculum makes early training easier (less noise to denoise) and progressively challenges the model with harder corruption levels. Without it, the model faces the hardest denoising tasks from step one, slowing convergence.

### How to run

```bash
uv run denoisr-train-phase2 \
    --checkpoint outputs/phase1.pt \
    --pgn data/lichess_2026-01.pgn.zst \
    --seq-len 5 \
    --max-trajectories 50000 \
    --batch-size 32 \
    --epochs 200 \
    --lr 1e-4 \
    --output outputs/phase2.pt
```

Key flags:

| Flag                 | Default             | Description                             |
| -------------------- | ------------------- | --------------------------------------- |
| `--checkpoint`       | (required)          | Phase 1 checkpoint to load              |
| `--pgn`              | (required)          | PGN file for trajectory extraction      |
| `--seq-len`          | `5`                 | Consecutive board states per trajectory |
| `--max-trajectories` | `50000`             | Maximum trajectories to extract         |
| `--batch-size`       | `32`                | Training batch size                     |
| `--epochs`           | `200`               | Training epochs                         |
| `--lr`               | `1e-4`              | Learning rate for diffusion parameters  |
| `--output`           | `outputs/phase2.pt` | Checkpoint output path                  |
| `--log-every`        | `10`                | Log every N batches                     |

### Gate to Phase 3

Diffusion-conditioned inference accuracy must exceed single-step accuracy by **>5 percentage points**. This confirms the diffusion module provides meaningful information beyond what the backbone already captures. The script reports the best diffusion loss; evaluate both engines on held-out positions to measure the gate metric.

## Phase 3: RL self-play

### What happens

The engine improves beyond human/Stockfish supervision by playing games against itself. This phase has two sub-phases:

- **Phase 3a (MCTS bootstrap):** Traditional MCTS in latent space generates self-play games. MCTS provides decent move quality even with a weak policy network, so the training data has meaningful winning/losing patterns from the start.
- **Phase 3b (Diffusion transition):** Once the diffusion model has absorbed enough patterns from MCTS-quality games, it gradually takes over move selection via alpha mixing. A virtuous cycle begins: better diffusion produces better games which produce better training data which produces better diffusion.

### Why MCTS bootstraps diffusion (not the other way around)

Early self-play games from a random policy are terrible — random moves produce random trajectories with no useful signal for the diffusion model. MCTS solves this cold-start problem: even with a weak neural network, tree search explores enough alternatives to find reasonable moves. The diffusion model then learns from these MCTS-quality games, absorbing tactical patterns that MCTS found through brute-force search.

### Why alpha mixing instead of a hard switch

A sudden switch from MCTS to diffusion would destabilize training: the game quality would drop sharply, producing worse training data, which would further degrade the diffusion model. Linear alpha mixing (from 0 to 1 over 50 generations) ensures a smooth transition where the diffusion model is only responsible for a fraction of decisions commensurate with its current ability.

### Temperature scheduling

Self-play uses temperature scheduling within each game: high temperature (exploratory, stochastic moves) for the first 30 moves to diversify openings, then temperature=0 (greedy) for the remainder to generate clean tactical data. Across generations, the base temperature decays by 3% per generation as the model trusts its improving policy more and needs less exploration.

### MuZero Reanalyse for sample efficiency

Rather than discarding old games after training on them once, the ReanalyseActor replays old positions through the current (improved) network's MCTS. This generates higher-quality policy targets from existing data without playing new games — one of MuZero's key sample efficiency innovations. A game played 100 generations ago might have had poor MCTS targets with the old network, but the current network's MCTS produces much better targets for the same positions.

### How to run

```bash
uv run denoisr-train-phase3 \
    --checkpoint outputs/phase2.pt \
    --generations 1000 \
    --games-per-gen 100 \
    --reanalyse-per-gen 50 \
    --mcts-sims 800 \
    --buffer-capacity 100000 \
    --alpha-generations 50 \
    --save-every 10 \
    --output outputs/phase3.pt
```

Key flags:

| Flag                  | Default             | Description                                             |
| --------------------- | ------------------- | ------------------------------------------------------- |
| `--checkpoint`        | (required)          | Phase 2 checkpoint to load                              |
| `--generations`       | `1000`              | Total self-play generations                             |
| `--games-per-gen`     | `100`               | Games played per generation                             |
| `--reanalyse-per-gen` | `50`                | Old games reanalysed per generation                     |
| `--mcts-sims`         | `800`               | MCTS simulations per move                               |
| `--buffer-capacity`   | `100000`            | Priority replay buffer capacity                         |
| `--alpha-generations` | `50`                | Generations to transition from MCTS to diffusion (0->1) |
| `--save-every`        | `10`                | Save checkpoint every N generations                     |
| `--output`            | `outputs/phase3.pt` | Checkpoint output path                                  |

### Success criteria

Elo increases over generations (measured via cutechess-cli). The ultimate goal: diffusion-only inference (alpha=1.0) matches or exceeds MCTS-based inference strength.

## Inference: playing with the trained model

The `denoisr-play` command starts a UCI-compatible chess engine that connects to any chess GUI (CuteChess, Arena, Banksia, etc.).

### Single-pass mode (fastest)

Direct encoder -> backbone -> policy head. No search, no diffusion. The simplest mode.

```bash
uv run denoisr-play \
    --checkpoint outputs/phase3.pt \
    --mode single
```

### Diffusion-enhanced mode (stronger, adjustable)

Adds diffusion imagination before the policy backbone. The `--denoising-steps` flag provides **anytime search**: more steps produce stronger play at the cost of inference time. This is the core innovation — the engine thinks deeper by running more denoising iterations rather than by building an explicit search tree.

```bash
uv run denoisr-play \
    --checkpoint outputs/phase3.pt \
    --mode diffusion \
    --denoising-steps 20
```

| Flag                | Default    | Description                                                       |
| ------------------- | ---------- | ----------------------------------------------------------------- |
| `--checkpoint`      | (required) | Path to any phase checkpoint                                      |
| `--mode`            | `single`   | `single` (fast) or `diffusion` (stronger)                         |
| `--denoising-steps` | `20`       | Denoising iterations for diffusion mode (more = stronger, slower) |

### Connecting to a chess GUI

Point your chess GUI at the engine command:

```
uv run denoisr-play --checkpoint outputs/phase3.pt --mode diffusion
```

The engine speaks standard UCI protocol (reads from stdin, writes to stdout). In CuteChess, add it via **Engines > Add** and paste the command above.

## Benchmarking with cutechess-cli

Measure Elo against a reference engine using SPRT (Sequential Probability Ratio Test) for statistical confidence:

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

## Complete training pipeline

```bash
# 1. Install dependencies
uv sync

# 2. Phase 1: Supervised learning (~hours with GPU)
uv run denoisr-train-phase1 \
    --pgn data/lichess_2026-01.pgn.zst \
    --stockfish /usr/local/bin/stockfish \
    --output outputs/phase1.pt

# 3. Phase 2: Diffusion bootstrapping (~hours with GPU)
uv run denoisr-train-phase2 \
    --checkpoint outputs/phase1.pt \
    --pgn data/lichess_2026-01.pgn.zst \
    --output outputs/phase2.pt

# 4. Phase 3: RL self-play (~days with GPU)
uv run denoisr-train-phase3 \
    --checkpoint outputs/phase2.pt \
    --output outputs/phase3.pt

# 5. Play!
uv run denoisr-play \
    --checkpoint outputs/phase3.pt \
    --mode diffusion

# 6. Benchmark
uv run denoisr-benchmark \
    --engine-cmd "uv run denoisr-play --checkpoint outputs/phase3.pt --mode diffusion" \
    --opponent-cmd stockfish \
    --games 100
```

## All available commands

| Command                       | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| `uv run denoisr-train-phase1` | Phase 1: Supervised learning with Stockfish targets |
| `uv run denoisr-train-phase2` | Phase 2: Diffusion bootstrapping on trajectories    |
| `uv run denoisr-train-phase3` | Phase 3: RL self-play with MCTS-to-diffusion mixing |
| `uv run denoisr-play`         | UCI chess engine (single-pass or diffusion)         |
| `uv run denoisr-benchmark`    | Elo benchmarking via cutechess-cli                  |

All commands support `--help` for full flag documentation:

```bash
uv run denoisr-train-phase1 --help
uv run denoisr-play --help
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

## Running tests

```bash
uv run pytest tests/ -v                  # all tests
uv run pytest tests/ -x                  # stop at first failure
uv run pytest tests/ -n auto             # parallel execution
uv run pytest tests/ -k "not stockfish"  # skip Stockfish-dependent tests
uv run pytest tests/test_nn/ -v          # just neural network tests
```
