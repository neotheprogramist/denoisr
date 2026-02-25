# Denoisr

A transformer-diffusion chess engine that learns to play chess by imagining futures rather than calculating move scores.

## What this is

Traditional chess engines evaluate positions with explicit scores ("e4 is worth +0.3 pawns"). Denoisr takes a fundamentally different approach: it learns to **dream up plausible future game continuations** using diffusion, then picks the move that leads to the best imagined futures. A single transformer forward pass is limited to constant-depth reasoning (TC^0), which cannot express minimax search. Diffusion's T denoising steps provide effective depth O(L\*T), breaking this barrier and enabling adjustable-depth reasoning at inference time.

The architecture combines ideas from several recent results:

| Result                                 | Source                  | What it provides                                                                                                        |
| -------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| BT4 +270 Elo over CNN                  | Lc0 project             | Transformer backbone (attention subsumes convolution)                                                                   |
| AlphaVile +180 Elo from input features | Czech et al., ECAI 2024 | Extended board encoder with 122 feature planes (including attack/defense maps, hanging pieces, pins, mobility, threats) |
| DiffuSearch +540 Elo over searchless   | Ye et al., ICLR 2025    | Diffusion-based iterative refinement for chess                                                                          |
| HarmonyDream 10-69% improvement        | Ma et al., ICML 2024    | Dynamic loss balancing across 6 training objectives                                                                     |
| EfficientZero consistency loss         | Yu et al.               | Prevents latent-space collapse in world model                                                                           |

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

### 3. Play against the engine

Denoisr includes a built-in chess GUI — no external software needed:

```bash
uv run denoisr-gui --checkpoint outputs/random_model.pt
```

This opens a window where you can play against the engine with click-to-move interaction.

> **Tip:** Denoisr also speaks the UCI protocol, so it works with any UCI-compatible GUI (CuteChess, Arena, Lucas Chess, etc.) if you prefer:
>
> ```bash
> uv run denoisr-play --checkpoint outputs/random_model.pt --mode single
> ```
>
> Then type `uci`, `isready`, `position startpos`, `go movetime 1000`, etc.

### 4. Play a game

In the GUI:

1. The checkpoint is pre-filled from the command line (or use **Browse** to select one)
2. Choose **single** or **diffusion** mode
3. Choose your color (white or black)
4. Click **New Game**
5. Click a piece, then click its destination to make moves

The engine responds automatically after each of your moves.

## Full training pipeline

After seeing how the random model plays, train it through all three phases to produce a strong chess engine.

### Prerequisites

- [Stockfish](https://stockfishchess.org/) for Phase 1 supervised targets (install via `brew install stockfish`, `sudo apt install stockfish`, or `sudo snap install stockfish`)
- A GPU is recommended (MPS on Apple Silicon, CUDA on Linux) but CPU works for small runs
- ~2 GB disk space for training data

### Recommended: `denoisr-train` (unified pipeline)

The simplest way to train is the unified pipeline command. It handles data download, model initialization, data generation, and all three training phases from env-based configuration.

```bash
cp .env.example .env
uv run denoisr-train
```

The CLI auto-loads `.env`. Configuration is env-only and fail-fast: startup aborts immediately if any required parameter is missing or malformed.

Hardware-tuned values in `.env.example` target:

- RTX 3060 12GB: high-occupancy training batches (`phase1=512`, `phase2=128`, `phase3=128`)
- Threadripper 3970X (64 threads): full CPU worker utilization (`DENOISR_WORKERS=64`)

The pipeline streams positions from raw PGN, generates Stockfish targets, trains Phase 1, then advances through Phase 2 and Phase 3.

**Pipeline state resumption:** Progress is saved to `pipeline_state.json` in the output directory. If interrupted, re-running the same command resumes from the last completed step. Use `--restart` to start fresh. Use `--only` to run specific steps:

```bash
# Resume from where you left off
uv run denoisr-train

# Start fresh, ignoring saved state
uv run denoisr-train --restart

# Run only specific steps
uv run denoisr-train --only fetch,init
```

| Flag        | Default     | Description                                                     |
| ----------- | ----------- | --------------------------------------------------------------- |
| `--restart` | off         | Ignore saved state and start fresh                              |
| `--only`    | (all steps) | Comma-separated steps to run: `fetch,init,phase1,phase2,phase3` |

### Advanced: per-phase commands

For fine-grained control, you can run each phase individually. This is useful for debugging, hyperparameter sweeps, or custom data preparation.
Examples below assume you copied `.env.example` to `.env` so required args are already provided via environment.

#### Step 1: Download training data

The [Lichess standard database](https://database.lichess.org/) provides all rated games in `.pgn.zst` format (natively supported by the streamer, no decompression needed):

```bash
mkdir -p data

# Download a month of rated games (~20-50 GB compressed, streams directly)
wget -P data/ https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst
```

#### Step 2: Initialize the model

Create a random model checkpoint that Phase 1 will train from:

```bash
uv run denoisr-init --output outputs/random_model.pt
```

This is the same random model from the quick start -- if you already created it, skip this step.

#### Step 3: Generate training examples

Data generation writes chunked shard files plus a `chunked_v1` manifest. Each worker runs its own Stockfish instance, and fixed-size chunk buffers bound peak RAM while supporting very large datasets:

```bash
uv run denoisr-generate-data \
    --pgn data/lichess_db_standard_rated_2025-01.pgn.zst \
    --max-examples 4000000 \
    --output outputs/training_data.pt
```

Stockfish is auto-detected from PATH. Pass `--stockfish /path/to/stockfish` to override.

Generation stops after writing `--max-examples` streamed positions from the PGN.
Defaults below reflect the recommended `.env.example` profile.

**What you'll see:**

```
Chunked generation: max_examples=4000000 workers=64 chunk_examples=1000000 (~44.36 GiB chunk buffers)
Evaluating positions: 45%|████████▌          | 1800000/4000000 [58:12<70:00, 523pos/s]
Wrote chunk 0 (1000000 examples): outputs/training_data_chunks/chunk_000000.pt
Wrote chunk 1 (1000000 examples): outputs/training_data_chunks/chunk_000001.pt
Saved chunked manifest with 4000000 examples across 4 chunks to outputs/training_data.pt
Done: 4000000 examples generated.
```

| Flag                   | Default                    | Description                                    |
| ---------------------- | -------------------------- | ---------------------------------------------- |
| `--pgn`                | (required)                 | Path to `.pgn` or `.pgn.zst` file              |
| `--stockfish`          | auto-detect PATH           | Path to Stockfish binary                       |
| `--stockfish-depth`    | `10`                       | Stockfish analysis depth (higher = better)     |
| `--max-examples`       | `4000000`                  | Training examples to generate                  |
| `--workers`            | `64`                       | Worker processes (each runs its own Stockfish) |
| `--policy-temperature` | `80`                       | Softmax temperature for policy targets         |
| `--label-smoothing`    | `0.02`                     | Label smoothing epsilon for policy targets     |
| `--seed`               | (none)                     | Random seed for reproducible sampling          |
| `--chunksize`          | `1024`                     | `imap_unordered` chunksize for worker batching |
| `--chunk-examples`     | `1000000`                  | Examples per output shard                      |
| `--output`             | `outputs/training_data.pt` | Output path for generated data                 |

#### Step 4: Phase 1 -- Supervised learning

The network learns basic chess from the pre-generated training data:

```bash
uv run denoisr-train-phase1 \
    --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt \
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

Training automatically stops when top-1 accuracy exceeds **50%** (Phase 1 gate).

| Flag             | Default             | Description                                                            |
| ---------------- | ------------------- | ---------------------------------------------------------------------- |
| `--checkpoint`   | (required)          | Checkpoint to load (create with `denoisr-init`)                        |
| `--data`         | (required)          | Training data `.pt` file (create with `denoisr-generate-data`)         |
| `--holdout-frac` | `0.05`              | Fraction for accuracy evaluation                                       |
| `--batch-size`   | `512`               | Batch size                                                             |
| `--epochs`       | `100`               | Maximum epochs                                                         |
| `--lr`           | `3e-4`              | Learning rate                                                          |
| `--output`       | `outputs/phase1.pt` | Checkpoint path                                                        |
| `--run-name`     | auto timestamp      | Run label added to training logs (see [Training logs](#training-logs)) |

Plus all [training optimization](#training-optimization-trainingconfig) flags.

#### Step 5: Phase 2 -- Diffusion bootstrapping

Trains all 6 loss terms (policy, value, diffusion, world model state/reward, consistency) with the Phase 1 encoder frozen and backbone at reduced learning rate:

```bash
uv run denoisr-train-phase2 \
    --checkpoint outputs/phase1.pt \
    --pgn data/lichess_db_standard_rated_2025-01.pgn.zst \
    --max-trajectories 30000 \
    --epochs 100 \
    --output outputs/phase2.pt
```

**What you'll see:**

```
Extracting trajectories: 72%|██████████████▍     | 21600/30000 [02:15<00:52, 267traj/s]
Epoch 45/100:  55%|███████████         | 860/1562 [00:32<00:26] loss=0.0234
Epoch 45/100: total_loss=0.0218 curriculum_steps=32
```

Gate to Phase 3: diffusion-conditioned accuracy must exceed single-step by >5 percentage points.

| Flag                 | Default             | Description                                                 |
| -------------------- | ------------------- | ----------------------------------------------------------- |
| `--checkpoint`       | (required)          | Phase 1 checkpoint                                          |
| `--pgn`              | (required)          | PGN file for trajectory extraction                          |
| `--seq-len`          | `10`                | Board states per trajectory (9 future states for diffusion) |
| `--max-trajectories` | `30000`             | Trajectories to extract                                     |
| `--batch-size`       | `128`               | Batch size                                                  |
| `--epochs`           | `100`               | Training epochs                                             |
| `--lr`               | `3e-4`              | Learning rate                                               |
| `--output`           | `outputs/phase2.pt` | Checkpoint path                                             |
| `--run-name`         | auto timestamp      | Run label added to training logs                            |

Plus all [training optimization](#training-optimization-trainingconfig) and [diffusion curriculum](#diffusion-curriculum) flags.

#### Step 6: Phase 3 -- RL self-play

The engine improves beyond human/Stockfish supervision by playing against itself:

```bash
uv run denoisr-train-phase3 \
    --checkpoint outputs/phase2.pt \
    --generations 400 \
    --games-per-gen 64 \
    --save-every 10 \
    --output outputs/phase3.pt
```

**What you'll see:**

```
Generations:  5%|█                   | 20/400 [1:41:30<31:59:00]
Gen 21 self-play:  34%|██████▊             | 22/64 [05:01<09:35] W=8 D=5 L=9
Gen 21/400: buffer=1344 alpha=0.00 temp=0.310 W/D/L=31/14/19 reanalysed=672
```

| Flag                  | Default             | Description                               |
| --------------------- | ------------------- | ----------------------------------------- |
| `--checkpoint`        | (required)          | Phase 2 checkpoint                        |
| `--generations`       | `400`               | Self-play generations                     |
| `--games-per-gen`     | `64`                | Games per generation                      |
| `--reanalyse-per-gen` | `32`                | Old games reanalysed per generation       |
| `--mcts-sims`         | `400`               | MCTS simulations per move                 |
| `--buffer-capacity`   | `50000`             | Replay buffer capacity                    |
| `--alpha-generations` | `40`                | Generations to transition MCTS->diffusion |
| `--save-every`        | `10`                | Checkpoint every N generations            |
| `--output`            | `outputs/phase3.pt` | Checkpoint path                           |

Plus all [training optimization](#training-optimization-trainingconfig) and [Phase 3 self-play](#phase-3-self-play-and-mcts-phase-3-only) flags.

### Training logs

Phase 1 and Phase 2 write human-readable metrics to a single file: `logs/denoisr.log`.

`--run-name` is still useful: it tags log lines with a run label so you can filter mixed logs from multiple experiments. Without `--run-name`, a timestamp like `2026-02-20_14-30-15` is generated automatically.

#### What gets logged

- Human-readable epoch summaries (single compact line per epoch)
- Human-readable hyperparameter summary at startup (`HPARAMS ...`)
- Human-readable grokking summaries and warnings (`GROK-EPOCH ...`, `GROKKING ...`)
- Regular script logs from all modules, all in the same file, with timestamps

#### Read logs directly

```bash
# Follow live training output
tail -f logs/denoisr.log

# Show only epoch summary lines
rg "denoisr.metrics" logs/denoisr.log
```

#### Agent-friendly mode (default)

By default, training scripts suppress tqdm progress bars and emit one structured log line per epoch via Python's `logging` module. This is designed for automated agents that monitor training and react to metrics:

```
epoch=1/100  loss=6.5663  policy_loss=5.8901  value_loss=0.6762  top1=0.0%  top5=0.0%  lr=1.00e-04  grad_norm_avg=0.342  grad_norm_peak=1.000  samples/s=496  epoch_time=3.8s  data_pct=12%  cpu=45%/98%  ram=2341mb
```

To enable tqdm progress bars for interactive use:

```bash
uv run denoisr-train-phase1 --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt --tqdm
```

#### Log directory layout

```
logs/
└── denoisr.log
```

### Grokking detection (Phase 1)

Neural networks sometimes exhibit **grokking** — a phenomenon where the model memorizes training data first, then suddenly generalizes to held-out data much later. Denoisr includes grokking detection that monitors weight dynamics, representation structure, and structured holdout performance to detect and accelerate this transition.

Grok tracking is enabled by default. You can still set it explicitly:

```bash
uv run denoisr-train-phase1 \
    --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt \
    --grok-tracking \
    --run-name grok-experiment
```

When enabled, training automatically:

1. **Splits holdout data** into 4 structured sets — random, game-level (entire games held out), opening-family (entire ECO letter groups), and piece-count (endgame positions) — to detect generalization across different axes
2. **Computes Tier 1 metrics every step** — weight norms per module, gradient norms, train/holdout loss gap
3. **Computes Tier 2 metrics periodically** — effective rank via SVD (every `--grok-erank-freq` steps), spectral norms and HTSR alpha power-law exponents (every `--grok-spectral-freq` steps)
4. **Runs a 4-state machine** (BASELINE → ONSET_DETECTED → TRANSITIONING → GROKKED) that increases evaluation frequency 5-10x when grokking signals appear
5. **Emits console alerts** via `logging.WARNING` when state transitions occur

#### Grokking metrics (structured events)

| Metric                              | Frequency          | What it measures                                                              |
| ----------------------------------- | ------------------ | ----------------------------------------------------------------------------- |
| `grok/weight_norm_total`            | Every step         | Total L2 norm across all parameters (drops during circuit formation)          |
| `grok/weight_norm/{module}`         | Every step         | Per-module norms (encoder, backbone, policy_head, value_head)                 |
| `grok/erank/layer_{i}`              | Every N steps      | Effective rank of layer activations (measures representation richness)        |
| `grok/spectral_norm/layer_{i}/attn` | Every N steps      | Largest singular value of attention weights (stability indicator)             |
| `grok/spectral_norm/layer_{i}/ffn`  | Every N steps      | Largest singular value of FFN weights                                         |
| `grok/alpha/layer_{i}`              | Every N steps      | HTSR power-law exponent (lower = better generalization)                       |
| `grok/holdout/{split}/accuracy`     | Every epoch        | Top-1 accuracy per holdout split                                              |
| `grok/holdout/{split}/loss`         | Every epoch        | Loss per holdout split                                                        |
| `grok/loss_gap`                     | Every epoch        | Train loss minus best holdout loss (memorization indicator)                   |
| `grok/state`                        | Every step + epoch | Current state machine value (0=baseline, 1=onset, 2=transitioning, 3=grokked) |

#### Grokfast acceleration

**Grokfast** is enabled by default and can accelerate grokking by amplifying slow-varying gradient components (the generalizing signal) while leaving fast-varying components (memorization) alone:

```bash
uv run denoisr-train-phase1 \
    --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt \
    --grok-tracking \
    --grokfast \
    --grokfast-alpha 0.98 \
    --grokfast-lamb 2.0 \
    --run-name grokfast-experiment
```

The filter is applied between gradient unscaling and gradient clipping in the training loop, so it works correctly with mixed precision training.

Disable either feature with `--no-grok-tracking` and `--no-grokfast`.

#### Grokking detection flags

| Flag                     | Default | What it controls                                                    |
| ------------------------ | ------- | ------------------------------------------------------------------- |
| `--grok-tracking`        | on      | Enable grokking detection metrics, structured holdouts, and alerts  |
| `--grok-erank-freq`      | `1000`  | Effective rank computation frequency (steps). Lower = more data     |
| `--grok-spectral-freq`   | `5000`  | Spectral norm / HTSR alpha frequency (steps)                        |
| `--grok-onset-threshold` | `0.95`  | Weight norm ratio for onset detection (lower = more sensitive)      |
| `--grokfast`             | on      | Enable Grokfast EMA gradient filtering (~50x grokking acceleration) |
| `--grokfast-alpha`       | `0.98`  | EMA decay rate. Higher = smoother (more historical averaging)       |
| `--grokfast-lamb`        | `2.0`   | Amplification factor. Higher = stronger boost to slow gradients     |

### Hyperparameters

All hyperparameters are configurable via CLI flags. They are centralized in `src/denoisr/scripts/config.py` as two frozen dataclasses:

- **`ModelConfig`** — architecture parameters saved in checkpoints (needed at inference)
- **`TrainingConfig`** — optimization parameters used only during training

Pass `--help` to any training command to see all available flags with defaults.

#### Model architecture (`ModelConfig`)

These control the neural network structure. Changing them creates a new architecture — you cannot load a checkpoint trained with different values.

| Flag                       | Default | What it controls                                                                                                                |
| -------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--d-s`                    | `256`   | Latent dimension per square token. All transformer layers use this width. Larger = more capacity but quadratic attention memory |
| `--num-heads`              | `8`     | Attention heads in the policy backbone. Must divide `d_s`. More heads = finer-grained attention patterns                        |
| `--num-layers`             | `15`    | Policy backbone transformer depth. More layers = deeper positional reasoning. Matches Lc0 BT4                                   |
| `--ffn-dim`                | `1024`  | Feed-forward hidden dim inside transformer blocks. Typically 4× `d_s`                                                           |
| `--num-timesteps`          | `100`   | DDPM diffusion timesteps. More = finer noise schedule = better generation quality, but slower                                   |
| `--world-model-layers`     | `12`    | World model transformer depth for latent dynamics prediction                                                                    |
| `--diffusion-layers`       | `6`     | DiT denoiser depth. Fewer layers since it operates in latent space                                                              |
| `--proj-dim`               | `256`   | Consistency projector dimension for SimSiam collapse prevention                                                                 |
| `--gradient-checkpointing` | off     | Trade compute for VRAM by recomputing activations in backward pass (~30% slower)                                                |

#### Training optimization (`TrainingConfig`)

These control how the model learns. Safe to change between runs without architectural incompatibility.

| Flag                      | Default | What it controls                                                                                                                                        |
| ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--lr`                    | `3e-4`  | Base learning rate for task heads (policy, value). The single most impactful hyperparameter                                                             |
| `--max-grad-norm`         | `5.0`   | Gradient clipping L2 threshold. Prevents instability from large gradient spikes. Check epoch summary lines and overflow warnings if instability appears |
| `--weight-decay`          | `1e-4`  | AdamW L2 regularization. Increase to 1e-2 for small datasets, decrease to 0 if underfitting                                                             |
| `--encoder-lr-multiplier` | `1.0`   | LR multiplier for encoder/backbone vs heads. Lower values preserve pretrained representations. 1.0 = encoder trains at same LR as heads                 |
| `--min-lr`                | `1e-6`  | Minimum LR at end of cosine annealing. Should be 10-100× smaller than `--lr`                                                                            |
| `--warmup-epochs`         | `10`    | Linear warmup epochs (LR ramps from 0 → target). Prevents destructive early updates                                                                     |
| `--workers`               | `0`     | DataLoader worker processes (`0` = auto)                                                                                                                |
| `--warm-restarts`         | on      | Use cosine annealing with warm restarts (T_0=20, T_mult=2) instead of plain cosine decay                                                                |
| `--threat-weight`         | `0.1`   | Weight for threat prediction auxiliary loss (forces intermediate representations to encode attack information)                                          |
| `--tqdm`                  | off     | Show tqdm progress bars. Off by default for agent-friendly log output                                                                                   |

#### Loss weights

These control the relative importance of each training objective. Higher weight = model prioritizes that loss more.

| Flag                       | Default | What it controls                                                                                   |
| -------------------------- | ------- | -------------------------------------------------------------------------------------------------- |
| `--policy-weight`          | `2.0`   | Policy (move prediction) cross-entropy. Set higher because correct moves matter most               |
| `--value-weight`           | `0.5`   | Value (win/draw/loss) cross-entropy. Lower than policy — evaluation is secondary in early training |
| `--consistency-weight`     | `1.0`   | SimSiam consistency loss. Prevents latent-space collapse in the world model                        |
| `--diffusion-weight`       | `1.0`   | Diffusion denoising MSE. Trains imagination of future trajectories                                 |
| `--reward-weight`          | `1.0`   | Reward prediction MSE. Teaches outcome prediction from latent states                               |
| `--ply-weight`             | `0.1`   | Game length prediction Huber loss. Auxiliary signal, low weight                                    |
| `--illegal-penalty-weight` | `0.01`  | L2 penalty on illegal-move logits. Encourages model to suppress illegal positions                  |
| `--harmony-dream`          | on      | Enable HarmonyDream dynamic loss balancing (auto-adjusts weights inversely to loss magnitudes)     |
| `--harmony-ema-decay`      | `0.99`  | EMA decay for HarmonyDream tracking. Higher = smoother adaptation                                  |

#### Diffusion curriculum

The diffusion module trains with a curriculum: start with easy (few-step) denoising, gradually increase to full difficulty.

| Flag                            | Default | What it controls                                                                          |
| ------------------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `--curriculum-initial-fraction` | `0.25`  | Fraction of timesteps used at training start. 0.25 = begin with T/4 steps                 |
| `--curriculum-growth`           | `1.02`  | Per-epoch step count multiplier. 1.02 = +2%/epoch, reaching full difficulty in ~70 epochs |

#### Phase gates

Training advances through phases only when measurable quality thresholds are met.

| Flag            | Default | What it controls                                                                     |
| --------------- | ------- | ------------------------------------------------------------------------------------ |
| `--phase1-gate` | `0.50`  | Top-1 accuracy to pass Phase 1 → 2. 50% ensures strong move-ranking before Phase 2   |
| `--phase2-gate` | `5.0`   | Percentage-point accuracy improvement from diffusion vs single-step, for Phase 2 → 3 |

#### Phase 3: Self-play and MCTS (Phase 3 only)

| Flag                             | Default | What it controls                                                                          |
| -------------------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `--c-puct`                       | `1.4`   | MCTS exploration constant (UCB). Higher = more exploration. 1.4 ≈ √2 from UCB1 theory     |
| `--dirichlet-alpha`              | `0.3`   | Root noise concentration. Smaller = sharper noise. 0.3 is chess-standard (vs 0.03 for Go) |
| `--dirichlet-epsilon`            | `0.25`  | Fraction of root prior replaced by noise. 0.25 = 75% policy, 25% exploration              |
| `--temperature-base`             | `1.0`   | Move selection temperature. Higher = more random. Decays across generations               |
| `--temperature-explore-moves`    | `30`    | Moves per game at full temperature, then greedy. Covers opening diversity                 |
| `--temperature-generation-decay` | `0.97`  | Per-generation temperature decay. 0.97 = ~50% temp after 23 generations                   |
| `--max-moves`                    | `300`   | Maximum moves per self-play game before draw                                              |
| `--reanalyse-simulations`        | `100`   | MCTS sims for MuZero Reanalyse (fewer than main MCTS for broader coverage)                |

#### Example: tuning for faster convergence

```bash
# Aggressive learning with higher LR, stronger policy focus, slower encoder
uv run denoisr-train-phase1 \
    --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt \
    --lr 1e-3 \
    --policy-weight 3.0 \
    --value-weight 0.3 \
    --encoder-lr-multiplier 0.1 \
    --run-name aggressive-lr1e-3
```

#### Example: training on limited VRAM

```bash
# Reduce memory usage with fewer workers
uv run denoisr-train-phase1 \
    --checkpoint outputs/random_model.pt \
    --data outputs/training_data.pt \
    --batch-size 16 \
    --workers 0 \
    --run-name low-vram
```

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

Use **Match** mode in the built-in GUI to pit different checkpoints against each other:

```bash
# Play against the trained model
uv run denoisr-gui --checkpoint outputs/phase3.pt --mode single

# Or diffusion-enhanced mode (stronger)
uv run denoisr-gui --checkpoint outputs/phase3.pt --mode diffusion
```

| Checkpoint        | Mode        | Expected strength          |
| ----------------- | ----------- | -------------------------- |
| `random_model.pt` | `single`    | Random legal moves         |
| `phase3.pt`       | `single`    | Fast, moderate strength    |
| `phase3.pt`       | `diffusion` | Stronger, uses imagination |

| Flag                | Default    | Description                                    |
| ------------------- | ---------- | ---------------------------------------------- |
| `--checkpoint`      | (required) | Path to any phase checkpoint                   |
| `--mode`            | `single`   | `single` (fast) or `diffusion` (stronger)      |
| `--denoising-steps` | `20`       | Denoising iterations (more = stronger, slower) |

## Benchmarking

### GUI match mode (no external tools needed)

Switch to **Match** mode in the GUI to run engine-vs-engine matches with live Elo/SPRT tracking:

```bash
uv run denoisr-gui --checkpoint outputs/phase3.pt --mode diffusion
```

### CLI benchmark (no external tools needed)

Estimate Elo against Stockfish with parallel game execution — no cutechess-cli required:

```bash
# Basic benchmark (100 games against Elo-limited Stockfish)
uv run denoisr-benchmark \
    --engine-cmd "uv run denoisr-play --checkpoint outputs/phase3.pt --mode diffusion" \
    --opponent-elo 1200 \
    --games 100

# With SPRT (stops early when statistically significant)
uv run denoisr-benchmark \
    --engine-cmd "uv run denoisr-play --checkpoint outputs/phase3.pt --mode diffusion" \
    --opponent-elo 1500 \
    --games 1000 \
    --sprt-elo0 0 \
    --sprt-elo1 50
```

Games run in parallel across `64` workers (each owning a persistent engine + Stockfish subprocess pair). A bundled opening book ensures game variety. Stockfish is auto-detected from PATH.

### Trained model vs random baseline with Elo estimation

Even when both models score 0/100 against Stockfish, ACPL (Average Centipawn Loss) analysis reveals meaningful quality differences. ACPL measures how far each move deviates from Stockfish's best — a model losing with ACPL 50 is much stronger than one losing with ACPL 200.

```bash
# Compare trained model vs random baseline, with ACPL analysis and PGN recording
uv run denoisr-benchmark \
    --engine-cmd "uv run denoisr-play --checkpoint outputs/phase1.pt" \
    --baseline-cmd "uv run denoisr-play --checkpoint outputs/random_model.pt" \
    --opponent-skill 0 \
    --games 20 --concurrency 4 \
    --pgn-out outputs/pgn/ \
    --analyze --analysis-depth 10
```

**What you'll see:**

```
12:34:56 INFO Benchmark: 20 games, 4 workers, TC 10+0.1 Skill 0 + baseline
12:34:56 INFO ============================================================
12:34:56 INFO   Engine: uv run denoisr-play --checkpoint outputs/phase1.pt
12:34:56 INFO ============================================================
12:35:12 INFO [engine] Game 20/20: +0 =0 -20 | Elo: N/A
12:35:12 INFO ============================================================
12:35:12 INFO   Baseline: uv run denoisr-play --checkpoint outputs/random_model.pt
12:35:12 INFO ============================================================
12:35:28 INFO [baseline] Game 20/20: +0 =0 -20 | Elo: N/A
12:35:28 INFO PGN saved: outputs/pgn/engine/ (20 games) + outputs/pgn/engine_all.pgn
12:35:28 INFO PGN saved: outputs/pgn/baseline/ (20 games) + outputs/pgn/baseline_all.pgn
12:35:30 INFO Analyzing Engine (20 games, depth 10)...
12:35:45 INFO Engine ACPL: 85.3
12:35:45 INFO Engine Est. Elo: 485
12:35:45 INFO Engine Blunders: 42
12:35:45 INFO Analyzing Baseline (20 games, depth 10)...
12:36:00 INFO Baseline ACPL: 198.7
12:36:00 INFO Baseline Est. Elo: 400
12:36:00 INFO Baseline Blunders: 156
12:36:00 INFO --- Engine ---
12:36:00 INFO Score: +0 =0 -20 (20 games)
12:36:00 INFO Elo:   N/A
12:36:00 INFO --- Baseline ---
12:36:00 INFO Score: +0 =0 -20 (20 games)
12:36:00 INFO Elo:   N/A
12:36:00 INFO
============================================================
  Comparison
============================================================
                       Engine           Baseline
  Score              +0 =0 -20        +0 =0 -20
  Elo vs opponent           N/A              N/A
  LOS                     0.0%             0.0%
  Score %                  0.0%             0.0%
  ACPL                    85.3            198.7
  Est. Elo                 485              400
  Blunders                  42              156

  Engine and baseline score identically.
```

Both engines lose every game, but the trained model plays with ACPL ~85 (estimated ~485 Elo) while the random model plays at ACPL ~199 (estimated ~400 Elo). This confirms training is working even before the model can beat Stockfish.

### PGN recording

Save every game as PGN for external analysis (e.g. in Lichess, ChessBase, or SCID):

```bash
uv run denoisr-benchmark \
    --engine-cmd "uv run denoisr-play --checkpoint outputs/phase1.pt" \
    --opponent-skill 0 --games 10 \
    --pgn-out outputs/pgn/
```

Output structure:

```
outputs/pgn/
├── engine/
│   ├── game_0000.pgn
│   ├── game_0001.pgn
│   └── ...
└── engine_all.pgn          # All games in one file
```

### Benchmark flags

| Flag                     | Default                 | Description                                |
| ------------------------ | ----------------------- | ------------------------------------------ |
| `--engine-cmd`           | (required)              | Command to run the Denoisr UCI engine      |
| `--baseline-cmd`         | (none)                  | Baseline engine for comparison             |
| `--head-to-head`         | off                     | Play engine vs baseline directly           |
| `--opponent-cmd`         | auto-detect `stockfish` | Opponent engine command                    |
| `--opponent-elo`         | full strength           | Limit opponent via UCI_Elo (e.g. 1200)     |
| `--opponent-skill`       | (none)                  | Stockfish Skill Level 0-20 (0 = weakest)   |
| `--games`                | `100`                   | Number of games to play                    |
| `--time-control`         | `10+0.1`                | Base+increment seconds                     |
| `--openings`             | bundled opening book    | Path to EPD opening book                   |
| `--concurrency`          | `64`                    | Parallel game workers                      |
| `--sprt-elo0`            | (none)                  | SPRT null hypothesis Elo difference        |
| `--sprt-elo1`            | (none)                  | SPRT alternative hypothesis Elo difference |
| `--pgn-out`              | (none)                  | Directory to save PGN files                |
| `--analyze`              | off                     | Run Stockfish ACPL analysis after games    |
| `--analysis-depth`       | `12`                    | Stockfish analysis depth                   |
| `--analysis-concurrency` | same as `--concurrency` | Parallel Stockfish analysis workers        |

## Architecture deep dive

### Board encoding → latent space → policy/value

```
Board position (chess.Board)
    |
    v
BoardEncoder  ──>  BoardTensor [122, 8, 8]
    |                   (AlphaVile-style extended features)
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

| Command                        | Description                                            |
| ------------------------------ | ------------------------------------------------------ |
| `uv run denoisr-train`         | **Full pipeline from env config** (recommended)        |
| `uv run denoisr-init`          | Initialize a random (untrained) model checkpoint       |
| `uv run denoisr-generate-data` | Generate training data from PGN + Stockfish            |
| `uv run denoisr-train-phase1`  | Phase 1: Supervised learning from generated data       |
| `uv run denoisr-train-phase2`  | Phase 2: Diffusion bootstrapping on trajectories       |
| `uv run denoisr-train-phase3`  | Phase 3: RL self-play with MCTS-to-diffusion mixing    |
| `uv run denoisr-play`          | UCI chess engine (single-pass or diffusion)            |
| `uv run denoisr-benchmark`     | Parallel Elo benchmarking against Stockfish            |
| `uv run denoisr-export-mlx`    | Export checkpoint to MLX safetensors for Apple Silicon |
| `uv run denoisr-gui`           | Chess GUI for play and engine-vs-engine matches        |

All commands support `--help` for full flag documentation:

```bash
uv run denoisr-train --help
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
│   ├── pipeline/       # Unified pipeline: env config, state persistence, step runner
│   ├── engine/         # Shared engine infrastructure (UCI, match engine, Elo/SPRT, openings)
│   ├── gui/            # Built-in chess GUI (play, match)
│   ├── inference/      # Chess engines (single-pass, diffusion-enhanced), UCI protocol
│   ├── evaluation/     # Self-contained parallel benchmarking
│   └── scripts/        # CLI entry points for all phases + inference + benchmarking
├── tests/              # Test suite mirroring src/ structure
├── fixtures/           # Sample PGN files for testing
├── logs/               # Unified training/runtime logs (gitignored)
└── outputs/            # Training artifacts (gitignored)
```
