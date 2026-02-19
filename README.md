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
ChessDataset  ──>  PyTorch DataLoader batches
```

### How to run

```python
import torch
from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.data.dataset import ChessDataset, generate_examples_from_game
from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.data.stockfish_oracle import StockfishOracle
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer

# --- Configuration ---
D_S = 256               # latent dimension
NUM_HEADS = 8           # attention heads
NUM_LAYERS = 15         # transformer layers
FFN_DIM = 1024          # feedforward dimension
LR = 1e-4               # base learning rate (heads get 1x, backbone 0.3x, encoder 0.1x)
STOCKFISH_PATH = "/usr/local/bin/stockfish"
PGN_PATH = "data/lichess_2026-01.pgn.zst"

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Build models ---
encoder = ChessEncoder(num_planes=12, d_s=D_S).to(device)
backbone = ChessPolicyBackbone(
    d_s=D_S, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, ffn_dim=FFN_DIM
).to(device)
policy_head = ChessPolicyHead(d_s=D_S).to(device)
value_head = ChessValueHead(d_s=D_S).to(device)

# --- Loss with HarmonyDream for dynamic coefficient balancing ---
# HarmonyDream tracks EMA of per-loss gradient norms and adjusts
# coefficients inversely proportional to balance contributions.
# This prevents any single loss term from dominating training.
loss_fn = ChessLossComputer(use_harmony_dream=True)

# --- Trainer with tiered learning rates + gradient clipping ---
trainer = SupervisedTrainer(
    encoder=encoder,
    backbone=backbone,
    policy_head=policy_head,
    value_head=value_head,
    loss_fn=loss_fn,
    lr=LR,
    device=device,
)

# --- Generate training data ---
streamer = SimplePGNStreamer()
board_encoder = SimpleBoardEncoder()

with StockfishOracle(path=STOCKFISH_PATH, depth=12) as oracle:
    examples = []
    for game_record in streamer.stream(PGN_PATH):
        for example in generate_examples_from_game(game_record, board_encoder, oracle):
            examples.append(example)
        if len(examples) >= 100_000:
            break

dataset = ChessDataset(examples)

# --- Train ---
from torch.utils.data import DataLoader

# ChessDataset returns (board_tensor, policy_tensor, value_tensor) tuples
# but SupervisedTrainer.train_step expects list[TrainingExample]
# so we train directly with example batches:

for epoch in range(100):
    batch = examples[:64]  # replace with proper DataLoader sampling
    total_loss, breakdown = trainer.train_step(batch)
    print(f"Epoch {epoch}: loss={total_loss:.4f} policy={breakdown['policy']:.4f} value={breakdown['value']:.4f}")

# --- Save checkpoint ---
from pathlib import Path
trainer.save_checkpoint(Path("outputs/phase1_checkpoint.pt"))
```

### Gate to Phase 2

Policy accuracy must exceed **30% top-1** on a held-out set of 10,000 positions. This threshold confirms the network has learned enough positional understanding to provide meaningful latent representations for the world model and diffusion module.

```python
from denoisr.training.phase_orchestrator import PhaseOrchestrator, PhaseConfig

orchestrator = PhaseOrchestrator(PhaseConfig(phase1_gate=0.30))

# Evaluate on held-out positions
top1_accuracy = evaluate_top1(trainer, held_out_examples)  # you implement this
if orchestrator.check_gate({"top1_accuracy": top1_accuracy}):
    print(f"Phase 1 complete! top-1 accuracy: {top1_accuracy:.1%}")
    print(f"Advancing to Phase {orchestrator.current_phase}")
```

## Phase 2: World model + diffusion bootstrapping

### What happens

Two new modules are trained on top of the frozen (or slowly-updating) Phase 1 representations:

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

```python
import torch
from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.world_model import ChessWorldModel
from denoisr.nn.consistency import ChessConsistencyProjector
from denoisr.training.diffusion_trainer import DiffusionTrainer

# --- Load Phase 1 checkpoint ---
trainer.load_checkpoint(Path("outputs/phase1_checkpoint.pt"))

# --- Build Phase 2 modules ---
NUM_TIMESTEPS = 100

world_model = ChessWorldModel(
    d_s=D_S, num_heads=NUM_HEADS, num_layers=12, ffn_dim=FFN_DIM
).to(device)
diffusion = ChessDiffusionModule(
    d_s=D_S, num_heads=NUM_HEADS, num_layers=6, num_timesteps=NUM_TIMESTEPS
).to(device)
consistency = ChessConsistencyProjector(d_s=D_S, proj_dim=256).to(device)
schedule = CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)

# --- Diffusion trainer (encoder frozen, only diffusion params updated) ---
diff_trainer = DiffusionTrainer(
    encoder=encoder,
    diffusion=diffusion,
    schedule=schedule,
    lr=1e-4,
    device=device,
)

# --- Train on trajectories extracted from Lichess games ---
# trajectories shape: [batch, time_steps, channels, 8, 8]
# e.g. 5 consecutive board states from the same game
for epoch in range(200):
    trajectories = sample_trajectory_batch(examples, seq_len=5)  # you implement this
    loss = diff_trainer.train_step(trajectories)

    # Advance curriculum every epoch (25% -> 100% of timesteps over training)
    diff_trainer.advance_curriculum()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: diffusion_loss={loss:.4f} max_steps={diff_trainer._current_max_steps}")

# Continue supervised training with all 6 loss terms active
for epoch in range(100):
    batch = examples[:64]
    total_loss, breakdown = trainer.train_step(batch)
    # Add auxiliary losses for consistency, diffusion, reward, ply:
    # total_loss, breakdown = loss_fn.compute(
    #     pred_policy, pred_value, target_policy, target_value,
    #     consistency_loss=..., diffusion_loss=..., reward_loss=..., ply_loss=...
    # )
```

### Gate to Phase 3

Diffusion-conditioned inference accuracy must exceed single-step accuracy by **>5 percentage points**. This confirms the diffusion module provides meaningful information beyond what the backbone already captures.

```python
single_step_acc = evaluate_engine(engine, held_out)
diffusion_acc = evaluate_engine(diffusion_engine, held_out)
improvement = (diffusion_acc - single_step_acc) * 100  # percentage points

if orchestrator.check_gate({"diffusion_improvement_pp": improvement}):
    print(f"Phase 2 complete! Diffusion improvement: +{improvement:.1f}pp")
    print(f"Advancing to Phase {orchestrator.current_phase}")
```

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

```python
from denoisr.training.self_play import SelfPlayActor, SelfPlayConfig, TemperatureSchedule
from denoisr.training.mcts import MCTS, MCTSConfig
from denoisr.training.replay_buffer import PriorityReplayBuffer
from denoisr.training.reanalyse import ReanalyseActor
from denoisr.training.phase_orchestrator import PhaseOrchestrator, PhaseConfig

# --- Self-play configuration ---
temp_schedule = TemperatureSchedule(
    base=1.0,
    explore_moves=30,      # stochastic for first 30 moves
    generation_decay=0.97,  # 3% decay per generation
)

config = SelfPlayConfig(
    num_simulations=800,   # MCTS simulations per move
    max_moves=300,
    temperature=1.0,
    c_puct=1.4,
    temp_schedule=temp_schedule,
)

# --- Build actor (uses latest model weights for MCTS) ---
def policy_value_fn(state):
    """Wraps model forward pass for MCTS."""
    features = backbone(state.unsqueeze(0))
    policy = policy_head(features).squeeze(0)
    wdl, _ = value_head(features)
    return policy, wdl.squeeze(0)

def world_model_fn(state, from_sq, to_sq):
    """Wraps world model for latent-space MCTS transitions."""
    action_tensor = torch.zeros(64, 64)
    action_tensor[from_sq, to_sq] = 1.0
    # ... encode action and step world model
    return next_state, reward

def encode_fn(board_tensor):
    return encoder(board_tensor.unsqueeze(0)).squeeze(0)

actor = SelfPlayActor(
    policy_value_fn=policy_value_fn,
    world_model_fn=world_model_fn,
    encode_fn=encode_fn,
    game=ChessGame(),
    board_encoder=SimpleBoardEncoder(),
    config=config,
)

# --- Replay buffer with priority sampling ---
buffer = PriorityReplayBuffer(capacity=100_000)

# --- Reanalyse actor ---
reanalyser = ReanalyseActor(
    policy_value_fn=policy_value_fn,
    world_model_fn=world_model_fn,
    encode_fn=encode_fn,
    game=ChessGame(),
    board_encoder=SimpleBoardEncoder(),
    num_simulations=100,
)

# --- Phase orchestrator ---
orchestrator = PhaseOrchestrator(PhaseConfig(alpha_generations=50))
# Advance to phase 3 (assumes gates already passed)
orchestrator.check_gate({"top1_accuracy": 0.35})
orchestrator.check_gate({"diffusion_improvement_pp": 6.0})

# --- Training loop ---
for generation in range(1000):
    # 1. Generate self-play games
    for _ in range(100):
        record = actor.play_game(generation=generation)
        buffer.add(record, priority=1.0)

    # 2. Reanalyse old games with current network
    old_records = buffer.sample(50)
    for old_record in old_records:
        reanalysed = reanalyser.reanalyse(old_record)
        # Train on reanalysed examples with improved policy targets

    # 3. Train on batch from replay buffer
    batch = buffer.sample(64)
    # ... convert to training examples and run trainer.train_step()

    # 4. Alpha mixing (Phase 3b): gradually shift from MCTS to diffusion
    alpha = orchestrator.get_alpha(generation)
    if alpha > 0:
        # mixed_policy = orchestrator.mix_policies(mcts_policy, diff_policy, alpha)
        pass

    print(f"Gen {generation}: buffer={len(buffer)} alpha={alpha:.2f} temp_base={temp_schedule.get_temperature(0, generation):.3f}")
```

### Success criteria

Elo increases over generations (measured via cutechess-cli). The ultimate goal: diffusion-only inference (alpha=1.0) matches or exceeds MCTS-based inference strength.

## Inference: playing with the trained model

### Single-pass engine (fastest)

Direct encoder -> backbone -> policy head. No search, no diffusion. The simplest mode.

```python
from denoisr.inference.engine import ChessEngine

engine = ChessEngine(
    encoder=encoder,
    backbone=backbone,
    policy_head=policy_head,
    value_head=value_head,
    board_encoder=SimpleBoardEncoder(),
    device=device,
)

import chess
board = chess.Board()
move = engine.select_move(board)       # returns chess.Move
wdl = engine.evaluate(board)           # returns (win, draw, loss) tuple
print(f"Best move: {move.uci()}, WDL: {wdl}")
```

### Diffusion-enhanced engine (stronger, adjustable)

Adds diffusion imagination before the policy backbone. The `num_denoising_steps` parameter provides **anytime search**: more steps produce stronger play at the cost of inference time. This is the core innovation — the engine thinks deeper by running more denoising iterations rather than by building an explicit search tree.

```python
from denoisr.inference.diffusion_engine import DiffusionChessEngine
from denoisr.nn.diffusion import CosineNoiseSchedule

diffusion_engine = DiffusionChessEngine(
    encoder=encoder,
    backbone=backbone,
    policy_head=policy_head,
    value_head=value_head,
    diffusion=diffusion,
    schedule=CosineNoiseSchedule(num_timesteps=100),
    board_encoder=SimpleBoardEncoder(),
    device=device,
    num_denoising_steps=20,  # more steps = stronger but slower
)

move = diffusion_engine.select_move(board)
wdl = diffusion_engine.evaluate(board)
```

### UCI protocol (connect to any chess GUI)

The UCI wrapper lets you connect Denoisr to any standard chess GUI (CuteChess, Arena, Banksia, etc.):

```python
from denoisr.inference.uci import run_uci_loop

# run_uci_loop reads from stdin, writes to stdout
run_uci_loop(engine_select_move_fn=engine.select_move)
```

Or wrap as a script:

```python
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["denoisr"]
# ///

from denoisr.inference.uci import run_uci_loop
# ... load model, create engine ...
run_uci_loop(engine_select_move_fn=engine.select_move)
```

Then point your chess GUI at this script as a UCI engine.

### Benchmarking with cutechess-cli

Measure Elo against a reference engine using SPRT (Sequential Probability Ratio Test) for statistical confidence:

```python
from denoisr.evaluation.benchmark import BenchmarkConfig, build_cutechess_command, parse_cutechess_output

config = BenchmarkConfig(
    engine_cmd="./run_denoisr.sh",
    opponent_cmd="stockfish",
    games=1000,
    time_control="10+0.1",
    sprt_elo0=0,       # null hypothesis: no Elo difference
    sprt_elo1=50,       # alternative: at least 50 Elo stronger
    concurrency=4,
)

cmd = build_cutechess_command(config)
print(cmd)
# cutechess-cli -engine cmd=./run_denoisr.sh proto=uci -engine cmd=stockfish proto=uci ...

# After running, parse the output:
result = parse_cutechess_output(output)
print(f"Elo: {result['elo_diff']:.1f} +/- {result['elo_error']:.1f}")
if result.get("sprt_result") == "H1":
    print("SPRT: Confirmed stronger!")
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
│   └── evaluation/     # cutechess-cli benchmarking harness
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
