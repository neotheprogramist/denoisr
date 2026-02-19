# Component Decomposition Design — Denoisr

A transformer-diffusion chess engine without game rules, decomposed into atomic testable components.

## Decisions

- **Hardware**: Apple Silicon (MPS) for dev, CUDA for scale training
- **Testing**: Property-based + unit tests (pytest + hypothesis)
- **Scope**: All 3 phases designed, implement Phase 1 first
- **Diffusion**: Latent-space (64×d_s tensors), not FEN token sequences
- **World model**: Full 12-layer causal transformer from the start
- **Decomposition**: Atomic components with Protocol interfaces (Approach C)

## Architectural Justifications

### Why iterative computation is mandatory (TC⁰ limitation)

Standard transformers with fixed depth L and O(log n) precision fall within **DLOGTIME-uniform TC⁰** — constant-depth, polynomial-size threshold circuits (Merrill, Sabharwal & Smith 2022–2023; Chiang et al. 2024). TC⁰ contains pattern matching, multiplication, sorting — but is widely conjectured to exclude Boolean formula evaluation, tree traversal, and **minimax search**. Evaluating a game tree of depth d requires Ω(d) sequential steps. A single transformer forward pass can learn strong heuristic evaluations (within TC⁰) but **cannot perform the depth of reasoning required for optimal play**. This makes iterative computation mechanisms mandatory.

Three candidates exist: looped transformers (Turing complete but unproven for game-playing), autoregressive chain-of-thought (MAV, +2923 Elo but expensive and non-correcting), and **diffusion-based iterative refinement**. Diffusion is chosen because: (1) T denoising steps provide effective depth O(L×T), breaking TC⁰, (2) it refines the entire trajectory simultaneously (bidirectional), (3) denoising steps are adjustable at inference (anytime search), (4) DiffuSearch validates it empirically on chess (+540 Elo over searchless, +14% over MCTS).

### Value equivalence principle

The world model's learned latent dynamics need not reconstruct the full board state. The value equivalence principle (Grimm, Barreto & Singh, NeurIPS 2020–2021) shows that latent dynamics only need to satisfy: T_m̃^π v = T_m\*^π v for all policies π and value functions v — they only preserve information relevant to value estimation and policy improvement. MuZero minimizes an upper bound on the proper value equivalence loss. This dramatically reduces representational burden.

### Why UniZero (vs alternatives)

- **DIAMOND** uses diffusion world modeling in pixel space with U-Net — computationally prohibitive for chess's long-horizon planning, incompatible with latent-space MCTS
- **IRIS** uses VQ-VAE + GPT — requires multiple tokens per frame, lacks search compatibility
- **STORM** uses DreamerV3-style actor-critic (no search) — less sample-efficient than MCTS-based policy improvement
- **UniZero** is the only transformer world model maintaining full MCTS compatibility while gaining attention-based dynamics expressivity

### Why attention over convolution

Cordonnier et al. (ICLR 2020) proved multi-head self-attention can express any convolutional layer — attention strictly subsumes convolution. A NeurIPS 2022 result shows approximating the self-attention function class with permutation-invariant FC networks requires exponential width: W\*(ξ, d, F) = Ω(exp(d)). Empirical confirmation: BT4 is **270 Elo stronger** than the best CNN (T78) with 40% fewer FLOPs.

### Key empirical anchors

| Result                                 | Source                 | Impact                                                 |
| -------------------------------------- | ---------------------- | ------------------------------------------------------ |
| BT4 +270 Elo over CNN                  | Lc0 project            | Validates transformer backbone                         |
| AlphaVile +180 Elo from input features | Czech et al. ECAI 2024 | Motivates extended board encoder                       |
| DiffuSearch +540 Elo over searchless   | Ye et al. ICLR 2025    | Validates diffusion for chess                          |
| HarmonyDream 10–69% improvement        | Ma et al. ICML 2024    | Motivates dynamic loss balancing                       |
| EfficientZero consistency loss         | Yu et al.              | Most impactful contribution — prevents latent collapse |

## MPS/CUDA Development Guidelines

- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` during development for unsupported MPS operations
- Use **float32 or bfloat16** throughout — no float64 (unsupported on MPS)
- Use `torch.nn.functional.scaled_dot_product_attention` instead of FlashAttention directly — dispatches optimally on both MPS and CUDA
- Expect **2–3× slower training** on MPS vs CUDA; develop on MPS, train at scale on CUDA

## Component Map

### Layer 0 — Domain Types (zero ML dependencies)

| Component        | Key types                                                                 |
| ---------------- | ------------------------------------------------------------------------- |
| `types.board`    | `BoardTensor` (frozen, shape `[C, 8, 8]`), `Square`, `PieceType`, `Color` |
| `types.action`   | `Action` (source-dest pair), `ActionIndex`, `LegalMask`                   |
| `types.latent`   | `LatentState` (frozen, shape `[64, d_s]`), `LatentTrajectory`             |
| `types.training` | `TrainingExample`, `GameRecord`, `PolicyTarget`, `ValueTarget` (WDL)      |

### Layer 1 — Game Interface

| Component         | Protocol        | Responsibility                                             |
| ----------------- | --------------- | ---------------------------------------------------------- |
| `game.chess_game` | `GameInterface` | Alpha-zero-general 8-method contract wrapping python-chess |

### Layer 2 — Data Pipeline

| Component                     | Protocol        | Input → Output                                                                                                                            |
| ----------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `data.pgn_streamer`           | `PGNStreamer`   | `.pgn.zst` path → stream of `GameRecord`                                                                                                  |
| `data.board_encoder`          | `BoardEncoder`  | `chess.Board` → `BoardTensor` (12-plane simple)                                                                                           |
| `data.extended_board_encoder` | `BoardEncoder`  | `chess.Board` → `BoardTensor` (~112-plane AlphaVile FX: history, castling, en passant, rule-50, repetition, material, check, opp bishops) |
| `data.action_encoder`         | `ActionEncoder` | `chess.Move` ↔ `ActionIndex` (bijective)                                                                                                  |
| `data.stockfish_oracle`       | `Oracle`        | `chess.Board` → `(PolicyTarget, ValueTarget, centipawn_eval)`                                                                             |
| `data.dataset`                | `ChessDataset`  | `list[GameRecord]` → PyTorch `Dataset`                                                                                                    |

### Layer 3 — Neural Network Modules

| Component            | Protocol               | Signature                                                             | Params |
| -------------------- | ---------------------- | --------------------------------------------------------------------- | ------ |
| `nn.encoder`         | `Encoder`              | `BoardTensor → LatentState`                                           | ~5M    |
| `nn.smolgen`         | `SmolgenBias`          | `LatentState → attention_bias [H×64×64]`                              | ~2M    |
| `nn.relative_pos`    | `RelativePositionBias` | `→ position_bias [H×64×64]` (Shaw relative PE, topology-aware)        | ~1K    |
| `nn.policy_backbone` | `PolicyBackbone`       | `LatentState → LatentState` (15-layer transformer, smolgen + Shaw PE) | ~150M  |
| `nn.policy_head`     | `PolicyHead`           | `LatentState → move_logits [64×64]`                                   | ~2M    |
| `nn.value_head`      | `ValueHead`            | `LatentState → (wdl_probs [3], ply [1])` (WDLP)                       | ~5K    |
| `nn.world_model`     | `WorldModel`           | `(LatentState, Action) → (LatentState, reward)`                       | ~100M  |
| `nn.diffusion`       | `DiffusionModule`      | `(noisy_latent, timestep, condition) → denoised_latent`               | ~80M   |
| `nn.consistency`     | `ConsistencyProjector` | `LatentState → projection [256]`                                      | ~1M    |

Total: ~340M parameters.

### Layer 4 — Training Infrastructure

| Component                     | Protocol            | Responsibility                                                                                             |
| ----------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| `training.loss`               | `LossComputer`      | 6-term combined loss (policy, value, consistency, diffusion, reward, ply) + HarmonyDream dynamic balancing |
| `training.supervised_trainer` | `Trainer`           | Supervised loop with parameter groups (separate LRs) + gradient clipping                                   |
| `training.replay_buffer`      | `ReplayBuffer`      | Simple FIFO (Phase 1) + Priority-based (Phase 3, EfficientZero V2)                                         |
| `training.mcts`               | `MCTS`              | Latent-space Monte Carlo Tree Search with Dirichlet noise                                                  |
| `training.self_play`          | `SelfPlayActor`     | Run games using MCTS with temperature scheduling                                                           |
| `training.diffusion_trainer`  | `DiffusionTrainer`  | DDPM denoising loss + diffusion step curriculum                                                            |
| `training.reanalyse`          | `ReanalyseActor`    | MuZero Reanalyse: re-run MCTS on old trajectories with current network                                     |
| `training.phase_orchestrator` | `PhaseOrchestrator` | Phase gates (30% top-1, +5pp diffusion), α mixing (MCTS→diffusion)                                         |

### Layer 5 — Inference & Interface

| Component                    | Protocol               | Responsibility                                                               |
| ---------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| `inference.engine`           | `ChessEngine`          | Single-pass inference: encoder → backbone → policy head                      |
| `inference.diffusion_engine` | `DiffusionChessEngine` | DiffuSearch: encoder → diffusion imagination → fused policy (anytime search) |
| `inference.uci`              | —                      | UCI protocol stdin/stdout wrapper                                            |

### Layer 6 — Evaluation

| Component              | Protocol | Responsibility                                                       |
| ---------------------- | -------- | -------------------------------------------------------------------- |
| `evaluation.benchmark` | —        | cutechess-cli wrapper: SPRT testing, Elo measurement, result parsing |

## Dependency Graph & Build Order

Seven tiers, each gated on the previous tier's tests passing:

| Tier | Components                                                                                                                                                                                                                                                 | Tests validate                                                    |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| T1   | `types.board`, `types.action`, `types.latent`, `types.training`                                                                                                                                                                                            | Shape invariants, immutability, serde                             |
| T2   | `game.chess_game`, `data.board_encoder`, `data.extended_board_encoder`, `data.action_encoder`                                                                                                                                                              | Bijective encoding, legal move correctness, AlphaVile features    |
| T3   | `data.pgn_streamer`, `data.stockfish_oracle`, `data.dataset`                                                                                                                                                                                               | Streaming, valid targets, batch shapes                            |
| T4   | `nn.encoder`, `nn.smolgen`, `nn.relative_pos`, `nn.value_head` (WDLP), `nn.policy_head`                                                                                                                                                                    | Shape contracts, gradient flow, topology-aware PE, ply prediction |
| T5   | `nn.policy_backbone` (smolgen+Shaw), `nn.world_model`, `nn.consistency`, `nn.diffusion`                                                                                                                                                                    | Forward pass, causal masking, cosine schedule                     |
| T6   | `training.loss` (6-term+HarmonyDream), `training.replay_buffer` (simple+priority), `training.mcts`, `training.supervised_trainer`, `training.diffusion_trainer`, `training.self_play` (temp schedule), `training.reanalyse`, `training.phase_orchestrator` | Loss finite, MCTS converges, phase gates, α mixing                |
| T7   | `inference.engine`, `inference.diffusion_engine`, `inference.uci`, `evaluation.benchmark`                                                                                                                                                                  | Legal moves, anytime search, UCI compliance, Elo measurement      |

**Gate principle**: No tier starts until the previous tier has 100% passing tests.

## Three Training Phases

### Phase 1: Supervised Learning (Lichess + Stockfish)

Components: T1–T4, `nn.policy_backbone`, `training.loss`, `training.supervised_trainer`

```
lichess_2026-01.pgn.zst → pgn_streamer → board_encoder → stockfish_oracle → dataset
  → encoder → policy_backbone → policy_head → cross-entropy vs Stockfish policy
                              → value_head  → cross-entropy vs Stockfish WDL
```

**Gate to Phase 2**: Policy accuracy >30% top-1 on 10k held-out positions.

### Phase 2: World Model + Diffusion Bootstrapping

Adds: `nn.world_model`, `nn.diffusion`, `nn.consistency`, `training.diffusion_trainer`, all 6 loss terms.

```
Encoded Lichess trajectories → world_model learns latent dynamics
                             → diffusion_trainer learns to denoise future trajectories
Inference: encoder → diffusion imagines future → enriched policy prediction
```

**Gate to Phase 3**: Diffusion-conditioned accuracy exceeds single-step by >5pp.

### Phase 3: RL Self-Play

Adds: `training.mcts`, `training.self_play`, `training.replay_buffer`, `inference.engine`, `inference.uci`

- **3a**: MCTS bootstrap — self-play with latent-space MCTS generates improved targets
- **3b**: Diffusion transition — α mixing from MCTS to diffusion-only (α: 0→1)

**Success**: Elo increases over generations. Diffusion-only matches MCTS strength.

## How the Diffusion Model Learns to Score Moves Through Self-Play

The core idea is surprisingly intuitive once you strip away the math.

### The basic concept: "Imagine the future, then pick the first move"

The diffusion model doesn't score moves directly like a traditional engine ("e4 is worth +0.3 pawns"). Instead, it learns to **dream up plausible future game continuations**, and then picks the move that leads to the best futures. Think of it like a chess player who doesn't calculate precisely but has a strong "feel" for where the game is heading.

### Step 1: Learning what real games look like

During self-play, the engine plays thousands of games against itself. Each game produces a sequence of positions and moves:

```
Position₁ → move₁ → Position₂ → move₂ → Position₃ → ... → Result (White wins)
```

The diffusion model trains on **trajectory snippets** from these games. For example, from a single game it might extract:

```
Current: White has rook on e1, black king on e8, open e-file
Future₁: Rook moves to e7 (7th rank)
Future₂: King pushed to back rank
Future₃: Checkmate delivered
Result: White wins
```

The model learns: "When there's a rook on an open file pointing at the king, games tend to continue like _this_."

### Step 2: The corruption-and-recovery game

Here's where diffusion gets clever. During training, the model plays a game with itself:

1. **Take a real future trajectory** (say, 4 moves ahead from the actual self-play game)
2. **Corrupt it with noise** — randomly replace some of the future positions/moves with garbage (like covering parts of a photograph with static)
3. **Ask the model to recover the original** — predict what the clean future should be

A concrete toy example with three possible moves from a position:

```
Real trajectory:    [Current pos] → Nf3 → [pos2] → d5 → [pos3] → Bg5 → [pos4]
Corrupted (t=15):   [Current pos] → ??? → [????] → ?? → [????] → ??? → [????]
Corrupted (t=10):   [Current pos] → N?3 → [partial] → d? → [????] → ??? → [????]
Corrupted (t=5):    [Current pos] → Nf3 → [pos2] → d5 → [partial] → B?? → [????]
```

At heavy noise (t=15), almost everything is masked. At light noise (t=5), only the distant future is unclear. The model learns to reconstruct at **every noise level**, which teaches it both local tactics (easy, low noise) and long-range strategy (hard, high noise).

### Step 3: How this becomes move scoring at inference

At game time, the model does this:

1. **Start with the real current position** and **pure noise** for the future:

   ```
   [Real position] → ???? → ???? → ???? → ????
   ```

2. **Denoise iteratively** (say 20 steps). Each step, the model looks at the partially-cleaned future and refines it:

   ```
   Step 1:  [Real pos] → ???e → ???? → ???? → ????
   Step 5:  [Real pos] → Nf3? → ?d5? → ???? → ????
   Step 10: [Real pos] → Nf3 → [pos2] → d5 → ????
   Step 20: [Real pos] → Nf3 → [pos2] → d5 → [pos3] → Bg5 → [pos4]
   ```

3. **Read off the first move** from the denoised trajectory. The model effectively "imagined" the most likely good continuation and told you move one.

### The self-play feedback loop

Here's where the self-play scoring comes in:

**Game 1 (random play):** The engine plays Qh5 early. The game continues chaotically. Black eventually wins. The model learns: "Positions where the queen goes to h5 early tend to lead to trajectories ending in a loss."

**Game 1000:** The engine has gotten slightly better. It plays Nf3, develops pieces, castles. White wins. The model learns: "Positions with solid development tend to lead to winning trajectories."

**Game 100,000:** Now the engine is decent. When it sees a position with a tactical shot (say, a knight fork), the self-play data overwhelmingly shows that games where the fork was played ended in wins. The diffusion model learns to "dream up" futures that include the fork.

### Concrete micro-example: learning a fork

Suppose this position keeps appearing in self-play:

```
White knight on d5, Black queen on c7, Black king on e8
```

Over thousands of games, the self-play data reveals:

- Games where White played **Nc7+** (forking king and rook): White won 85% of the time
- Games where White played **other moves**: White won 45% of the time

The diffusion model sees many trajectory snippets starting from this position. The "Nc7+" trajectories dominate the winning data. When the model denoises from this position at inference, the most probable future trajectory it reconstructs will start with Nc7+ — because that's the pattern most consistent with the winning continuations it learned.

**The move isn't "scored" as +2.5 pawns. Instead, the model has learned that the _most plausible good future_ from this position starts with Nc7+.** The scoring is implicit in which trajectory the denoising process converges to.

### The MCTS bootstrap phase makes this practical

In early training, the self-play games are terrible (random play produces random trajectories — no useful signal). This is why the architecture uses a two-phase approach:

1. **Phase 3a (MCTS bootstrap):** Traditional MCTS (tree search in the world model) generates the self-play games. MCTS provides decent move quality even with a weak neural network, so the training data has meaningful winning/losing patterns from the start.

2. **Phase 3b (Diffusion transition):** Once the diffusion model has absorbed enough patterns from MCTS-quality games, it gradually takes over move selection (via α mixing). Now _its own_ games produce the training data, and a virtuous cycle begins — better diffusion → better games → better training data → better diffusion.

The diffusion model never needs an explicit score for any move. It just needs to learn: "what do futures look like from positions where strong play happened?" The self-play loop ensures that over time, "strong play" gets progressively stronger, and the model's imagined futures get progressively more accurate.

## Project Structure

```
denoisr/
├── src/denoisr/
│   ├── types/          # Layer 0
│   ├── game/           # Layer 1
│   ├── data/           # Layer 2
│   ├── nn/             # Layer 3
│   ├── training/       # Layer 4
│   ├── inference/      # Layer 5
│   └── evaluation/     # Layer 6
├── tests/              # mirrors src/ structure
├── fixtures/           # sample_10games.pgn, known_positions.json
├── scripts/            # run_benchmark.sh, etc.
├── docs/plans/         # design docs
└── outputs/            # training artifacts (gitignored)
```

One `protocols.py` per layer. `src/` layout for editable installs.

## Testing Infrastructure

Dependencies: `pytest`, `hypothesis`, `pytest-xdist`

Key fixtures: `random_board`, `random_board_tensor`, `random_latent`, `device` (auto MPS/CUDA/CPU), `small_d_s` (64 for fast tests).

```bash
uv run pytest tests/test_types/          # one layer
uv run pytest tests/ -x                  # all, stop first failure
uv run pytest tests/ -n auto             # parallel
uv run pytest tests/ -k "not stockfish"  # skip oracle tests
```
