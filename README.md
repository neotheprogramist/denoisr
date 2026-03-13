# Denoisr

Denoisr is a `uv` workspace with two separate projects:

- `denoisr-chess`: chess model training, inference, GUI play, and benchmarking
- `denoisr-crypto`: crypto market-data collection, feature engineering, entry-quality modeling, backtesting, and tokenization

There is also one small shared package:

- `denoisr-common`: runtime and interrupt helpers used by both projects

## Workspace Setup

Clone the repository and sync the full workspace:

```bash
git clone <repo-url> && cd denoisr
uv sync --all-packages --dev
```

Run commands from the workspace root with an explicit package target:

```bash
uv run --package denoisr-chess denoisr-chess-train --help
uv run --package denoisr-crypto denoisr-crypto-collect-binance --help
```

## Repository Structure

```text
packages/
├── common/   # shared runtime helpers only
├── chess/    # chess package
└── crypto/   # crypto package

tests/
├── test_common/
├── test_chess/
└── test_crypto/
```

Active source roots:

- `packages/common/src/denoisr_common`
- `packages/chess/src/denoisr_chess`
- `packages/crypto/src/denoisr_crypto`

## Chess Guide

Chess code lives under `packages/chess/src/denoisr_chess`.

Main areas:

```text
packages/chess/src/denoisr_chess/
├── apps/        # CLI entrypoints
├── config/      # model and training configuration
├── data/        # encoders, PGN streaming, Stockfish targets
├── engine/      # UCI and benchmarking engine helpers
├── evaluation/
├── game/
├── gui/
├── inference/
├── nn/
├── pipeline/    # unified training pipeline
├── training/
└── types/
```

### Prerequisites

- Install Stockfish if you want to generate supervised data or benchmark against a strong engine.
- MLX export is optional and only relevant on Apple Silicon.

### Quick Start

Create an untrained checkpoint and open the GUI:

```bash
uv run --package denoisr-chess denoisr-chess-init \
    --output outputs/random_model.pt

uv run --package denoisr-chess denoisr-chess-gui \
    --checkpoint outputs/random_model.pt
```

Run the engine without the GUI:

```bash
uv run --package denoisr-chess denoisr-chess-play \
    --checkpoint outputs/random_model.pt \
    --mode single
```

### Training Workflows

Recommended entrypoint:

```bash
uv run --package denoisr-chess denoisr-chess-train --help
```

That command drives the chess pipeline defined in `denoisr_chess.pipeline`.

Lower-level phase commands:

```bash
uv run --package denoisr-chess denoisr-chess-generate-data --help
uv run --package denoisr-chess denoisr-chess-train-phase1 --help
uv run --package denoisr-chess denoisr-chess-train-phase2 --help
uv run --package denoisr-chess denoisr-chess-train-phase3 --help
```

Typical manual flow:

1. Generate supervised chess training data from PGN + Stockfish.
2. Train phase 1 supervised policy/value models.
3. Train phase 2 world model and diffusion components.
4. Train phase 3 self-play RL.

### Evaluation and Export

```bash
uv run --package denoisr-chess denoisr-chess-benchmark --help
uv run --package denoisr-chess denoisr-chess-export-mlx --help
```

## Crypto Guide

Crypto code lives under `packages/crypto/src/denoisr_crypto`.

Main areas:

```text
packages/crypto/src/denoisr_crypto/
├── apps/          # CLI entrypoints
├── data/          # ingestion, schemas, catalog, validation
├── evaluation/    # bars-only simulator
├── features/      # OHLCV feature engineering
├── labels/        # forward entry-quality labels
├── tokenization/  # corpus building and FSQ tokenizer
├── training/      # baseline and entry-quality model training
├── visualization/
└── types.py
```

### Current Scope

- Binance spot historical data only
- Local Parquet-first research workflow
- Canonical data lake under `data/execution/binance`
- `1m` source bars with derived `5m` and `15m`
- Entry-quality supervised datasets and confidence models
- Bars-only execution simulator
- Tokenizer corpus generation and FSQ tokenizer training

This is research tooling, not a live trading system.

### Quick Start

Small one-symbol smoke run:

```bash
uv run --package denoisr-crypto denoisr-crypto-collect-binance \
    --storage-root data \
    --symbols BTCUSDT \
    --interval 1m \
    --start 2025-03-01 \
    --end 2025-03-31

uv run --package denoisr-crypto denoisr-crypto-build-features \
    --storage-root data \
    --symbols BTCUSDT

uv run --package denoisr-crypto denoisr-crypto-build-entry-dataset \
    --storage-root data \
    --symbols BTCUSDT \
    --decision-interval 15m \
    --horizon-hours 48

uv run --package denoisr-crypto denoisr-crypto-train-entry-model \
    --storage-root data \
    --symbols BTCUSDT \
    --decision-interval 15m \
    --loss p6 \
    --epochs 1 \
    --run-name entry_quality_smoke
```

### Full Research Flow

Recommended Phase 1 symbols:

- `BTCUSDT`
- `ETHUSDT`

Recommended window:

- `2025-03-01` to `2026-02-28`

End-to-end flow:

```bash
uv run --package denoisr-crypto denoisr-crypto-collect-binance \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT \
    --interval 1m \
    --start 2025-03-01 \
    --end 2026-02-28

uv run --package denoisr-crypto denoisr-crypto-build-features \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT

uv run --package denoisr-crypto denoisr-crypto-validate-binance \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT

uv run --package denoisr-crypto denoisr-crypto-build-entry-dataset \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT \
    --decision-interval 15m \
    --horizon-hours 48

uv run --package denoisr-crypto denoisr-crypto-train-entry-model \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT \
    --decision-interval 15m \
    --loss p6 \
    --run-name entry_quality_model

uv run --package denoisr-crypto denoisr-crypto-backtest-poc \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT

uv run --package denoisr-crypto denoisr-crypto-build-tokenizer-corpus \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT

uv run --package denoisr-crypto denoisr-crypto-train-fsq-tokenizer \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT \
    --run-name fsq_tokenizer

uv run --package denoisr-crypto denoisr-crypto-export-token-dataset \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT \
    --run-name fsq_tokenizer
```

Optional visualization:

```bash
uv run --package denoisr-crypto denoisr-crypto-visualize-poc \
    --storage-root data \
    --symbols BTCUSDT,ETHUSDT
```

### Crypto Outputs

Canonical output layout:

```text
data/execution/binance/
├── bronze/
├── silver/
│   └── market=spot/dataset=bars/interval=1m|5m|15m/year=YYYY/month=MM/symbol=...
└── gold/
    ├── catalog/
    ├── features/
    ├── training/
    │   ├── labels/
    │   ├── datasets/
    │   ├── baseline/
    │   └── entry_quality/
    └── tokenizer/
        ├── corpus/
        ├── models/
        └── exports/
```

If a required upstream artifact is missing, the crypto CLIs fail immediately with a descriptive error.

## Shared Components

Shared code is intentionally small:

```text
packages/common/src/denoisr_common/
├── interrupts.py
└── runtime.py
```

Anything domain-specific should stay in `denoisr_chess` or `denoisr_crypto`.

## Testing

Run the full workspace suite:

```bash
uv run pytest -q
```

Run only one project:

```bash
uv run pytest tests/test_common -q
uv run pytest tests/test_chess -q
uv run pytest tests/test_crypto -q
```
