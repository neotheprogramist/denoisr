"""Train a simple supervised baseline on execution feature artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.training.baseline import train_baseline
from denoisr_crypto.types import (
    DEFAULT_SYMBOLS,
    StorageLayout,
    parse_symbol_list,
)
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


@graceful_main("denoisr-crypto-train-poc", logger=log)
def main() -> None:
    parser = build_parser("Train the execution POC baseline model")
    add_env_argument(
        parser,
        "--storage-root",
        type=Path,
        env_var="DENOISR_STORAGE_ROOT",
        default=Path("data"),
        help="Root directory for execution data",
    )
    add_env_argument(
        parser,
        "--symbols",
        type=parse_symbol_list,
        env_var="DENOISR_EXECUTION_SYMBOLS",
        default=DEFAULT_SYMBOLS,
        help="Comma-separated Binance spot symbols",
    )
    add_env_argument(
        parser,
        "--epochs",
        type=int,
        env_var="DENOISR_EXECUTION_BASELINE_EPOCHS",
        default=10,
        help="Number of baseline training epochs",
    )
    add_env_argument(
        parser,
        "--batch-size",
        type=int,
        env_var="DENOISR_EXECUTION_BASELINE_BATCH_SIZE",
        default=512,
        help="Baseline training batch size",
    )
    add_env_argument(
        parser,
        "--lr",
        type=float,
        env_var="DENOISR_EXECUTION_BASELINE_LR",
        default=1e-3,
        help="Baseline training learning rate",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    artifacts = train_baseline(
        layout=StorageLayout(Path(args.storage_root)),
        symbols=args.symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    if artifacts is None:
        raise FileNotFoundError(
            "Missing canonical feature datasets. "
            "Run denoisr-crypto-build-features before training the baseline."
        )
    log.info("checkpoint -> %s", artifacts.checkpoint_path)
    log.info("metrics -> %s", artifacts.metrics_path)


if __name__ == "__main__":
    main()
