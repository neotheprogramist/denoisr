"""Train the entry-quality confidence model."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.training import LossConfig, train_entry_quality_model
from denoisr_crypto.types import (
    DEFAULT_ENTRY_DECISION_INTERVAL,
    DEFAULT_ENTRY_SIGNAL_RATE,
    DEFAULT_SYMBOLS,
    StorageLayout,
    parse_symbol_list,
)
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


@graceful_main("denoisr-crypto-train-entry-model", logger=log)
def main() -> None:
    parser = build_parser("Train the entry-quality confidence model")
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
        "--decision-interval",
        type=str,
        env_var="DENOISR_ENTRY_DECISION_INTERVAL",
        default=DEFAULT_ENTRY_DECISION_INTERVAL,
        help="Decision interval used for the entry dataset",
    )
    add_env_argument(
        parser,
        "--epochs",
        type=int,
        env_var="DENOISR_ENTRY_EPOCHS",
        default=20,
        help="Number of training epochs",
    )
    add_env_argument(
        parser,
        "--batch-size",
        type=int,
        env_var="DENOISR_ENTRY_BATCH_SIZE",
        default=256,
        help="Training batch size",
    )
    add_env_argument(
        parser,
        "--lr",
        type=float,
        env_var="DENOISR_ENTRY_LR",
        default=3e-4,
        help="Training learning rate",
    )
    add_env_argument(
        parser,
        "--run-name",
        type=str,
        env_var="DENOISR_ENTRY_RUN_NAME",
        default="entry_quality_model",
        help="Checkpoint stem for model artifacts",
    )
    add_env_argument(
        parser,
        "--loss",
        type=str,
        env_var="DENOISR_ENTRY_LOSS",
        default="p6",
        help="Loss family: p1, p2, p3, p5, or p6",
    )
    add_env_argument(
        parser,
        "--signal-rate",
        type=float,
        env_var="DENOISR_ENTRY_SIGNAL_RATE",
        default=DEFAULT_ENTRY_SIGNAL_RATE,
        help="Target soft firing rate for sparsity regularization",
    )
    add_env_argument(
        parser,
        "--miss-policy",
        type=str,
        env_var="DENOISR_ENTRY_MISS_POLICY",
        default="opportunity_cost",
        help="No-buy policy: mask or opportunity_cost",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    artifacts = train_entry_quality_model(
        layout=StorageLayout(Path(args.storage_root)),
        symbols=args.symbols,
        decision_interval=args.decision_interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=args.run_name,
        loss_config=LossConfig(
            name=args.loss,
            sparsity_target=args.signal_rate,
            miss_policy=args.miss_policy,
        ),
    )
    if artifacts is None:
        raise FileNotFoundError(
            "Missing entry-quality dataset artifacts. "
            "Run denoisr-crypto-build-entry-dataset before training the model."
        )
    log.info("checkpoint -> %s", artifacts.checkpoint_path)
    log.info("metrics -> %s", artifacts.metrics_path)


if __name__ == "__main__":
    main()
