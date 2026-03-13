"""Train an FSQ tokenizer on execution corpora."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.tokenization import train_fsq_tokenizer
from denoisr_crypto.types import DEFAULT_SYMBOLS, StorageLayout, parse_symbol_list
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


@graceful_main("denoisr-crypto-train-fsq-tokenizer", logger=log)
def main() -> None:
    parser = build_parser("Train an FSQ tokenizer on execution sequences")
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
        env_var="DENOISR_EXECUTION_FSQ_EPOCHS",
        default=5,
        help="Number of FSQ tokenizer epochs",
    )
    add_env_argument(
        parser,
        "--batch-size",
        type=int,
        env_var="DENOISR_EXECUTION_FSQ_BATCH_SIZE",
        default=64,
        help="FSQ tokenizer batch size",
    )
    add_env_argument(
        parser,
        "--lr",
        type=float,
        env_var="DENOISR_EXECUTION_FSQ_LR",
        default=3e-4,
        help="FSQ tokenizer learning rate",
    )
    add_env_argument(
        parser,
        "--run-name",
        type=str,
        env_var="DENOISR_EXECUTION_FSQ_RUN_NAME",
        default="fsq_tokenizer",
        help="Checkpoint stem for tokenizer artifacts",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    artifacts = train_fsq_tokenizer(
        layout=StorageLayout(Path(args.storage_root)),
        symbols=args.symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=args.run_name,
    )
    if artifacts is None:
        raise FileNotFoundError(
            "Missing tokenizer corpus artifacts. "
            "Run denoisr-crypto-build-tokenizer-corpus before training the tokenizer."
        )
    log.info("checkpoint -> %s", artifacts.checkpoint_path)
    log.info("metrics -> %s", artifacts.metrics_path)


if __name__ == "__main__":
    main()
