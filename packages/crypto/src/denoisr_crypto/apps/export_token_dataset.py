"""Export discrete token IDs from a trained FSQ tokenizer."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.tokenization import export_token_dataset
from denoisr_crypto.types import DEFAULT_SYMBOLS, StorageLayout, parse_symbol_list
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


@graceful_main("denoisr-crypto-export-token-dataset", logger=log)
def main() -> None:
    parser = build_parser("Export token IDs from a trained FSQ tokenizer")
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
        "--run-name",
        type=str,
        env_var="DENOISR_EXECUTION_FSQ_RUN_NAME",
        default="fsq_tokenizer",
        help="Checkpoint stem for tokenizer artifacts",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    artifacts = export_token_dataset(
        layout=StorageLayout(Path(args.storage_root)),
        symbols=args.symbols,
        run_name=args.run_name,
    )
    if artifacts is None:
        raise FileNotFoundError(
            "Missing tokenizer checkpoint or tokenizer corpus artifacts. "
            "Run denoisr-crypto-train-fsq-tokenizer after building the corpus."
        )
    for (symbol, split), path in sorted(artifacts.export_paths.items()):
        log.info("%s_%s_tokens -> %s", symbol, split, path)


if __name__ == "__main__":
    main()
