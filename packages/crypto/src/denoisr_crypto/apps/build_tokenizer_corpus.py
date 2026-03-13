"""Build tokenizer-ready corpora from canonical 1m OHLCV bars."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.tokenization import build_tokenizer_corpus
from denoisr_crypto.types import (
    DEFAULT_SYMBOLS,
    DEFAULT_TOKENIZER_CONTEXT_LENGTH,
    DEFAULT_TOKENIZER_STRIDE,
    StorageLayout,
    parse_symbol_list,
)
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


@graceful_main("denoisr-crypto-build-tokenizer-corpus", logger=log)
def main() -> None:
    parser = build_parser("Build tokenizer corpora from execution bars")
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
        "--context-length",
        type=int,
        env_var="DENOISR_EXECUTION_TOKEN_CONTEXT",
        default=DEFAULT_TOKENIZER_CONTEXT_LENGTH,
        help="Number of 1m bars per tokenizer sequence",
    )
    add_env_argument(
        parser,
        "--stride",
        type=int,
        env_var="DENOISR_EXECUTION_TOKEN_STRIDE",
        default=DEFAULT_TOKENIZER_STRIDE,
        help="Stride between tokenizer windows",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    artifacts = build_tokenizer_corpus(
        layout=StorageLayout(Path(args.storage_root)),
        symbols=args.symbols,
        context_length=args.context_length,
        stride=args.stride,
    )
    if artifacts is None:
        raise FileNotFoundError(
            "Missing canonical 1m bars. "
            "Run denoisr-crypto-collect-binance before building tokenizer corpora."
        )
    log.info("corpus_manifest -> %s", artifacts.manifest_path)
    for symbol, path in sorted(artifacts.metadata_paths.items()):
        log.info("%s_corpus_metadata -> %s", symbol, path)


if __name__ == "__main__":
    main()
