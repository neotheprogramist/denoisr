"""Derive multi-interval bars and feature tables."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.features.ohlcv import (
    build_feature_artifacts,
    ensure_derived_bars,
)
from denoisr_crypto.types import (
    DEFAULT_ROLLING_ZSCORE_WINDOW,
    DEFAULT_SOURCE_INTERVAL,
    DEFAULT_SYMBOLS,
    StorageLayout,
    parse_symbol_list,
)
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


def _missing_bars_message(layout: StorageLayout, *, symbol: str, interval: str) -> str:
    return (
        f"Missing canonical {interval} bars for {symbol} under "
        f"{layout.silver_dataset_dir(interval)}. "
        "Run denoisr-crypto-collect-binance first."
    )


@graceful_main("denoisr-crypto-build-features", logger=log)
def main() -> None:
    parser = build_parser("Build multi-interval execution feature artifacts")
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
        "--source-interval",
        type=str,
        env_var="DENOISR_EXECUTION_INTERVAL",
        default=DEFAULT_SOURCE_INTERVAL,
        help="Source interval to derive from",
    )
    add_env_argument(
        parser,
        "--normalize-window",
        type=int,
        env_var="DENOISR_EXECUTION_NORMALIZE_WINDOW",
        default=DEFAULT_ROLLING_ZSCORE_WINDOW,
        help="Rolling window for normalized feature columns",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    layout = StorageLayout(Path(args.storage_root))
    for symbol in args.symbols:
        derived_outputs = ensure_derived_bars(
            layout=layout,
            symbol=symbol,
            source_interval=args.source_interval,
        )
        if derived_outputs is None:
            raise FileNotFoundError(
                _missing_bars_message(
                    layout,
                    symbol=symbol,
                    interval=args.source_interval,
                )
            )
        outputs = build_feature_artifacts(
            layout=layout,
            symbol=symbol,
            normalize_window=args.normalize_window,
        )
        if outputs is None:
            raise FileNotFoundError(
                f"Missing required canonical feature inputs for {symbol}. "
                f"Expected 1m/5m/15m bars under {layout.exchange_root() / 'silver'}."
            )
        for name, path in outputs.items():
            log.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
