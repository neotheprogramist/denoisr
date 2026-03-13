"""Collect Binance spot kline archives and normalize 1m bars."""

from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path

from denoisr_crypto.data.collectors.binance import collect_binance_klines
from denoisr_crypto.data.transforms.klines import materialize_binance_klines
from denoisr_crypto.types import (
    DEFAULT_SOURCE_INTERVAL,
    DEFAULT_SYMBOLS,
    DEFAULT_WINDOW_END,
    DEFAULT_WINDOW_START,
    FixedWindow,
    StorageLayout,
    parse_symbol_list,
)
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


def _parse_date(raw: str) -> date:
    return date.fromisoformat(raw)


@graceful_main("denoisr-crypto-collect-binance", logger=log)
def main() -> None:
    parser = build_parser("Collect Binance spot archives and normalize 1m bars")
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
        "--interval",
        type=str,
        env_var="DENOISR_EXECUTION_INTERVAL",
        default=DEFAULT_SOURCE_INTERVAL,
        help="Source interval to collect",
    )
    add_env_argument(
        parser,
        "--start",
        type=_parse_date,
        env_var="DENOISR_EXECUTION_START",
        default=DEFAULT_WINDOW_START,
        help="Fixed collection window start date",
    )
    add_env_argument(
        parser,
        "--end",
        type=_parse_date,
        env_var="DENOISR_EXECUTION_END",
        default=DEFAULT_WINDOW_END,
        help="Fixed collection window end date",
    )
    add_env_argument(
        parser,
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        env_var="DENOISR_EXECUTION_SKIP_EXISTING",
        default=True,
        help="Skip already-downloaded monthly archives",
    )
    add_env_argument(
        parser,
        "--verify-checksums",
        action=argparse.BooleanOptionalAction,
        env_var="DENOISR_EXECUTION_VERIFY_CHECKSUMS",
        default=True,
        help="Verify official Binance SHA256 checksums before accepting archives",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    layout = StorageLayout(Path(args.storage_root))
    window = FixedWindow(start=args.start, end=args.end)
    manifest_path = collect_binance_klines(
        layout=layout,
        symbols=args.symbols,
        interval=args.interval,
        window=window,
        skip_existing=args.skip_existing,
        verify_checksums=args.verify_checksums,
    )
    artifacts = materialize_binance_klines(
        layout=layout,
        symbols=args.symbols,
        interval=args.interval,
        window=window,
        source_manifest_path=manifest_path,
    )
    log.info("Collection manifest: %s", manifest_path)
    for symbol, item in artifacts.items():
        log.info("%s partition count -> %d", symbol, len(item.partition_paths))
        log.info("%s dataset manifest -> %s", symbol, item.dataset_manifest_path)
        log.info("%s validation report -> %s", symbol, item.validation_report_path)


if __name__ == "__main__":
    main()
