"""Validate canonical execution datasets and write a combined report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from denoisr_crypto.data.validation import (
    aggregation_report,
    continuity_report,
    structural_invariant_report,
    write_validation_report,
)
from denoisr_crypto.features.ohlcv import load_bars
from denoisr_crypto.types import DEFAULT_SYMBOLS, StorageLayout, parse_symbol_list
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


def _parse_intervals(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one interval")
    return values


@graceful_main("denoisr-crypto-validate-binance", logger=log)
def main() -> None:
    parser = build_parser("Validate canonical Binance execution datasets")
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
        "--intervals",
        type=_parse_intervals,
        env_var="DENOISR_EXECUTION_VALIDATE_INTERVALS",
        default=("1m", "5m", "15m"),
        help="Comma-separated intervals to validate",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    layout = StorageLayout(Path(args.storage_root))
    report: dict[str, object] = {"symbols": {}}
    for symbol in args.symbols:
        symbol_report: dict[str, object] = {}
        base_1m = load_bars(layout, symbol, "1m")
        if base_1m is None:
            raise FileNotFoundError(
                f"Missing canonical 1m bars for {symbol} under {layout.silver_dataset_dir('1m')}. "
                "Run denoisr-crypto-collect-binance first."
            )
        for interval in args.intervals:
            frame = load_bars(layout, symbol, interval)
            if frame is None:
                raise FileNotFoundError(
                    f"Missing canonical {interval} bars for {symbol} under {layout.silver_dataset_dir(interval)}."
                )
            interval_report = {
                "structural": structural_invariant_report(frame),
                "continuity": continuity_report(frame, interval=interval),
            }
            if interval != "1m":
                interval_report["aggregation"] = aggregation_report(base_1m, frame, interval=interval)
            symbol_report[interval] = interval_report
        report["symbols"][symbol] = symbol_report
    artifacts = write_validation_report(
        layout=layout,
        report_name="execution_dataset_validation",
        report=report,
    )
    log.info("validation_report -> %s", artifacts.report_path)


if __name__ == "__main__":
    main()
