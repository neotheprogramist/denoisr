"""Run the bars-only execution backtest proof of concept."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.evaluation.simulator import (
    run_backtest,
    write_backtest_outputs,
)
from denoisr_crypto.features.ohlcv import load_bars
from denoisr_crypto.types import (
    DEFAULT_SYMBOLS,
    StorageLayout,
    parse_symbol_list,
)
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


@graceful_main("denoisr-crypto-backtest-poc", logger=log)
def main() -> None:
    parser = build_parser("Run the bars-only execution backtest POC")
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
        "--sampling-stride",
        type=int,
        env_var="DENOISR_EXECUTION_BACKTEST_STRIDE",
        default=720,
        help="Stride in 1m bars between synthetic parent orders",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    layout = StorageLayout(Path(args.storage_root))
    for symbol in args.symbols:
        bars = load_bars(layout, symbol, "1m")
        if bars is None:
            raise FileNotFoundError(
                f"Missing canonical 1m bars for {symbol} under {layout.silver_dataset_dir('1m')}. "
                "Run denoisr-crypto-collect-binance first."
            )
        orders, fills, order_results, summary = run_backtest(
            bars,
            symbol=symbol,
            sampling_stride=args.sampling_stride,
        )
        outputs = write_backtest_outputs(
            layout=layout,
            symbol=symbol,
            orders=orders,
            fills=fills,
            order_results=order_results,
            summary=summary,
        )
        for name, path in outputs.items():
            log.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
