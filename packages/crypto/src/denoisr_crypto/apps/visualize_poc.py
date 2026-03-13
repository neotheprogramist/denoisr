"""Generate HTML reports for execution training artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.types import (
    DEFAULT_SYMBOLS,
    StorageLayout,
    parse_symbol_list,
)
from denoisr_crypto.visualization import build_visualization_reports
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


@graceful_main("denoisr-crypto-visualize-poc", logger=log)
def main() -> None:
    parser = build_parser("Generate HTML reports for execution training data")
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
        "--max-points",
        type=int,
        env_var="DENOISR_EXECUTION_VIS_MAX_POINTS",
        default=720,
        help="Maximum points to render per line chart after downsampling",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    artifacts = build_visualization_reports(
        layout=StorageLayout(Path(args.storage_root)),
        symbols=args.symbols,
        max_points=args.max_points,
    )
    if artifacts is None:
        raise FileNotFoundError(
            "Missing canonical feature datasets. "
            "Run denoisr-crypto-build-features before generating reports."
        )
    log.info("combined_report -> %s", artifacts.combined_report_path)
    for symbol, path in sorted(artifacts.symbol_report_paths.items()):
        log.info("%s_report -> %s", symbol, path)


if __name__ == "__main__":
    main()
