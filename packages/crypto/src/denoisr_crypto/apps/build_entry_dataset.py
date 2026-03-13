"""Build forward entry-quality labels and supervised datasets."""

from __future__ import annotations

import logging
from pathlib import Path

from denoisr_crypto.labels import EntryLabelConfig, build_entry_quality_dataset
from denoisr_crypto.types import (
    DEFAULT_ENTRY_DECISION_INTERVAL,
    DEFAULT_ENTRY_HORIZON_HOURS,
    DEFAULT_ENTRY_MIN_GAIN,
    DEFAULT_SYMBOLS,
    StorageLayout,
    parse_symbol_list,
)
from denoisr_common.interrupts import graceful_main
from denoisr_common.runtime import add_env_argument, build_parser, configure_logging

log = logging.getLogger(__name__)


@graceful_main("denoisr-crypto-build-entry-dataset", logger=log)
def main() -> None:
    parser = build_parser("Build forward entry-quality labels and training datasets")
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
        help="Decision interval used for forward entry labels",
    )
    add_env_argument(
        parser,
        "--horizon-hours",
        type=int,
        env_var="DENOISR_ENTRY_HORIZON_HOURS",
        default=DEFAULT_ENTRY_HORIZON_HOURS,
        help="Forward horizon in hours for entry-quality labels",
    )
    add_env_argument(
        parser,
        "--r-min",
        type=float,
        env_var="DENOISR_ENTRY_R_MIN",
        default=DEFAULT_ENTRY_MIN_GAIN,
        help="Minimum absolute upside threshold used in labels",
    )
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)
    artifacts = build_entry_quality_dataset(
        layout=StorageLayout(Path(args.storage_root)),
        symbols=args.symbols,
        config=EntryLabelConfig(
            decision_interval=args.decision_interval,
            horizon_hours=args.horizon_hours,
            r_min=args.r_min,
        ),
    )
    if artifacts is None:
        raise FileNotFoundError(
            "Missing canonical features or decision-interval bars. "
            "Run denoisr-crypto-build-features before building the entry dataset."
        )
    for symbol, path in sorted(artifacts.dataset_paths.items()):
        log.info("%s_entry_dataset -> %s", symbol, path)
        log.info("%s_entry_manifest -> %s", symbol, artifacts.manifest_paths[symbol])


if __name__ == "__main__":
    main()
