"""Normalization utilities for Binance spot kline archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from io import BytesIO
import json
import logging
from pathlib import Path
import zipfile

import polars as pl

from denoisr_crypto.data.schemas import (
    FLOAT_COLUMNS,
    INT_COLUMNS,
    KLINE_RAW_COLUMNS,
    SILVER_BAR_COLUMNS,
)
from denoisr_crypto.data.validation import (
    continuity_report,
    structural_invariant_report,
    write_validation_report,
)
from denoisr_crypto.types import FixedWindow, StorageLayout, iter_month_starts, parse_interval_minutes

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaterializationArtifacts:
    partition_paths: tuple[Path, ...]
    dataset_manifest_path: Path
    validation_report_path: Path


def _read_archive_csv(archive_path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV member in {archive_path}, got {members}")
        with archive.open(members[0]) as handle:
            return pl.read_csv(
                BytesIO(handle.read()),
                has_header=False,
                new_columns=list(KLINE_RAW_COLUMNS),
            )


def _infer_epoch_unit(raw: pl.DataFrame, column: str) -> str:
    value = raw.select(pl.col(column).max()).item()
    if value >= 10**15:
        return "us"
    if value >= 10**12:
        return "ms"
    raise ValueError(f"Unsupported epoch precision for {column}: {value}")


def normalize_archive(
    archive_path: Path,
    *,
    symbol: str,
    interval: str,
    window: FixedWindow,
    exchange: str,
    market: str,
) -> pl.DataFrame:
    raw = _read_archive_csv(archive_path)
    open_time_unit = _infer_epoch_unit(raw, "open_time_ms")
    close_time_unit = _infer_epoch_unit(raw, "close_time_ms")
    start_dt = datetime.combine(window.start, time.min)
    end_exclusive = datetime.combine(window.end + timedelta(days=1), time.min)

    frame = raw.with_columns(
        *(pl.col(name).cast(pl.Float64) for name in FLOAT_COLUMNS),
        *(pl.col(name).cast(pl.Int64) for name in INT_COLUMNS),
    ).with_columns(
        pl.from_epoch(pl.col("open_time_ms"), time_unit=open_time_unit).alias("open_time"),
        pl.from_epoch(pl.col("close_time_ms"), time_unit=close_time_unit).alias("close_time"),
        pl.lit(exchange).alias("exchange"),
        pl.lit(market).alias("market"),
        pl.lit(symbol).alias("symbol"),
        pl.lit(interval).alias("interval"),
    ).with_columns(
        pl.col("open_time").dt.cast_time_unit("us"),
        pl.col("close_time").dt.cast_time_unit("us"),
    ).filter(
        (pl.col("open_time") >= pl.lit(start_dt))
        & (pl.col("open_time") < pl.lit(end_exclusive))
    ).select(*SILVER_BAR_COLUMNS)

    return frame


def validate_bar_continuity(frame: pl.DataFrame, *, interval: str, symbol: str) -> None:
    if frame.is_empty():
        raise ValueError(f"No bars available for {symbol} {interval}")
    if frame.select(pl.col("open_time").is_duplicated().any()).item():
        raise ValueError(f"Duplicate bar timestamps detected for {symbol} {interval}")
    interval_gap = timedelta(minutes=parse_interval_minutes(interval))
    bad_gaps = (
        frame.select(pl.col("open_time").diff().alias("gap"))
        .filter(pl.col("gap").is_not_null() & (pl.col("gap") != pl.lit(interval_gap)))
        .height
    )
    if bad_gaps:
        raise ValueError(f"Gap/continuity violation detected for {symbol} {interval}")


def _write_partitioned_bars(
    layout: StorageLayout,
    *,
    frame: pl.DataFrame,
    symbol: str,
    interval: str,
) -> tuple[Path, ...]:
    partitioned = frame.with_columns(
        pl.col("open_time").dt.year().alias("year"),
        pl.col("open_time").dt.month().alias("month"),
    )
    outputs: list[Path] = []
    for (year, month), partition in partitioned.partition_by(
        ["year", "month"],
        as_dict=True,
        maintain_order=True,
    ).items():
        path = layout.silver_partition_path(
            symbol,
            interval,
            year=int(year),
            month=int(month),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        partition.drop("year", "month").write_parquet(path)
        outputs.append(path)
    return tuple(outputs)


def _write_dataset_manifest(
    *,
    layout: StorageLayout,
    dataset_name: str,
    frame: pl.DataFrame,
    symbol: str,
    interval: str,
    window: FixedWindow,
    partition_paths: tuple[Path, ...],
    source_manifest_path: Path | None,
) -> Path:
    manifest = {
        "dataset_name": dataset_name,
        "exchange": layout.exchange,
        "market": layout.market,
        "symbol": symbol,
        "interval": interval,
        "window": {
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
        },
        "row_count": frame.height,
        "min_open_time": str(frame["open_time"][0]),
        "max_open_time": str(frame["open_time"][-1]),
        "partition_paths": [str(path) for path in partition_paths],
        "source_manifest_path": str(source_manifest_path) if source_manifest_path else None,
        "schema_version": 2,
    }
    path = layout.dataset_manifest_path(dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def materialize_binance_klines(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    interval: str,
    window: FixedWindow,
    source_manifest_path: Path | None = None,
) -> dict[str, MaterializationArtifacts]:
    """Convert bronze archive zips into canonical silver parquet tables."""
    outputs: dict[str, MaterializationArtifacts] = {}
    for symbol in symbols:
        monthly_frames: list[pl.DataFrame] = []
        for month_start in iter_month_starts(window.start, window.end):
            archive_path = layout.bronze_archive_path(symbol, interval, month_start)
            if not archive_path.exists():
                raise FileNotFoundError(f"Missing bronze archive: {archive_path}")
            monthly_frames.append(
                normalize_archive(
                    archive_path,
                    symbol=symbol,
                    interval=interval,
                    window=window,
                    exchange=layout.exchange,
                    market=layout.market,
                )
            )
        frame = (
            pl.concat(monthly_frames, how="vertical")
            .sort("open_time")
            .unique(subset=["open_time"], keep="last", maintain_order=True)
        )
        validate_bar_continuity(frame, interval=interval, symbol=symbol)
        partition_paths = _write_partitioned_bars(layout, frame=frame, symbol=symbol, interval=interval)
        dataset_name = f"bars_{interval}_{symbol}_{window.slug}"
        dataset_manifest_path = _write_dataset_manifest(
            layout=layout,
            dataset_name=dataset_name,
            frame=frame,
            symbol=symbol,
            interval=interval,
            window=window,
            partition_paths=partition_paths,
            source_manifest_path=source_manifest_path,
        )
        validation = {
            "dataset_name": dataset_name,
            "symbol": symbol,
            "interval": interval,
            "structural": structural_invariant_report(frame),
            "continuity": continuity_report(frame, interval=interval),
        }
        validation_artifacts = write_validation_report(
            layout=layout,
            report_name=f"{dataset_name}_validation",
            report=validation,
        )
        outputs[symbol] = MaterializationArtifacts(
            partition_paths=partition_paths,
            dataset_manifest_path=dataset_manifest_path,
            validation_report_path=validation_artifacts.report_path,
        )
        log.info(
            "Wrote %s canonical partitions for %s under %s",
            interval,
            symbol,
            layout.silver_dataset_dir(interval),
        )
    return outputs
