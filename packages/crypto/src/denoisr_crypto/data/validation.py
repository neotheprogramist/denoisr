"""Validation helpers and JSON report generation for execution datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path
from typing import Any

import polars as pl

from denoisr_crypto.types import StorageLayout, parse_interval_minutes


@dataclass(frozen=True)
class ValidationArtifacts:
    report_name: str
    report_path: Path
    report: dict[str, Any]


def structural_invariant_report(frame: pl.DataFrame) -> dict[str, Any]:
    duplicate_count = int(
        frame.select(pl.struct("symbol", "interval", "open_time").is_duplicated().sum()).item()
    )
    violations = {
        "high_below_body": int(
            frame.filter(
                pl.col("high") < pl.max_horizontal("open", "close")
            ).height
        ),
        "low_above_body": int(
            frame.filter(
                pl.col("low") > pl.min_horizontal("open", "close")
            ).height
        ),
        "negative_volume": int(frame.filter(pl.col("volume") < 0).height),
        "negative_quote_volume": int(frame.filter(pl.col("quote_volume") < 0).height),
        "negative_trade_count": int(frame.filter(pl.col("trade_count") < 0).height),
        "duplicate_timestamps": duplicate_count,
    }
    return {
        "row_count": frame.height,
        "valid": all(count == 0 for count in violations.values()),
        "violations": violations,
    }


def continuity_report(frame: pl.DataFrame, *, interval: str) -> dict[str, Any]:
    expected_gap = timedelta(minutes=parse_interval_minutes(interval))
    gap_frame = frame.sort("open_time").with_columns(
        pl.col("open_time").diff().alias("gap")
    ).filter(pl.col("gap").is_not_null())
    mismatches = gap_frame.filter(pl.col("gap") != pl.lit(expected_gap))
    return {
        "interval": interval,
        "expected_gap_seconds": expected_gap.total_seconds(),
        "valid": mismatches.is_empty(),
        "gap_mismatch_count": mismatches.height,
        "examples": mismatches.select("open_time", "gap").head(5).to_dicts(),
    }


def aggregation_report(
    base_1m: pl.DataFrame,
    derived: pl.DataFrame,
    *,
    interval: str,
) -> dict[str, Any]:
    minutes = parse_interval_minutes(interval)
    expected = (
        base_1m.sort("open_time")
        .group_by_dynamic(
            "open_time",
            every=interval,
            period=interval,
            label="left",
            closed="left",
            start_by="window",
        )
        .agg(
            pl.col("close").last().alias("expected_close"),
            pl.col("high").max().alias("expected_high"),
            pl.col("low").min().alias("expected_low"),
            pl.col("volume").sum().alias("expected_volume"),
            pl.col("quote_volume").sum().alias("expected_quote_volume"),
            pl.col("trade_count").sum().alias("expected_trade_count"),
            pl.col("taker_buy_base_volume").sum().alias("expected_taker_buy_base_volume"),
            pl.col("taker_buy_quote_volume").sum().alias("expected_taker_buy_quote_volume"),
            pl.len().alias("bar_count"),
        )
        .filter(pl.col("bar_count") == minutes)
        .drop("bar_count")
    )
    joined = expected.join(
        derived.select(
            "open_time",
            "close",
            "high",
            "low",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ),
        on="open_time",
        how="inner",
    )
    mismatches = joined.filter(
        (pl.col("expected_close") != pl.col("close"))
        | (pl.col("expected_high") != pl.col("high"))
        | (pl.col("expected_low") != pl.col("low"))
        | (pl.col("expected_volume") != pl.col("volume"))
        | (pl.col("expected_quote_volume") != pl.col("quote_volume"))
        | (pl.col("expected_trade_count") != pl.col("trade_count"))
        | (pl.col("expected_taker_buy_base_volume") != pl.col("taker_buy_base_volume"))
        | (pl.col("expected_taker_buy_quote_volume") != pl.col("taker_buy_quote_volume"))
    )
    return {
        "interval": interval,
        "valid": mismatches.is_empty(),
        "mismatch_count": mismatches.height,
        "examples": mismatches.head(5).to_dicts(),
    }


def write_validation_report(
    *,
    layout: StorageLayout,
    report_name: str,
    report: dict[str, Any],
) -> ValidationArtifacts:
    path = layout.validation_path(report_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return ValidationArtifacts(report_name=report_name, report_path=path, report=report)
