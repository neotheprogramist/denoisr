"""OHLCV feature engineering, canonical dataset writes, and multi-timeframe alignment."""

from __future__ import annotations

import json
import math
from datetime import timedelta
import logging
from pathlib import Path

import polars as pl

from denoisr_crypto.data.catalog import load_partitioned_bars
from denoisr_crypto.data.validation import (
    aggregation_report,
    continuity_report,
    structural_invariant_report,
    write_validation_report,
)
from denoisr_crypto.types import (
    DEFAULT_DERIVED_INTERVALS,
    DEFAULT_ROLLING_ZSCORE_WINDOW,
    StorageLayout,
    parse_interval_minutes,
)

log = logging.getLogger(__name__)
_KEY_COLUMNS = ("exchange", "market", "symbol", "open_time")
_RAW_BAR_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "close_time",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)


def _safe_log_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when((numerator > 0) & (denominator > 0))
        .then(numerator.log() - denominator.log())
        .otherwise(None)
    )


def load_bars(layout: StorageLayout, symbol: str, interval: str) -> pl.DataFrame | None:
    return load_partitioned_bars(layout, symbol, interval)


def derive_interval_bars(frame: pl.DataFrame, *, interval: str) -> pl.DataFrame:
    minutes = parse_interval_minutes(interval)
    derived = (
        frame.sort("open_time")
        .group_by_dynamic(
            "open_time",
            every=interval,
            period=interval,
            label="left",
            closed="left",
            start_by="window",
        )
        .agg(
            pl.col("exchange").first().alias("exchange"),
            pl.col("market").first().alias("market"),
            pl.col("symbol").first().alias("symbol"),
            pl.col("close_time").last().alias("close_time"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("trade_count").sum().alias("trade_count"),
            pl.col("taker_buy_base_volume").sum().alias("taker_buy_base_volume"),
            pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote_volume"),
            pl.len().alias("bar_count"),
        )
        .filter(pl.col("bar_count") == minutes)
        .drop("bar_count")
        .with_columns(pl.lit(interval).alias("interval"))
        .select(
            "exchange",
            "market",
            "symbol",
            "interval",
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        )
    )
    validate_bar_continuity(derived, interval=interval, symbol=derived["symbol"][0])
    return derived


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
    partitions = partitioned.partition_by(["year", "month"], as_dict=True, maintain_order=True)
    for key, partition in partitions.items():
        year, month = (key if isinstance(key, tuple) else (key, None))
        if month is None:
            raise ValueError(f"Unexpected partition key: {key!r}")
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


def _write_bar_manifest(
    *,
    layout: StorageLayout,
    frame: pl.DataFrame,
    symbol: str,
    interval: str,
    partition_paths: tuple[Path, ...],
) -> Path:
    manifest = {
        "dataset_name": f"bars_{interval}",
        "symbol": symbol,
        "interval": interval,
        "row_count": frame.height,
        "partition_paths": [str(path) for path in partition_paths],
        "min_open_time": str(frame["open_time"][0]),
        "max_open_time": str(frame["open_time"][-1]),
        "schema_version": 2,
    }
    path = layout.dataset_manifest_path(f"bars_{interval}_{symbol}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def ensure_derived_bars(
    *,
    layout: StorageLayout,
    symbol: str,
    source_interval: str = "1m",
    derived_intervals: tuple[str, ...] = DEFAULT_DERIVED_INTERVALS,
) -> dict[str, Path] | None:
    base = load_bars(layout, symbol, source_interval)
    if base is None:
        return None
    outputs: dict[str, Path] = {}
    for interval in derived_intervals:
        derived = derive_interval_bars(base, interval=interval)
        partition_paths = _write_partitioned_bars(
            layout,
            frame=derived,
            symbol=symbol,
            interval=interval,
        )
        manifest_path = _write_bar_manifest(
            layout=layout,
            frame=derived,
            symbol=symbol,
            interval=interval,
            partition_paths=partition_paths,
        )
        validation = {
            "dataset_name": f"bars_{interval}",
            "symbol": symbol,
            "interval": interval,
            "structural": structural_invariant_report(derived),
            "continuity": continuity_report(derived, interval=interval),
            "aggregation": aggregation_report(base, derived, interval=interval),
        }
        write_validation_report(
            layout=layout,
            report_name=f"bars_{interval}_{symbol}_validation",
            report=validation,
        )
        outputs[interval] = manifest_path
        log.info(
            "Wrote derived %s bars for %s under %s",
            interval,
            symbol,
            layout.silver_dataset_dir(interval),
        )
    return outputs


def _append_normalized_features(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
    window: int,
) -> pl.DataFrame:
    expressions = []
    for column in columns:
        expr = pl.col(column)
        expressions.append(
            pl.when(expr.rolling_std(window) > 0)
            .then((expr - expr.rolling_mean(window)) / expr.rolling_std(window))
            .otherwise(None)
            .alias(f"{column}_norm")
        )
    return frame.with_columns(*expressions)


def _rolling_rsi(close: pl.Expr, period: int) -> pl.Expr:
    delta = close.diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.rolling_mean(period)
    avg_loss = loss.rolling_mean(period)
    rs = pl.when(avg_loss > 0).then(avg_gain / avg_loss).otherwise(None)
    return (100.0 - (100.0 / (1.0 + rs))).alias(f"rsi_{period}")


def _macd_columns(close: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    ema_fast = close.ewm_mean(span=12, adjust=False)
    ema_slow = close.ewm_mean(span=26, adjust=False)
    macd_line = (ema_fast - ema_slow).alias("macd_line")
    signal_line = (ema_fast - ema_slow).ewm_mean(span=9, adjust=False).alias("macd_signal")
    return macd_line, signal_line


def _yang_zhang_columns(
    *,
    open_: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    windows: tuple[int, ...] = (20, 50),
) -> tuple[pl.Expr, ...]:
    prev_close = close.shift(1)
    log_open_prev = _safe_log_ratio(open_, prev_close)
    log_close_open = _safe_log_ratio(close, open_)
    log_high_open = _safe_log_ratio(high, open_)
    log_low_open = _safe_log_ratio(low, open_)
    rs = (
        log_high_open * (log_high_open - log_close_open)
        + log_low_open * (log_low_open - log_close_open)
    )
    expressions: list[pl.Expr] = []
    for window in windows:
        k = 0.34 / (1.34 + ((window + 1) / max(window - 1, 1)))
        yz = (
            (log_open_prev.pow(2)).rolling_mean(window)
            + k * (log_close_open.pow(2)).rolling_mean(window)
            + (1 - k) * rs.rolling_mean(window)
        ).sqrt().alias(f"yang_zhang_vol_{window}")
        expressions.append(yz)
    return tuple(expressions)


def build_interval_feature_frame(
    frame: pl.DataFrame,
    *,
    interval: str,
    normalize_window: int = DEFAULT_ROLLING_ZSCORE_WINDOW,
) -> pl.DataFrame:
    hour_angle = (
        (pl.col("open_time").dt.hour() + (pl.col("open_time").dt.minute() / 60.0))
        * (2.0 * math.pi / 24.0)
    )
    dow_angle = pl.col("open_time").dt.weekday() * (2.0 * math.pi / 7.0)
    close = pl.col("close")
    open_ = pl.col("open")
    high = pl.col("high")
    low = pl.col("low")
    volume = pl.col("volume")
    trade_count = pl.col("trade_count").cast(pl.Float64)
    taker_buy = pl.col("taker_buy_base_volume")
    taker_sell = pl.col("volume") - taker_buy
    macd_line, signal_line = _macd_columns(close)

    frame = frame.sort("open_time").with_columns(
        _safe_log_ratio(close, close.shift(1)).alias("log_return_1"),
        _safe_log_ratio(close, close.shift(5)).alias("log_return_5"),
        _safe_log_ratio(close, close.shift(20)).alias("log_return_20"),
        _safe_log_ratio(close, close.shift(60)).alias("momentum_1h"),
        _safe_log_ratio(close, close.shift(240)).alias("momentum_4h"),
        _safe_log_ratio(close, close.shift(1440)).alias("momentum_24h"),
        _safe_log_ratio(close, close.shift(1)).rolling_std(15).alias("rolling_vol_15"),
        _safe_log_ratio(close, close.shift(1)).rolling_std(60).alias("rolling_vol_60"),
        _safe_log_ratio(close, close.shift(1)).rolling_std(240).alias("rolling_vol_240"),
        _safe_log_ratio(close.shift(1), close.shift(2)).rolling_std(20).alias("realized_vol_20"),
        _safe_log_ratio(close.shift(1), close.shift(2)).rolling_std(50).alias("realized_vol_50"),
        _rolling_rsi(close, 14),
        macd_line,
        signal_line,
        (macd_line - signal_line).alias("macd_hist"),
    ).with_columns(
        ((close - close.rolling_mean(20)) / (close.rolling_std(20) * 2.0)).alias("bollinger_pos_20_2"),
        ((high - low) / close).alias("high_low_range_pct"),
        pl.when(volume > 0)
        .then(taker_buy / volume)
        .otherwise(None)
        .alias("taker_buy_ratio"),
        pl.when(taker_sell > 0)
        .then(taker_buy / taker_sell)
        .otherwise(None)
        .alias("taker_buy_sell_ratio"),
        pl.col("quote_volume").alias("dollar_volume"),
        ((volume - volume.rolling_mean(60)) / volume.rolling_std(60)).alias("volume_zscore_60"),
        ((trade_count - trade_count.rolling_mean(60)) / trade_count.rolling_std(60)).alias(
            "trade_count_zscore_60"
        ),
        ((taker_buy - taker_buy.rolling_mean(100)) / taker_buy.rolling_std(100)).alias(
            "taker_buy_zscore_100"
        ),
        hour_angle.sin().alias("hour_sin"),
        hour_angle.cos().alias("hour_cos"),
        dow_angle.sin().alias("dow_sin"),
        dow_angle.cos().alias("dow_cos"),
        pl.lit(interval).alias("feature_interval"),
        *_yang_zhang_columns(open_=open_, high=high, low=low, close=close),
    )
    return _append_normalized_features(
        frame,
        columns=(
            "log_return_1",
            "log_return_5",
            "log_return_20",
            "momentum_1h",
            "momentum_4h",
            "momentum_24h",
            "rolling_vol_15",
            "rolling_vol_60",
            "rolling_vol_240",
            "realized_vol_20",
            "realized_vol_50",
            "yang_zhang_vol_20",
            "yang_zhang_vol_50",
            "taker_buy_ratio",
            "taker_buy_sell_ratio",
            "volume_zscore_60",
            "trade_count_zscore_60",
            "taker_buy_zscore_100",
            "bollinger_pos_20_2",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "rsi_14",
        ),
        window=normalize_window,
    )


def _rename_non_keys(
    frame: pl.DataFrame,
    suffix: str,
    *,
    keep_all_keys: bool,
) -> pl.DataFrame:
    keep_columns = _KEY_COLUMNS if keep_all_keys else ("open_time",)
    rename_map = {
        column: f"{column}_{suffix}"
        for column in frame.columns
        if column not in keep_columns
    }
    return frame.select(*keep_columns, *rename_map.keys()).rename(rename_map)


def _shift_feature_availability(frame: pl.DataFrame, *, interval: str) -> pl.DataFrame:
    minutes = parse_interval_minutes(interval)
    if minutes == 1:
        return frame
    return frame.with_columns(
        (pl.col("open_time") + timedelta(minutes=minutes)).alias("open_time")
    )


def _build_targets(frame: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close_1m")
    forward_returns_5 = [
        _safe_log_ratio(close.shift(-i), close.shift(-(i - 1))) for i in range(1, 6)
    ]
    forward_returns_15 = [
        _safe_log_ratio(close.shift(-i), close.shift(-(i - 1))) for i in range(1, 16)
    ]
    return frame.with_columns(
        _safe_log_ratio(close.shift(-1), close).alias("target_return_1m"),
        _safe_log_ratio(close.shift(-5), close).alias("target_return_5m"),
        pl.sum_horizontal([(expr**2) for expr in forward_returns_5])
        .sqrt()
        .alias("target_vol_5m"),
        pl.sum_horizontal([(expr**2) for expr in forward_returns_15])
        .sqrt()
        .alias("target_vol_15m"),
    )


def _write_feature_outputs(
    *,
    output_dir: Path,
    outputs: dict[str, pl.DataFrame],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / f"{name}.parquet"
        for name in outputs
    }
    for name, frame in outputs.items():
        frame.write_parquet(paths[name])
    return paths


def build_feature_artifacts(
    *,
    layout: StorageLayout,
    symbol: str,
    normalize_window: int = DEFAULT_ROLLING_ZSCORE_WINDOW,
) -> dict[str, Path] | None:
    base_1m = load_bars(layout, symbol, "1m")
    base_5m = load_bars(layout, symbol, "5m")
    base_15m = load_bars(layout, symbol, "15m")
    if base_1m is None or base_5m is None or base_15m is None:
        return None

    raw_1m = _rename_non_keys(base_1m, "1m", keep_all_keys=True)
    raw_5m = _rename_non_keys(
        _shift_feature_availability(base_5m, interval="5m"),
        "5m",
        keep_all_keys=False,
    )
    raw_15m = _rename_non_keys(
        _shift_feature_availability(base_15m, interval="15m"),
        "15m",
        keep_all_keys=False,
    )
    raw_multi = raw_1m.join_asof(raw_5m, on="open_time").join_asof(raw_15m, on="open_time")

    feat_1m = _rename_non_keys(
        build_interval_feature_frame(base_1m, interval="1m", normalize_window=normalize_window),
        "1m",
        keep_all_keys=True,
    )
    feat_5m = _rename_non_keys(
        _shift_feature_availability(
            build_interval_feature_frame(base_5m, interval="5m", normalize_window=normalize_window),
            interval="5m",
        ),
        "5m",
        keep_all_keys=False,
    )
    feat_15m = _rename_non_keys(
        _shift_feature_availability(
            build_interval_feature_frame(base_15m, interval="15m", normalize_window=normalize_window),
            interval="15m",
        ),
        "15m",
        keep_all_keys=False,
    )
    feature_multi = feat_1m.join_asof(feat_5m, on="open_time").join_asof(feat_15m, on="open_time")
    feature_multi = _build_targets(feature_multi).drop_nulls(
        subset=[
            "target_return_1m",
            "target_return_5m",
            "target_vol_5m",
            "target_vol_15m",
        ]
    )
    raw_multi = raw_multi.drop_nulls()

    output_frames = {
        "features_1m": feat_1m,
        "features_5m": feat_5m,
        "features_15m": feat_15m,
        "raw_multi_interval": raw_multi,
        "features_multi_interval": feature_multi,
    }
    canonical_paths = _write_feature_outputs(
        output_dir=layout.feature_symbol_dir(symbol),
        outputs=output_frames,
    )
    feature_manifest = {
        "symbol": symbol,
        "normalize_window": normalize_window,
        "feature_columns": feature_multi.columns,
        "normalized_columns": [column for column in feature_multi.columns if column.endswith("_norm")],
        "source_paths": {
            "bars_1m": str(layout.silver_dataset_dir("1m")),
            "bars_5m": str(layout.silver_dataset_dir("5m")),
            "bars_15m": str(layout.silver_dataset_dir("15m")),
        },
        "output_paths": {name: str(path) for name, path in canonical_paths.items()},
        "schema_version": 2,
    }
    feature_manifest_path = layout.feature_manifest_path(symbol)
    feature_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2, sort_keys=True))
    outputs = dict(canonical_paths)
    outputs["feature_manifest"] = feature_manifest_path
    return outputs
