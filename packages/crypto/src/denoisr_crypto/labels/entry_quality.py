"""Forward labels and supervised datasets for entry-quality modeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from denoisr_crypto.features.ohlcv import load_bars
from denoisr_crypto.types import (
    DEFAULT_ENTRY_DECISION_INTERVAL,
    DEFAULT_ENTRY_HORIZON_HOURS,
    DEFAULT_ENTRY_MIN_GAIN,
    StorageLayout,
    parse_interval_minutes,
)

ENTRY_LABEL_COLUMNS = (
    "entry_price",
    "p_max",
    "p_min_dd",
    "t_max_hours",
    "r_up",
    "r_dd",
    "f_dd",
    "a_pain_15m",
    "a_profit_15m",
    "f_pain_15m",
    "f_profit_15m",
    "regret_up",
    "r_up_disc",
    "quality_ratio_disc",
    "volatility_scale",
    "r_up_vol_scaled",
    "r_dd_vol_scaled",
    "r_min_scaled",
    "opportunity_flag",
)


@dataclass(frozen=True)
class EntryLabelConfig:
    decision_interval: str = DEFAULT_ENTRY_DECISION_INTERVAL
    horizon_hours: int = DEFAULT_ENTRY_HORIZON_HOURS
    gamma_time_discount: float = 0.98
    drawdown_lambda: float = 1.0
    epsilon: float = 1e-4
    r_min: float = DEFAULT_ENTRY_MIN_GAIN
    volatility_window_bars: int = 672
    volatility_floor: float = 1e-4
    volatility_threshold_scale: float = 2.0

    @property
    def decision_minutes(self) -> int:
        return parse_interval_minutes(self.decision_interval)

    @property
    def horizon_bars(self) -> int:
        return int((self.horizon_hours * 60) / self.decision_minutes)


@dataclass(frozen=True)
class EntryLabelArtifacts:
    label_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class EntryDatasetArtifacts:
    label_artifacts: dict[str, EntryLabelArtifacts]
    dataset_paths: dict[str, Path]
    manifest_paths: dict[str, Path]


def _decision_features(
    *,
    layout: StorageLayout,
    symbol: str,
    decision_interval: str,
) -> pl.DataFrame | None:
    feature_path = layout.feature_path(symbol, "features_multi_interval.parquet")
    if not feature_path.exists():
        return None
    feature_frame = pl.read_parquet(feature_path).sort("open_time")
    decision_bars = load_bars(layout, symbol, decision_interval)
    if decision_bars is None:
        return None
    return feature_frame.join(
        decision_bars.select("open_time"),
        on="open_time",
        how="inner",
    ).sort("open_time")


def _volatility_scale(close: np.ndarray, window: int, floor: float) -> np.ndarray:
    returns = np.full(close.shape[0], np.nan, dtype=np.float64)
    returns[1:] = np.log(close[1:] / close[:-1])
    scale = np.full(close.shape[0], np.nan, dtype=np.float64)
    if window <= 1:
        scale[1:] = np.maximum(np.abs(returns[1:]), floor)
        return scale
    for index in range(window, close.shape[0]):
        sample = returns[index - window + 1 : index + 1]
        scale[index] = max(float(np.nanstd(sample)), floor)
    return scale


def _window_view(values: np.ndarray, horizon_bars: int) -> np.ndarray:
    if values.shape[0] <= horizon_bars:
        return np.empty((0, horizon_bars), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(values[1:], horizon_bars)


def _label_frame(
    *,
    bars: pl.DataFrame,
    config: EntryLabelConfig,
) -> pl.DataFrame:
    close = bars["close"].to_numpy().astype(np.float64)
    open_times = bars["open_time"]
    windows = _window_view(close, config.horizon_bars)
    if windows.shape[0] == 0:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "open_time": pl.Datetime("us"),
                **{column: pl.Float64 for column in ENTRY_LABEL_COLUMNS[:-1]},
                "opportunity_flag": pl.Int8,
            }
        )

    entry = close[: windows.shape[0]]
    max_idx = windows.argmax(axis=1)
    max_vals = windows[np.arange(windows.shape[0]), max_idx]
    log_paths = np.log(windows / entry[:, None])
    pain = np.maximum(-log_paths, 0.0)
    profit = np.maximum(log_paths, 0.0)
    min_pre_peak = np.fromiter(
        (
            float(min(entry_price, np.min(window[: peak_index + 1])))
            for entry_price, window, peak_index in zip(entry, windows, max_idx, strict=True)
        ),
        dtype=np.float64,
        count=windows.shape[0],
    )
    drawdown_counts = np.fromiter(
        (
            int(np.sum(window[: peak_index + 1] < entry_price))
            for entry_price, window, peak_index in zip(entry, windows, max_idx, strict=True)
        ),
        dtype=np.int32,
        count=windows.shape[0],
    )
    t_max_hours = (max_idx + 1) * (config.decision_minutes / 60.0)
    r_up = (max_vals - entry) / entry
    r_dd = np.maximum((entry - min_pre_peak) / entry, 0.0)
    f_dd = drawdown_counts / config.horizon_bars
    r_up_disc = r_up * np.power(config.gamma_time_discount, t_max_hours)
    quality_ratio_disc = r_up_disc / (r_dd * (1.0 + config.drawdown_lambda * f_dd) + config.epsilon)
    ideal_upside = ((max_vals - min_pre_peak) / np.maximum(min_pre_peak, config.epsilon)) * config.gamma_time_discount
    regret_up = np.maximum(ideal_upside - r_up_disc, 0.0)
    a_pain = pain.mean(axis=1)
    a_profit = profit.mean(axis=1)
    f_pain = (log_paths < 0).mean(axis=1)
    f_profit = (log_paths > 0).mean(axis=1)
    volatility_scale = _volatility_scale(close, config.volatility_window_bars, config.volatility_floor)
    valid_volatility = volatility_scale[: windows.shape[0]]
    r_up_vol_scaled = r_up / np.maximum(valid_volatility, config.volatility_floor)
    r_dd_vol_scaled = r_dd / np.maximum(valid_volatility, config.volatility_floor)
    r_min_scaled = np.maximum(config.r_min, np.maximum(valid_volatility, config.volatility_floor) * config.volatility_threshold_scale)
    opportunity_flag = (
        (r_up >= r_min_scaled)
        & (quality_ratio_disc >= 1.0)
    ).astype(np.int8)

    frame = pl.DataFrame(
        {
            "symbol": bars["symbol"][: windows.shape[0]],
            "open_time": open_times[: windows.shape[0]],
            "entry_price": entry,
            "p_max": max_vals,
            "p_min_dd": min_pre_peak,
            "t_max_hours": t_max_hours,
            "r_up": r_up,
            "r_dd": r_dd,
            "f_dd": f_dd,
            "a_pain_15m": a_pain,
            "a_profit_15m": a_profit,
            "f_pain_15m": f_pain,
            "f_profit_15m": f_profit,
            "regret_up": regret_up,
            "r_up_disc": r_up_disc,
            "quality_ratio_disc": quality_ratio_disc,
            "volatility_scale": valid_volatility,
            "r_up_vol_scaled": r_up_vol_scaled,
            "r_dd_vol_scaled": r_dd_vol_scaled,
            "r_min_scaled": r_min_scaled,
            "opportunity_flag": opportunity_flag,
        }
    )
    return (
        frame
        .drop_nulls(subset=["volatility_scale"])
        .drop_nans(subset=["volatility_scale", "r_up_vol_scaled", "r_dd_vol_scaled", "r_min_scaled"])
    )


def _summary(frame: pl.DataFrame) -> dict[str, object]:
    if frame.is_empty():
        return {"row_count": 0}
    summary_columns = (
        "r_up",
        "r_dd",
        "f_dd",
        "quality_ratio_disc",
        "a_pain_15m",
        "volatility_scale",
    )
    return {
        "row_count": frame.height,
        "opportunity_rate": float(frame["opportunity_flag"].mean()),
        "columns": {
                    column: {
                "mean": float(frame[column].mean()),
                "p50": float(frame[column].quantile(0.5)),
                "p90": float(frame[column].quantile(0.9)),
                "p99": float(frame[column].quantile(0.99)),
            }
            for column in summary_columns
        },
    }


def build_entry_quality_labels(
    *,
    layout: StorageLayout,
    symbol: str,
    config: EntryLabelConfig,
) -> EntryLabelArtifacts | None:
    bars = load_bars(layout, symbol, config.decision_interval)
    if bars is None:
        return None
    labels = _label_frame(bars=bars.sort("open_time"), config=config)
    label_path = layout.entry_label_path(symbol, config.decision_interval)
    manifest_path = layout.entry_label_manifest_path(symbol, config.decision_interval)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    labels.write_parquet(label_path)
    manifest_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "config": asdict(config),
                "summary": _summary(labels),
                "schema_version": 1,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return EntryLabelArtifacts(label_path=label_path, manifest_path=manifest_path)


def _join_dataset(
    *,
    decision_features: pl.DataFrame,
    labels: pl.DataFrame,
) -> pl.DataFrame:
    return decision_features.join(labels, on=["symbol", "open_time"], how="inner").sort("open_time")


def build_entry_quality_dataset(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    config: EntryLabelConfig,
) -> EntryDatasetArtifacts | None:
    label_artifacts: dict[str, EntryLabelArtifacts] = {}
    dataset_paths: dict[str, Path] = {}
    manifest_paths: dict[str, Path] = {}
    for symbol in symbols:
        decision_features = _decision_features(
            layout=layout,
            symbol=symbol,
            decision_interval=config.decision_interval,
        )
        if decision_features is None:
            return None
        label_artifact = build_entry_quality_labels(layout=layout, symbol=symbol, config=config)
        if label_artifact is None:
            return None
        labels = pl.read_parquet(label_artifact.label_path)
        dataset = _join_dataset(decision_features=decision_features, labels=labels)
        dataset_path = layout.entry_dataset_path(symbol, config.decision_interval)
        manifest_path = layout.entry_dataset_manifest_path(symbol, config.decision_interval)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_parquet(dataset_path)
        manifest_path.write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "decision_interval": config.decision_interval,
                    "horizon_hours": config.horizon_hours,
                    "row_count": dataset.height,
                    "label_columns": list(ENTRY_LABEL_COLUMNS),
                    "feature_columns": [column for column in dataset.columns if column not in ENTRY_LABEL_COLUMNS],
                    "schema_version": 1,
                },
                indent=2,
                sort_keys=True,
            )
        )
        label_artifacts[symbol] = label_artifact
        dataset_paths[symbol] = dataset_path
        manifest_paths[symbol] = manifest_path
    return EntryDatasetArtifacts(
        label_artifacts=label_artifacts,
        dataset_paths=dataset_paths,
        manifest_paths=manifest_paths,
    )
