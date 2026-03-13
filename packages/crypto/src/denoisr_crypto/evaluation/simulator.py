"""Bars-only execution simulator and simple schedule baselines."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path

import polars as pl

from denoisr_crypto.types import (
    DEFAULT_HORIZONS_MINUTES,
    DEFAULT_PARTICIPATION_RATES,
    StorageLayout,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarRecord:
    open_time: object
    open: float
    high: float
    low: float
    close: float
    volume: float


def generate_parent_orders(
    frame: pl.DataFrame,
    *,
    symbol: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS_MINUTES,
    participation_rates: tuple[float, ...] = DEFAULT_PARTICIPATION_RATES,
    sampling_stride: int = 720,
    history_window: int = 60,
) -> pl.DataFrame:
    records = frame.sort("open_time").to_dicts()
    max_horizon = max(horizons)
    orders: list[dict[str, object]] = []
    order_id = 0
    for index in range(history_window, len(records) - max_horizon, sampling_stride):
        history = records[index - history_window : index]
        expected_volume = sum(float(item["volume"]) for item in history) / history_window
        start_bar = records[index]
        for horizon in horizons:
            for side in ("buy", "sell"):
                for rate in participation_rates:
                    target_qty = expected_volume * horizon * rate
                    orders.append(
                        {
                            "order_id": order_id,
                            "symbol": symbol,
                            "side": side,
                            "start_index": index,
                            "open_time": start_bar["open_time"],
                            "arrival_price": float(start_bar["open"]),
                            "horizon_minutes": horizon,
                            "participation_rate": rate,
                            "target_qty": target_qty,
                        }
                    )
                    order_id += 1
    return pl.DataFrame(orders)


def _strategy_slices(
    strategy: str,
    *,
    target_qty: float,
    participation_rate: float,
    bars: list[dict[str, object]],
) -> list[float]:
    if strategy == "immediate":
        return [target_qty] + [0.0] * (len(bars) - 1)
    if strategy == "twap":
        slice_qty = target_qty / len(bars)
        return [slice_qty for _ in bars]
    if strategy == "pov":
        remaining = target_qty
        allocations: list[float] = []
        for bar in bars:
            bar_volume = max(float(bar["volume"]), 1e-9)
            child = min(remaining, participation_rate * bar_volume)
            allocations.append(child)
            remaining -= child
        if remaining > 0 and allocations:
            allocations[-1] += remaining
        return allocations
    raise ValueError(f"Unknown strategy: {strategy}")


def _fill_price(side: str, qty: float, bar: dict[str, object]) -> float:
    close = float(bar["close"])
    high = float(bar["high"])
    low = float(bar["low"])
    volume = max(float(bar["volume"]), 1e-9)
    participation = max(qty / volume, 0.0)
    base_price = (close + high) / 2.0 if side == "buy" else (close + low) / 2.0
    range_proxy = max(high - low, close * 0.0005)
    impact = range_proxy * math.sqrt(participation)
    return base_price + impact if side == "buy" else base_price - impact


def run_backtest(
    frame: pl.DataFrame,
    *,
    symbol: str,
    sampling_stride: int = 720,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    bars = frame.sort("open_time").to_dicts()
    orders = generate_parent_orders(
        frame,
        symbol=symbol,
        sampling_stride=sampling_stride,
    )
    strategies = ("immediate", "twap", "pov")
    fills: list[dict[str, object]] = []
    order_results: list[dict[str, object]] = []

    for order in orders.to_dicts():
        start_index = int(order["start_index"])
        horizon = int(order["horizon_minutes"])
        horizon_bars = bars[start_index : start_index + horizon]
        benchmark_twap = sum(float(bar["close"]) for bar in horizon_bars) / len(horizon_bars)
        target_qty = float(order["target_qty"])
        arrival_price = float(order["arrival_price"])
        side = str(order["side"])
        side_sign = 1.0 if side == "buy" else -1.0

        for strategy in strategies:
            allocations = _strategy_slices(
                strategy,
                target_qty=target_qty,
                participation_rate=float(order["participation_rate"]),
                bars=horizon_bars,
            )
            total_notional = 0.0
            filled_qty = 0.0
            total_participation = 0.0
            fill_count = 0
            for offset, (bar, qty) in enumerate(zip(horizon_bars, allocations, strict=True)):
                if qty <= 0:
                    continue
                price = _fill_price(side, qty, bar)
                participation = qty / max(float(bar["volume"]), 1e-9)
                fills.append(
                    {
                        "order_id": int(order["order_id"]),
                        "strategy": strategy,
                        "symbol": symbol,
                        "side": side,
                        "offset": offset,
                        "open_time": bar["open_time"],
                        "qty": qty,
                        "price": price,
                        "participation": participation,
                    }
                )
                total_notional += qty * price
                filled_qty += qty
                total_participation += participation
                fill_count += 1

            average_price = total_notional / max(filled_qty, 1e-9)
            shortfall_bps = side_sign * ((average_price - arrival_price) / arrival_price) * 1e4
            twap_deviation_bps = (
                side_sign * ((average_price - benchmark_twap) / benchmark_twap) * 1e4
            )
            order_results.append(
                {
                    "order_id": int(order["order_id"]),
                    "strategy": strategy,
                    "symbol": symbol,
                    "side": side,
                    "horizon_minutes": horizon,
                    "participation_rate": float(order["participation_rate"]),
                    "arrival_price": arrival_price,
                    "average_execution_price": average_price,
                    "benchmark_twap_price": benchmark_twap,
                    "target_qty": target_qty,
                    "filled_qty": filled_qty,
                    "completion_ratio": filled_qty / max(target_qty, 1e-9),
                    "implementation_shortfall_bps": shortfall_bps,
                    "twap_deviation_bps": twap_deviation_bps,
                    "average_participation": total_participation / max(fill_count, 1),
                }
            )

    fills_df = pl.DataFrame(fills)
    order_results_df = pl.DataFrame(order_results)
    summary_df = (
        order_results_df.group_by("symbol", "strategy", "horizon_minutes")
        .agg(
            pl.col("implementation_shortfall_bps").mean().alias("mean_shortfall_bps"),
            pl.col("implementation_shortfall_bps").median().alias("median_shortfall_bps"),
            pl.col("twap_deviation_bps").mean().alias("mean_twap_deviation_bps"),
            pl.col("completion_ratio").mean().alias("mean_completion_ratio"),
            pl.col("average_participation").mean().alias("mean_average_participation"),
            pl.len().alias("num_orders"),
        )
        .sort("symbol", "strategy", "horizon_minutes")
    )
    return orders, fills_df, order_results_df, summary_df


def write_backtest_outputs(
    *,
    layout: StorageLayout,
    symbol: str,
    orders: pl.DataFrame,
    fills: pl.DataFrame,
    order_results: pl.DataFrame,
    summary: pl.DataFrame,
) -> dict[str, Path]:
    simulator_dir = layout.simulator_dir(symbol)
    simulator_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "parent_orders": simulator_dir / "parent_orders.parquet",
        "fills": simulator_dir / "fills.parquet",
        "order_results": simulator_dir / "order_results.parquet",
        "summary": simulator_dir / "summary.parquet",
    }
    orders.write_parquet(outputs["parent_orders"])
    fills.write_parquet(outputs["fills"])
    order_results.write_parquet(outputs["order_results"])
    summary.write_parquet(outputs["summary"])
    return outputs

