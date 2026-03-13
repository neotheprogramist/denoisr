"""Tabular schemas for Binance kline normalization."""

from __future__ import annotations

import polars as pl

KLINE_RAW_COLUMNS: tuple[str, ...] = (
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)

SILVER_BAR_COLUMNS: tuple[str, ...] = (
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

FLOAT_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)

INT_COLUMNS: tuple[str, ...] = ("open_time_ms", "close_time_ms", "trade_count")


def silver_bar_schema() -> dict[str, pl.DataType]:
    return {
        "exchange": pl.String,
        "market": pl.String,
        "symbol": pl.String,
        "interval": pl.String,
        "open_time": pl.Datetime("us"),
        "close_time": pl.Datetime("us"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
        "trade_count": pl.Int64,
        "taker_buy_base_volume": pl.Float64,
        "taker_buy_quote_volume": pl.Float64,
    }
