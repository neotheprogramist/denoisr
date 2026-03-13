"""DuckDB-backed catalog helpers for execution datasets."""

from __future__ import annotations

from contextlib import closing
import glob
from pathlib import Path

import duckdb
import polars as pl

from denoisr_crypto.types import StorageLayout


def _matches(pattern: str) -> bool:
    return bool(glob.glob(pattern))


def _sql_string(path: str) -> str:
    return "'" + path.replace("'", "''") + "'"


def open_catalog(layout: StorageLayout) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(database=":memory:")
    bars_1m = layout.silver_glob("1m")
    bars_5m = layout.silver_glob("5m")
    bars_15m = layout.silver_glob("15m")
    features = str(layout.feature_root() / "symbol=*" / "features_multi_interval.parquet")
    tokenizer_exports = str(layout.tokenizer_root() / "exports" / "model=*" / "symbol=*" / "*.parquet")

    if _matches(bars_1m):
        connection.execute(
            f"CREATE VIEW bars_1m AS SELECT * FROM read_parquet({_sql_string(bars_1m)}, hive_partitioning=1)"
        )
    if _matches(bars_5m):
        connection.execute(
            f"CREATE VIEW bars_5m AS SELECT * FROM read_parquet({_sql_string(bars_5m)}, hive_partitioning=1)"
        )
    if _matches(bars_15m):
        connection.execute(
            f"CREATE VIEW bars_15m AS SELECT * FROM read_parquet({_sql_string(bars_15m)}, hive_partitioning=1)"
        )
    if _matches(features):
        connection.execute(
            f"CREATE VIEW features_multi_timeframe AS SELECT * FROM read_parquet({_sql_string(features)})"
        )
    if _matches(tokenizer_exports):
        connection.execute(
            f"CREATE VIEW tokenizer_exports AS SELECT * FROM read_parquet({_sql_string(tokenizer_exports)}, hive_partitioning=1)"
        )
    return connection


def query_catalog(layout: StorageLayout, sql: str) -> pl.DataFrame:
    with closing(open_catalog(layout)) as connection:
        return pl.DataFrame(connection.execute(sql).fetchnumpy())


def load_partitioned_bars(layout: StorageLayout, symbol: str, interval: str) -> pl.DataFrame | None:
    if not _matches(layout.silver_glob(interval)):
        return None
    with closing(open_catalog(layout)) as connection:
        view_name = f"bars_{interval}"
        frame = pl.DataFrame(connection.execute(
            f"""
            SELECT exchange, market, symbol, interval, open_time, close_time, open, high, low, close,
                   volume, quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume
            FROM {view_name}
            WHERE symbol = ?
            ORDER BY open_time
            """,
            [symbol],
        ).fetchnumpy())
    return None if frame.is_empty() else frame
