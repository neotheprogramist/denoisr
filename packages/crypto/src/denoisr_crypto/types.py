"""Shared constants and path helpers for the execution domain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

DEFAULT_EXCHANGE: Final[str] = "binance"
DEFAULT_MARKET: Final[str] = "spot"
DEFAULT_SOURCE_INTERVAL: Final[str] = "1m"
DEFAULT_DERIVED_INTERVALS: Final[tuple[str, ...]] = ("5m", "15m")
DEFAULT_SYMBOLS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT")
DEFAULT_WINDOW_START: Final[date] = date(2025, 3, 1)
DEFAULT_WINDOW_END: Final[date] = date(2026, 2, 28)
DEFAULT_HORIZONS_MINUTES: Final[tuple[int, ...]] = (15, 30, 60)
DEFAULT_PARTICIPATION_RATES: Final[tuple[float, ...]] = (0.01, 0.025, 0.05, 0.10)
DEFAULT_ROLLING_ZSCORE_WINDOW: Final[int] = 200
DEFAULT_TOKENIZER_CONTEXT_LENGTH: Final[int] = 512
DEFAULT_TOKENIZER_STRIDE: Final[int] = 64
DEFAULT_TOKENIZER_PATCH_SIZE: Final[int] = 16
DEFAULT_ENTRY_DECISION_INTERVAL: Final[str] = "15m"
DEFAULT_ENTRY_HORIZON_HOURS: Final[int] = 48
DEFAULT_ENTRY_SIGNAL_RATE: Final[float] = 0.05
DEFAULT_ENTRY_MIN_GAIN: Final[float] = 0.03


def parse_symbol_list(raw: str) -> tuple[str, ...]:
    symbols = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not symbols:
        raise ValueError("Expected at least one symbol")
    return symbols


def parse_interval_minutes(interval: str) -> int:
    if not interval.endswith("m"):
        raise ValueError(f"Unsupported interval: {interval!r}")
    minutes = int(interval[:-1])
    if minutes <= 0:
        raise ValueError(f"Interval must be > 0 minutes, got {interval!r}")
    return minutes


def month_key(month_start: date) -> str:
    return f"{month_start.year:04d}-{month_start.month:02d}"


def iter_month_starts(start: date, end: date) -> tuple[date, ...]:
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    months: list[date] = []
    while current <= last:
        months.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return tuple(months)


@dataclass(frozen=True)
class FixedWindow:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"Window end must be >= start, got {self.start} -> {self.end}"
            )

    @property
    def slug(self) -> str:
        return f"{self.start.isoformat()}_{self.end.isoformat()}"


@dataclass(frozen=True)
class DatasetSpec:
    exchange: str
    market: str
    symbols: tuple[str, ...]
    interval: str
    window: FixedWindow


@dataclass(frozen=True)
class PartitionSpec:
    interval: str
    year: int
    month: int
    symbol: str

    @property
    def month_key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


@dataclass(frozen=True)
class StorageLayout:
    root: Path
    exchange: str = DEFAULT_EXCHANGE
    market: str = DEFAULT_MARKET

    def exchange_root(self) -> Path:
        return self.root / "execution" / self.exchange

    def bronze_archive_path(self, symbol: str, interval: str, month_start: date) -> Path:
        return (
            self.exchange_root()
            / "bronze"
            / f"market={self.market}"
            / "dataset=klines"
            / f"symbol={symbol}"
            / f"interval={interval}"
            / f"dt={month_key(month_start)}"
            / "source.zip"
        )

    def bronze_checksum_path(self, symbol: str, interval: str, month_start: date) -> Path:
        return self.bronze_archive_path(symbol, interval, month_start).with_suffix(".CHECKSUM")

    def bronze_metadata_path(self, symbol: str, interval: str, month_start: date) -> Path:
        return self.bronze_archive_path(symbol, interval, month_start).with_suffix(".json")

    def silver_dataset_dir(self, interval: str) -> Path:
        return (
            self.exchange_root()
            / "silver"
            / f"market={self.market}"
            / "dataset=bars"
            / f"interval={interval}"
        )

    def silver_partition_path(
        self,
        symbol: str,
        interval: str,
        *,
        year: int,
        month: int,
        part: str = "part-0000.parquet",
    ) -> Path:
        return (
            self.silver_dataset_dir(interval)
            / f"year={year:04d}"
            / f"month={month:02d}"
            / f"symbol={symbol}"
            / part
        )

    def silver_glob(self, interval: str) -> str:
        return str(self.silver_dataset_dir(interval) / "year=*" / "month=*" / "symbol=*" / "*.parquet")

    def gold_root(self) -> Path:
        return self.exchange_root() / "gold"

    def feature_root(self) -> Path:
        return self.gold_root() / "features"

    def feature_symbol_dir(self, symbol: str) -> Path:
        return self.feature_root() / f"symbol={symbol}"

    def feature_manifest_path(self, symbol: str) -> Path:
        return self.feature_symbol_dir(symbol) / "feature_manifest.json"

    def dataset_manifest_dir(self) -> Path:
        return self.gold_root() / "catalog" / "datasets"

    def dataset_manifest_path(self, dataset_name: str) -> Path:
        return self.dataset_manifest_dir() / f"{dataset_name}.json"

    def validation_dir(self) -> Path:
        return self.gold_root() / "catalog" / "validation"

    def validation_path(self, report_name: str) -> Path:
        return self.validation_dir() / f"{report_name}.json"

    def tokenizer_root(self) -> Path:
        return self.gold_root() / "tokenizer"

    def tokenizer_corpus_dir(self) -> Path:
        return self.tokenizer_root() / "corpus"

    def tokenizer_corpus_manifest_path(self) -> Path:
        return self.tokenizer_corpus_dir() / "manifest.json"

    def tokenizer_corpus_path(self, symbol: str, split: str) -> Path:
        return self.tokenizer_corpus_dir() / f"symbol={symbol}" / f"{split}_sequences.safetensors"

    def tokenizer_sequence_index_path(self, symbol: str, split: str) -> Path:
        return self.tokenizer_corpus_dir() / f"symbol={symbol}" / f"{split}_index.parquet"

    def tokenizer_metadata_path(self, symbol: str) -> Path:
        return self.tokenizer_corpus_dir() / f"symbol={symbol}" / "metadata.json"

    def tokenizer_model_dir(self) -> Path:
        return self.tokenizer_root() / "models"

    def tokenizer_checkpoint_path(self, run_name: str = "fsq_tokenizer") -> Path:
        return self.tokenizer_model_dir() / f"{run_name}.pt"

    def tokenizer_metrics_path(self, run_name: str = "fsq_tokenizer") -> Path:
        return self.tokenizer_model_dir() / f"{run_name}_metrics.json"

    def tokenizer_export_dir(self, run_name: str = "fsq_tokenizer") -> Path:
        return self.tokenizer_root() / "exports" / f"model={run_name}"

    def tokenizer_export_path(self, run_name: str, symbol: str, split: str) -> Path:
        return self.tokenizer_export_dir(run_name) / f"symbol={symbol}" / f"{split}_tokens.parquet"

    def reports_dir(self) -> Path:
        return self.gold_root() / "reports"

    def report_symbol_dir(self, symbol: str) -> Path:
        return self.reports_dir() / f"symbol={symbol}"

    def training_dir(self) -> Path:
        return self.gold_root() / "training"

    def training_baseline_dir(self) -> Path:
        return self.training_dir() / "baseline"

    def entry_label_root(self) -> Path:
        return self.training_dir() / "labels"

    def entry_label_dir(self, symbol: str) -> Path:
        return self.entry_label_root() / f"symbol={symbol}"

    def entry_label_path(self, symbol: str, decision_interval: str) -> Path:
        return self.entry_label_dir(symbol) / f"entry_quality_labels_{decision_interval}.parquet"

    def entry_label_manifest_path(self, symbol: str, decision_interval: str) -> Path:
        return self.entry_label_dir(symbol) / f"entry_quality_labels_{decision_interval}.json"

    def entry_dataset_root(self) -> Path:
        return self.training_dir() / "datasets"

    def entry_dataset_dir(self, symbol: str) -> Path:
        return self.entry_dataset_root() / f"symbol={symbol}"

    def entry_dataset_path(self, symbol: str, decision_interval: str) -> Path:
        return self.entry_dataset_dir(symbol) / f"entry_quality_dataset_{decision_interval}.parquet"

    def entry_dataset_manifest_path(self, symbol: str, decision_interval: str) -> Path:
        return self.entry_dataset_dir(symbol) / f"entry_quality_dataset_{decision_interval}.json"

    def entry_model_dir(self) -> Path:
        return self.training_dir() / "entry_quality"

    def entry_model_checkpoint_path(self, run_name: str = "entry_quality_model") -> Path:
        return self.entry_model_dir() / f"{run_name}.pt"

    def entry_model_metrics_path(self, run_name: str = "entry_quality_model") -> Path:
        return self.entry_model_dir() / f"{run_name}_metrics.json"

    def simulator_dir(self, symbol: str) -> Path:
        return self.gold_root() / "simulator" / f"symbol={symbol}"

    def feature_path(self, symbol: str, name: str) -> Path:
        return self.feature_symbol_dir(symbol) / name

    def manifest_dir(self) -> Path:
        return self.exchange_root() / "manifests" / "collection_runs"

    def manifest_path(self, run_label: str) -> Path:
        return self.manifest_dir() / f"{run_label}.json"
