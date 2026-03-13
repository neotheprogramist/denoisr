"""Tokenizer corpus generation for 1m OHLCV bars."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import polars as pl
from safetensors.torch import save_file
import torch

from denoisr_crypto.features.ohlcv import load_bars
from denoisr_crypto.types import (
    DEFAULT_TOKENIZER_CONTEXT_LENGTH,
    DEFAULT_TOKENIZER_STRIDE,
    StorageLayout,
)

_TOKEN_COLUMNS = (
    "open_rel",
    "high_rel",
    "low_rel",
    "close_rel",
    "log_volume",
    "log_quote_volume",
)


@dataclass(frozen=True)
class TokenizerCorpusArtifacts:
    manifest_path: Path
    metadata_paths: dict[str, Path]
    tensor_paths: dict[tuple[str, str], Path]


def _prepare_token_frame(frame: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close")
    open_ = pl.col("open")
    prev_close = close.shift(1)
    return (
        frame.sort("open_time")
        .with_columns(
            pl.when((prev_close > 0) & (open_ > 0))
            .then(open_.log() - prev_close.log())
            .otherwise(None)
            .alias("open_rel"),
            pl.when((open_ > 0) & (pl.col("high") > 0))
            .then(pl.col("high").log() - open_.log())
            .otherwise(None)
            .alias("high_rel"),
            pl.when((open_ > 0) & (pl.col("low") > 0))
            .then(pl.col("low").log() - open_.log())
            .otherwise(None)
            .alias("low_rel"),
            pl.when((open_ > 0) & (close > 0))
            .then(close.log() - open_.log())
            .otherwise(None)
            .alias("close_rel"),
            pl.col("volume").log1p().alias("log_volume"),
            pl.col("quote_volume").log1p().alias("log_quote_volume"),
        )
        .drop_nulls(subset=["open_rel", "high_rel", "low_rel", "close_rel"])
        .select("symbol", "open_time", *_TOKEN_COLUMNS)
    )


def _split_frame(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    n = frame.height
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return {
        "train": frame[:train_end],
        "val": frame[train_end:val_end],
        "test": frame[val_end:],
    }


def _window_tensors(
    frame: pl.DataFrame,
    *,
    context_length: int,
    stride: int,
) -> tuple[torch.Tensor, pl.DataFrame]:
    if frame.height < context_length:
        empty_tensor = torch.empty((0, context_length, len(_TOKEN_COLUMNS)), dtype=torch.float32)
        empty_index = pl.DataFrame(
            {
                "sequence_id": [],
                "start_open_time": [],
                "end_open_time": [],
            },
            schema={
                "sequence_id": pl.Int64,
                "start_open_time": pl.Datetime("us"),
                "end_open_time": pl.Datetime("us"),
            },
        )
        return empty_tensor, empty_index
    values = frame.select(_TOKEN_COLUMNS).to_numpy()
    open_times = frame["open_time"]
    windows: list[torch.Tensor] = []
    index_rows: list[dict[str, object]] = []
    sequence_id = 0
    for start in range(0, frame.height - context_length + 1, stride):
        end = start + context_length
        windows.append(torch.tensor(values[start:end], dtype=torch.float32))
        index_rows.append(
            {
                "sequence_id": sequence_id,
                "start_open_time": open_times[start],
                "end_open_time": open_times[end - 1],
            }
        )
        sequence_id += 1
    return torch.stack(windows), pl.DataFrame(index_rows)


def _global_train_stats(split_frames: dict[str, dict[str, pl.DataFrame]]) -> tuple[list[float], list[float]]:
    train_frames = [frames["train"].select(_TOKEN_COLUMNS) for frames in split_frames.values() if not frames["train"].is_empty()]
    if not train_frames:
        raise ValueError("No train rows available to build tokenizer corpus")
    combined = pl.concat(train_frames, how="vertical")
    means = [float(combined[column].mean()) for column in _TOKEN_COLUMNS]
    stds = [
        float(combined[column].std()) if combined[column].std() and combined[column].std() > 0 else 1.0
        for column in _TOKEN_COLUMNS
    ]
    return means, stds


def _normalize_frame(frame: pl.DataFrame, *, means: list[float], stds: list[float]) -> pl.DataFrame:
    expressions = []
    for idx, column in enumerate(_TOKEN_COLUMNS):
        expressions.append(((pl.col(column) - means[idx]) / stds[idx]).alias(column))
    return frame.with_columns(*expressions)


def build_tokenizer_corpus(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    context_length: int = DEFAULT_TOKENIZER_CONTEXT_LENGTH,
    stride: int = DEFAULT_TOKENIZER_STRIDE,
) -> TokenizerCorpusArtifacts | None:
    source_frames: dict[str, pl.DataFrame] = {}
    for symbol in symbols:
        bars = load_bars(layout, symbol, "1m")
        if bars is None:
            return None
        source_frames[symbol] = _prepare_token_frame(bars)
    split_frames = {symbol: _split_frame(frame) for symbol, frame in source_frames.items()}
    means, stds = _global_train_stats(split_frames)

    metadata_paths: dict[str, Path] = {}
    tensor_paths: dict[tuple[str, str], Path] = {}
    manifest = {
        "symbols": list(symbols),
        "feature_names": list(_TOKEN_COLUMNS),
        "context_length": context_length,
        "stride": stride,
        "means": means,
        "stds": stds,
        "schema_version": 1,
        "per_symbol": {},
    }
    for symbol, frames in split_frames.items():
        symbol_dir = layout.tokenizer_corpus_dir() / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        split_counts: dict[str, int] = {}
        row_counts = {split: frame.height for split, frame in frames.items()}
        for split, frame in frames.items():
            normalized = _normalize_frame(frame, means=means, stds=stds)
            tensor, index_frame = _window_tensors(
                normalized,
                context_length=context_length,
                stride=stride,
            )
            tensor_path = layout.tokenizer_corpus_path(symbol, split)
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"inputs": tensor}, str(tensor_path))
            index_path = layout.tokenizer_sequence_index_path(symbol, split)
            index_frame.write_parquet(index_path)
            tensor_paths[(symbol, split)] = tensor_path
            split_counts[split] = int(tensor.shape[0])
        metadata = {
            "symbol": symbol,
            "feature_names": list(_TOKEN_COLUMNS),
            "context_length": context_length,
            "stride": stride,
            "row_counts": row_counts,
            "sequence_counts": split_counts,
            "means": means,
            "stds": stds,
        }
        metadata_path = layout.tokenizer_metadata_path(symbol)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        metadata_paths[symbol] = metadata_path
        manifest["per_symbol"][symbol] = metadata

    manifest_path = layout.tokenizer_corpus_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return TokenizerCorpusArtifacts(
        manifest_path=manifest_path,
        metadata_paths=metadata_paths,
        tensor_paths=tensor_paths,
    )
