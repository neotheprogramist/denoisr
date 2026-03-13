from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from safetensors.torch import load_file
import torch

from denoisr_crypto.data.catalog import query_catalog
from denoisr_crypto.data.validation import aggregation_report
from denoisr_crypto.evaluation.simulator import run_backtest
from denoisr_crypto.features.ohlcv import (
    build_feature_artifacts,
    ensure_derived_bars,
    load_bars,
)
from denoisr_crypto.labels import EntryLabelConfig, build_entry_quality_dataset
from denoisr_crypto.tokenization import (
    build_tokenizer_corpus,
    export_token_dataset,
    train_fsq_tokenizer,
)
from denoisr_crypto.training import LossConfig, compute_entry_loss, train_entry_quality_model
from denoisr_crypto.training.baseline import train_baseline
from denoisr_crypto.types import (
    DEFAULT_TOKENIZER_PATCH_SIZE,
    StorageLayout,
)
from denoisr_crypto.visualization import build_visualization_reports


def _write_sample_1m_bars(root: Path, *, symbol: str, rows: int = 600) -> StorageLayout:
    layout = StorageLayout(root)
    start = datetime(2025, 3, 1, 0, 0, 0)
    data: list[dict[str, object]] = []
    base_price = 100.0 if symbol == "BTCUSDT" else 50.0
    for i in range(rows):
        open_time = start + timedelta(minutes=i)
        close_time = open_time + timedelta(seconds=59)
        open_px = base_price + i * 0.01
        close_px = open_px + 0.02
        high_px = close_px + 0.03
        low_px = open_px - 0.03
        volume = 10.0 + (i % 20)
        quote_volume = volume * close_px
        data.append(
            {
                "exchange": "binance",
                "market": "spot",
                "symbol": symbol,
                "interval": "1m",
                "open_time": open_time,
                "close_time": close_time,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "volume": volume,
                "quote_volume": quote_volume,
                "trade_count": 100 + i,
                "taker_buy_base_volume": volume * 0.52,
                "taker_buy_quote_volume": quote_volume * 0.52,
            }
        )
    frame = pl.DataFrame(data)
    partition_path = layout.silver_partition_path(symbol, "1m", year=2025, month=3)
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(partition_path)
    return layout


def test_feature_builder_creates_multi_interval_artifacts(tmp_path: Path) -> None:
    layout = _write_sample_1m_bars(tmp_path, symbol="BTCUSDT")
    derived_outputs = ensure_derived_bars(layout=layout, symbol="BTCUSDT")
    assert derived_outputs is not None

    bars_5m = load_bars(layout, "BTCUSDT", "5m")
    bars_15m = load_bars(layout, "BTCUSDT", "15m")
    assert bars_5m is not None
    assert bars_15m is not None
    assert bars_5m.height == 120
    assert bars_15m.height == 40

    outputs = build_feature_artifacts(layout=layout, symbol="BTCUSDT")
    assert outputs is not None
    feature_frame = pl.read_parquet(outputs["features_multi_interval"])
    assert feature_frame.height > 0
    assert "close_1m" in feature_frame.columns
    assert "rolling_vol_15_5m" in feature_frame.columns
    assert "yang_zhang_vol_20_1m" in feature_frame.columns
    assert "rsi_14_1m" in feature_frame.columns
    assert "target_return_5m" in feature_frame.columns
    assert outputs["raw_multi_interval"].exists()
    assert outputs["feature_manifest"].exists()
    manifest = json.loads(outputs["feature_manifest"].read_text())
    assert "output_paths" in manifest
    assert "canonical_paths" not in manifest
    assert "compatibility_paths" not in manifest
    assert manifest["source_paths"]["bars_1m"].endswith("/dataset=bars/interval=1m")


def test_backtest_generates_strategy_summaries(tmp_path: Path) -> None:
    layout = _write_sample_1m_bars(tmp_path, symbol="BTCUSDT")
    bars = load_bars(layout, "BTCUSDT", "1m")
    assert bars is not None

    orders, fills, results, summary = run_backtest(
        bars,
        symbol="BTCUSDT",
        sampling_stride=120,
    )

    assert orders.height > 0
    assert fills.height > 0
    assert results.height > 0
    assert summary.height > 0
    assert set(summary["strategy"].to_list()) == {"immediate", "pov", "twap"}


def test_partitioned_load_and_catalog_query_work(tmp_path: Path) -> None:
    layout = _write_sample_1m_bars(tmp_path, symbol="BTCUSDT")
    loaded = load_bars(layout, "BTCUSDT", "1m")
    assert loaded is not None
    assert loaded.height == 600

    result = query_catalog(
        layout,
        "SELECT symbol, COUNT(*) AS row_count FROM bars_1m GROUP BY symbol",
    )
    assert result.height == 1
    assert result["row_count"][0] == 600


def test_baseline_training_emits_artifacts(tmp_path: Path) -> None:
    layout = _write_sample_1m_bars(tmp_path, symbol="BTCUSDT")
    _write_sample_1m_bars(tmp_path, symbol="ETHUSDT")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        assert ensure_derived_bars(layout=layout, symbol=symbol) is not None
        assert build_feature_artifacts(layout=layout, symbol=symbol) is not None

    artifacts = train_baseline(
        layout=layout,
        symbols=("BTCUSDT", "ETHUSDT"),
        epochs=1,
        batch_size=128,
        lr=1e-3,
    )

    assert artifacts.checkpoint_path.exists()
    assert artifacts.metrics_path.exists()


def test_visualization_reports_render_html(tmp_path: Path) -> None:
    layout = _write_sample_1m_bars(tmp_path, symbol="BTCUSDT")
    _write_sample_1m_bars(tmp_path, symbol="ETHUSDT")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        assert ensure_derived_bars(layout=layout, symbol=symbol) is not None
        assert build_feature_artifacts(layout=layout, symbol=symbol) is not None
    artifacts_train = train_baseline(
        layout=layout,
        symbols=("BTCUSDT", "ETHUSDT"),
        epochs=1,
        batch_size=128,
        lr=1e-3,
    )
    assert artifacts_train is not None

    artifacts = build_visualization_reports(
        layout=layout,
        symbols=("BTCUSDT", "ETHUSDT"),
        max_points=120,
    )
    assert artifacts is not None

    combined_html = artifacts.combined_report_path.read_text()
    symbol_html = artifacts.symbol_report_paths["BTCUSDT"].read_text()
    assert "Execution Training Data Report" in combined_html
    assert "BTCUSDT Training Data Report" in symbol_html


def test_aggregation_report_matches_derived_bars(tmp_path: Path) -> None:
    layout = _write_sample_1m_bars(tmp_path, symbol="BTCUSDT")
    assert ensure_derived_bars(layout=layout, symbol="BTCUSDT") is not None
    base = load_bars(layout, "BTCUSDT", "1m")
    derived = load_bars(layout, "BTCUSDT", "5m")
    assert base is not None
    assert derived is not None
    report = aggregation_report(base, derived, interval="5m")
    assert report["valid"] is True
    assert report["mismatch_count"] == 0


def test_tokenizer_corpus_training_and_export(tmp_path: Path) -> None:
    layout = _write_sample_1m_bars(tmp_path, symbol="BTCUSDT", rows=900)
    _write_sample_1m_bars(tmp_path, symbol="ETHUSDT", rows=900)
    corpus = build_tokenizer_corpus(
        layout=layout,
        symbols=("BTCUSDT", "ETHUSDT"),
        context_length=64,
        stride=32,
    )
    assert corpus is not None
    assert corpus.manifest_path.exists()
    train_tensor = load_file(str(layout.tokenizer_corpus_path("BTCUSDT", "train")))["inputs"]
    assert train_tensor.shape[1:] == (64, 6)

    artifacts = train_fsq_tokenizer(
        layout=layout,
        symbols=("BTCUSDT", "ETHUSDT"),
        epochs=1,
        batch_size=16,
        run_name="fsq_test",
    )
    assert artifacts is not None
    assert artifacts.checkpoint_path.exists()
    assert artifacts.metrics_path.exists()

    exported = export_token_dataset(
        layout=layout,
        symbols=("BTCUSDT", "ETHUSDT"),
        run_name="fsq_test",
    )
    assert exported is not None
    export_path = exported.export_paths[("BTCUSDT", "train")]
    export_frame = pl.read_parquet(export_path)
    assert export_frame.height > 0
    assert "tokens" in export_frame.columns
    metadata = json.loads(layout.tokenizer_metadata_path("BTCUSDT").read_text())
    assert export_frame.height == metadata["sequence_counts"]["train"]
    assert len(export_frame["tokens"][0]) == 64 // DEFAULT_TOKENIZER_PATCH_SIZE


def test_entry_dataset_builder_and_training_emit_artifacts(tmp_path: Path) -> None:
    layout = _write_sample_1m_bars(tmp_path, symbol="BTCUSDT", rows=6000)
    _write_sample_1m_bars(tmp_path, symbol="ETHUSDT", rows=6000)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        assert ensure_derived_bars(layout=layout, symbol=symbol) is not None
        assert build_feature_artifacts(layout=layout, symbol=symbol) is not None

    dataset_artifacts = build_entry_quality_dataset(
        layout=layout,
        symbols=("BTCUSDT", "ETHUSDT"),
        config=EntryLabelConfig(
            decision_interval="15m",
            horizon_hours=12,
            volatility_window_bars=32,
        ),
    )
    assert dataset_artifacts is not None
    dataset = pl.read_parquet(dataset_artifacts.dataset_paths["BTCUSDT"])
    assert dataset.height > 0
    assert "quality_ratio_disc" in dataset.columns
    assert "a_pain_15m" in dataset.columns
    assert "opportunity_flag" in dataset.columns

    train_artifacts = train_entry_quality_model(
        layout=layout,
        symbols=("BTCUSDT", "ETHUSDT"),
        decision_interval="15m",
        epochs=1,
        batch_size=64,
        run_name="entry_quality_test",
        loss_config=LossConfig(name="p6"),
    )
    assert train_artifacts is not None
    assert train_artifacts.checkpoint_path.exists()
    metrics = json.loads(train_artifacts.metrics_path.read_text())
    assert metrics["best_val_loss"] >= 0.0
    assert "threshold_sweep" in metrics["val"]
    assert "calibration_ece" in metrics["test"]


def test_entry_loss_registry_produces_finite_values() -> None:
    score = torch.tensor([0.2, 0.7, 0.9], dtype=torch.float32)
    batch = {
        "r_up": torch.tensor([0.01, 0.08, 0.12]),
        "r_dd": torch.tensor([0.02, 0.01, 0.03]),
        "f_dd": torch.tensor([0.1, 0.2, 0.3]),
        "t_max_hours": torch.tensor([2.0, 6.0, 12.0]),
        "regret_up": torch.tensor([0.03, 0.01, 0.02]),
        "a_pain_15m": torch.tensor([0.001, 0.002, 0.003]),
        "quality_ratio_disc": torch.tensor([0.5, 2.0, 3.5]),
        "opportunity_flag": torch.tensor([0.0, 1.0, 1.0]),
        "r_up_vol_scaled": torch.tensor([1.0, 2.0, 3.0]),
        "r_dd_vol_scaled": torch.tensor([0.5, 0.3, 0.4]),
        "r_min_scaled": torch.tensor([0.03, 0.03, 0.03]),
    }
    for name in ("p1", "p2", "p3", "p5", "p6"):
        artifacts = compute_entry_loss(
            score=score,
            temp=1.0,
            batch=batch,
            config=LossConfig(name=name),
            denoise_loss=torch.tensor(0.05),
        )
        assert torch.isfinite(artifacts.total)
        assert artifacts.components["loss"] == artifacts.components["loss"]


def test_internal_execution_functions_return_none_for_missing_expected_artifacts(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(tmp_path)
    assert load_bars(layout, "BTCUSDT", "1m") is None
    assert ensure_derived_bars(layout=layout, symbol="BTCUSDT") is None
    assert build_feature_artifacts(layout=layout, symbol="BTCUSDT") is None
    assert build_entry_quality_dataset(
        layout=layout,
        symbols=("BTCUSDT",),
        config=EntryLabelConfig(decision_interval="15m", horizon_hours=12),
    ) is None
    assert build_tokenizer_corpus(layout=layout, symbols=("BTCUSDT",), context_length=64, stride=32) is None
    assert train_fsq_tokenizer(layout=layout, symbols=("BTCUSDT",), epochs=1, run_name="missing") is None
    assert export_token_dataset(layout=layout, symbols=("BTCUSDT",), run_name="missing") is None
    assert train_entry_quality_model(
        layout=layout,
        symbols=("BTCUSDT",),
        decision_interval="15m",
        epochs=1,
        run_name="missing",
    ) is None
