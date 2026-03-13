"""Baseline multi-task trainer for execution POC features."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from denoisr_crypto.types import StorageLayout

log = logging.getLogger(__name__)
_METADATA_COLUMNS = {"exchange", "market", "symbol", "open_time"}
_TARGET_COLUMNS = {"target_return_1m", "target_return_5m", "target_vol_5m", "target_vol_15m"}


class ExecutionBaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.return_head = nn.Linear(hidden_dim, 2)
        self.vol_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.trunk(x)
        return self.return_head(trunk), self.vol_head(trunk).squeeze(-1)


@dataclass(frozen=True)
class BaselineArtifacts:
    checkpoint_path: Path
    metrics_path: Path


def load_training_frame(layout: StorageLayout, symbols: tuple[str, ...]) -> pl.DataFrame | None:
    frames: list[pl.DataFrame] = []
    for symbol_index, symbol in enumerate(symbols):
        source_path = layout.feature_path(symbol, "features_multi_interval.parquet")
        if not source_path.exists():
            return None
        frame = pl.read_parquet(source_path).with_columns(
            pl.lit(symbol_index).cast(pl.Float64).alias("symbol_id")
        )
        frames.append(frame)
    return None if not frames else pl.concat(frames, how="vertical").sort("open_time")


def feature_columns(frame: pl.DataFrame) -> list[str]:
    schema = frame.schema
    return [
        column
        for column in frame.columns
        if column not in _METADATA_COLUMNS and column not in _TARGET_COLUMNS
        and schema[column] != pl.String
        and "time" not in column
    ]


def split_indices(row_count: int) -> tuple[int, int]:
    train_end = int(row_count * 0.7)
    val_end = int(row_count * 0.85)
    if train_end == 0 or val_end <= train_end:
        raise ValueError("Not enough rows to create train/val/test splits")
    return train_end, val_end


def train_baseline(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> BaselineArtifacts | None:
    frame = load_training_frame(layout, symbols)
    if frame is None:
        return None
    frame = frame.drop_nulls(subset=["target_return_5m", "target_vol_15m"])
    selected_feature_columns = feature_columns(frame)
    feature_frame = frame.select(selected_feature_columns).with_columns(pl.all().fill_null(0.0))
    features = torch.tensor(
        feature_frame.to_numpy(),
        dtype=torch.float32,
    )
    return_target = torch.tensor(
        (frame["target_return_5m"] > 0).to_numpy(),
        dtype=torch.long,
    )
    vol_target = torch.tensor(
        frame["target_vol_15m"].to_numpy(),
        dtype=torch.float32,
    )

    n = features.shape[0]
    train_end, val_end = split_indices(n)

    x_train = features[:train_end]
    x_val = features[train_end:val_end]
    x_test = features[val_end:]
    y_train = return_target[:train_end]
    y_val = return_target[train_end:val_end]
    y_test = return_target[val_end:]
    v_train = vol_target[:train_end]
    v_val = vol_target[train_end:val_end]
    v_test = vol_target[val_end:]

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std = torch.where(std > 0, std, torch.ones_like(std))
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    model = ExecutionBaselineMLP(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    cls_loss = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(x_train, y_train, v_train),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_accuracy = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y, batch_v in train_loader:
            optimizer.zero_grad(set_to_none=True)
            return_logits, vol_pred = model(batch_x)
            loss = cls_loss(return_logits, batch_y) + reg_loss(vol_pred, batch_v)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_x.size(0)

        model.eval()
        with torch.no_grad():
            val_logits, val_vol = model(x_val)
            val_accuracy = float((val_logits.argmax(dim=1) == y_val).float().mean().item())
            val_vol_mse = float(reg_loss(val_vol, v_val).item())
        log.info(
            "epoch=%d train_loss=%.4f val_accuracy=%.4f val_vol_mse=%.6f",
            epoch + 1,
            running_loss / max(len(x_train), 1),
            val_accuracy,
            val_vol_mse,
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed to produce a model checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits, test_vol = model(x_test)
        test_accuracy = float((test_logits.argmax(dim=1) == y_test).float().mean().item())
        test_vol_mse = float(reg_loss(test_vol, v_test).item())

    output_dir = layout.training_baseline_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "baseline_mlp.pt"
    metrics_path = output_dir / "baseline_mlp_metrics.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_columns": selected_feature_columns,
            "mean": mean,
            "std": std,
            "symbols": symbols,
        },
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "train_rows": len(x_train),
                "val_rows": len(x_val),
                "test_rows": len(x_test),
                "val_accuracy_best": best_val_accuracy,
                "test_accuracy": test_accuracy,
                "test_vol_mse": test_vol_mse,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return BaselineArtifacts(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )
