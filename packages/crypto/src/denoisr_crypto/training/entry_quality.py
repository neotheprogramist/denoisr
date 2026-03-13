"""Supervised entry-quality model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from denoisr_crypto.labels import ENTRY_LABEL_COLUMNS
from denoisr_crypto.training.losses import LossConfig, compute_entry_loss, temperature_for_epoch
from denoisr_crypto.types import DEFAULT_ENTRY_DECISION_INTERVAL, StorageLayout

_METADATA_COLUMNS = {"exchange", "market", "symbol", "open_time"}
_LABEL_COLUMNS = tuple(ENTRY_LABEL_COLUMNS)


@dataclass(frozen=True)
class EntryTrainingArtifacts:
    checkpoint_path: Path
    metrics_path: Path


class EntryQualityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.denoise_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        noise_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.trunk(x)
        score = torch.sigmoid(self.score_head(latent).squeeze(-1))
        sampled_noise = torch.randn_like(latent) if noise is None else noise
        sampled_scale = (
            torch.rand(latent.shape[0], 1, device=latent.device) * 0.20 + 0.05
            if noise_scale is None
            else noise_scale
        )
        noisy_latent = latent + (sampled_noise * sampled_scale)
        predicted_noise = self.denoise_head(noisy_latent)
        return score, predicted_noise, sampled_noise


def load_entry_training_frame(
    layout: StorageLayout,
    symbols: tuple[str, ...],
    *,
    decision_interval: str = DEFAULT_ENTRY_DECISION_INTERVAL,
) -> pl.DataFrame | None:
    frames: list[pl.DataFrame] = []
    for symbol_index, symbol in enumerate(symbols):
        dataset_path = layout.entry_dataset_path(symbol, decision_interval)
        if not dataset_path.exists():
            return None
        frame = pl.read_parquet(dataset_path).drop_nans(subset=_LABEL_COLUMNS).with_columns(
            pl.lit(symbol_index).cast(pl.Float64).alias("symbol_id")
        )
        frames.append(frame)
    return None if not frames else pl.concat(frames, how="vertical").sort("open_time")


def entry_feature_columns(frame: pl.DataFrame) -> list[str]:
    schema = frame.schema
    return [
        column
        for column in frame.columns
        if column not in _METADATA_COLUMNS
        and column not in _LABEL_COLUMNS
        and schema[column] != pl.String
        and "time" not in column
    ]


def _split_indices(row_count: int) -> tuple[int, int]:
    train_end = int(row_count * 0.7)
    val_end = int(row_count * 0.85)
    if train_end == 0 or val_end <= train_end:
        raise ValueError("Not enough rows to create train/val/test splits")
    return train_end, val_end


def _labels_to_tensor(frame: pl.DataFrame) -> torch.Tensor:
    return torch.tensor(frame.select(_LABEL_COLUMNS).to_numpy(), dtype=torch.float32)


def _tensor_batch(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: labels[:, index] for index, name in enumerate(_LABEL_COLUMNS)}


def _calibration_error(score: torch.Tensor, opportunity: torch.Tensor, bins: int = 10) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=score.device)
    ece = torch.zeros((), device=score.device)
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        mask = (score >= start) & (score < end if end < 1.0 else score <= end)
        if mask.any():
            confidence = score[mask].mean()
            accuracy = opportunity[mask].mean()
            ece = ece + (mask.float().mean() * torch.abs(confidence - accuracy))
    return float(ece.item())


def _threshold_sweep(
    *,
    score: torch.Tensor,
    labels: dict[str, torch.Tensor],
    taus: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    quality = labels["quality_ratio_disc"]
    sweep: dict[str, dict[str, float]] = {}
    for tau in taus:
        fired = score >= tau
        firing_rate = float(fired.float().mean().item())
        if fired.any():
            selected_quality = float(quality[fired].mean().item())
            selected_r_up = float(labels["r_up"][fired].mean().item())
            selected_r_dd = float(labels["r_dd"][fired].mean().item())
            opportunity_precision = float(labels["opportunity_flag"][fired].mean().item())
        else:
            selected_quality = 0.0
            selected_r_up = 0.0
            selected_r_dd = 0.0
            opportunity_precision = 0.0
        missed = (~fired) & (labels["opportunity_flag"] > 0.5)
        sweep[f"{tau:.2f}"] = {
            "firing_rate": firing_rate,
            "selected_quality_ratio_disc": selected_quality,
            "selected_r_up": selected_r_up,
            "selected_r_dd": selected_r_dd,
            "opportunity_precision": opportunity_precision,
            "missed_opportunity_rate": float(missed.float().mean().item()),
        }
    return sweep


def _evaluate_split(
    *,
    model: EntryQualityMLP,
    features: torch.Tensor,
    labels: torch.Tensor,
    config: LossConfig,
    epoch: int,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        score, predicted_noise, sampled_noise = model(features)
        denoise_loss = nn.functional.mse_loss(predicted_noise, sampled_noise)
        batch = _tensor_batch(labels)
        loss_artifacts = compute_entry_loss(
            score=score,
            temp=temperature_for_epoch(epoch),
            batch=batch,
            config=config,
            denoise_loss=denoise_loss,
        )
        opportunity = (batch["opportunity_flag"] > 0.5).float()
    return {
        "loss": float(loss_artifacts.total.item()),
        "components": loss_artifacts.components,
        "calibration_ece": _calibration_error(score, opportunity),
        "threshold_sweep": _threshold_sweep(
            score=score,
            labels=batch,
            taus=(0.30, 0.40, 0.50, 0.60, 0.70),
        ),
    }


def train_entry_quality_model(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    decision_interval: str = DEFAULT_ENTRY_DECISION_INTERVAL,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    run_name: str = "entry_quality_model",
    loss_config: LossConfig | None = None,
) -> EntryTrainingArtifacts | None:
    frame = load_entry_training_frame(layout, symbols, decision_interval=decision_interval)
    if frame is None:
        return None
    feature_cols = entry_feature_columns(frame)
    feature_frame = frame.select(feature_cols).with_columns(pl.all().fill_null(0.0))
    features = torch.tensor(feature_frame.to_numpy(), dtype=torch.float32)
    labels = _labels_to_tensor(frame)
    train_end, val_end = _split_indices(features.shape[0])

    x_train = features[:train_end]
    x_val = features[train_end:val_end]
    x_test = features[val_end:]
    y_train = labels[:train_end]
    y_val = labels[train_end:val_end]
    y_test = labels[val_end:]

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std = torch.where(std > 0, std, torch.ones_like(std))
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    model = EntryQualityMLP(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    effective_config = loss_config or LossConfig()

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            score, predicted_noise, sampled_noise = model(batch_x)
            denoise_loss = nn.functional.mse_loss(predicted_noise, sampled_noise)
            loss_artifacts = compute_entry_loss(
                score=score,
                temp=temperature_for_epoch(epoch),
                batch=_tensor_batch(batch_y),
                config=effective_config,
                denoise_loss=denoise_loss,
            )
            loss_artifacts.total.backward()
            optimizer.step()
            epoch_loss += float(loss_artifacts.total.item())
            batch_count += 1

        val_metrics = _evaluate_split(
            model=model,
            features=x_val,
            labels=y_val,
            config=effective_config,
            epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(batch_count, 1),
                "val_loss": float(val_metrics["loss"]),
                "temperature": temperature_for_epoch(epoch),
            }
        )
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Entry-quality training did not produce a checkpoint")

    model.load_state_dict(best_state)
    val_metrics = _evaluate_split(
        model=model,
        features=x_val,
        labels=y_val,
        config=effective_config,
        epoch=epochs,
    )
    test_metrics = _evaluate_split(
        model=model,
        features=x_test,
        labels=y_test,
        config=effective_config,
        epoch=epochs,
    )

    checkpoint_path = layout.entry_model_checkpoint_path(run_name)
    metrics_path = layout.entry_model_metrics_path(run_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_columns": feature_cols,
            "mean": mean,
            "std": std,
            "symbols": symbols,
            "decision_interval": decision_interval,
            "loss_config": asdict(effective_config),
        },
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "train_rows": int(x_train.shape[0]),
                "val_rows": int(x_val.shape[0]),
                "test_rows": int(x_test.shape[0]),
                "best_val_loss": best_val_loss,
                "history": history,
                "val": val_metrics,
                "test": test_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return EntryTrainingArtifacts(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )
