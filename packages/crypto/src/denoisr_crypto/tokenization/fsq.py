"""Finite scalar quantization tokenizer training and export."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import polars as pl
from safetensors.torch import load_file
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from denoisr_crypto.types import (
    DEFAULT_TOKENIZER_CONTEXT_LENGTH,
    DEFAULT_TOKENIZER_PATCH_SIZE,
    StorageLayout,
)


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels: tuple[int, ...]) -> None:
        super().__init__()
        self.levels = levels
        self.codebook_size = math.prod(levels)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = torch.tanh(latent)
        quantized_dims: list[torch.Tensor] = []
        index_dims: list[torch.Tensor] = []
        for dim, level_count in enumerate(self.levels):
            scale = level_count - 1
            normalized = (bounded[..., dim] + 1.0) * 0.5
            indices = torch.round(normalized * scale).clamp(0, scale)
            quantized = ((indices / scale) * 2.0) - 1.0
            quantized_dims.append(quantized)
            index_dims.append(indices.to(torch.long))
        stacked_quantized = torch.stack(quantized_dims, dim=-1)
        stacked_indices = torch.stack(index_dims, dim=-1)
        straight_through = latent + (stacked_quantized - latent).detach()
        token_ids = self._indices_to_token_ids(stacked_indices)
        return straight_through, token_ids

    def _indices_to_token_ids(self, indices: torch.Tensor) -> torch.Tensor:
        token_ids = torch.zeros(indices.shape[:-1], dtype=torch.long, device=indices.device)
        multiplier = 1
        for dim, level_count in enumerate(self.levels):
            token_ids = token_ids + (indices[..., dim] * multiplier)
            multiplier *= level_count
        return token_ids


class FsqSequenceAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        context_length: int,
        patch_size: int,
        model_dim: int,
        latent_dim: int,
        levels: tuple[int, ...],
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if context_length % patch_size != 0:
            raise ValueError("Context length must be divisible by patch size")
        self.context_length = context_length
        self.patch_size = patch_size
        self.input_dim = input_dim
        patch_dim = patch_size * input_dim
        self.input_proj = nn.Linear(patch_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(model_dim, latent_dim)
        self.quantizer = FiniteScalarQuantizer(levels)
        self.from_latent = nn.Linear(latent_dim, model_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, patch_dim)

    def _patchify(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_dim = inputs.shape
        num_patches = self.context_length // self.patch_size
        return inputs.view(batch_size, num_patches, self.patch_size * input_dim)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, patch_dim = patches.shape
        return patches.view(batch_size, num_patches * self.patch_size, self.input_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self._patchify(inputs)
        encoded = self.encoder(self.input_proj(patches))
        latent = self.to_latent(encoded)
        quantized, token_ids = self.quantizer(latent)
        decoded = self.decoder(self.from_latent(quantized))
        reconstruction = self._unpatchify(self.output_proj(decoded))
        return reconstruction, token_ids

    def encode_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(inputs)
        encoded = self.encoder(self.input_proj(patches))
        latent = self.to_latent(encoded)
        _, token_ids = self.quantizer(latent)
        return token_ids


@dataclass(frozen=True)
class FsqTokenizerArtifacts:
    checkpoint_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class TokenExportArtifacts:
    export_paths: dict[tuple[str, str], Path]


def _load_split_tensors(
    layout: StorageLayout,
    symbols: tuple[str, ...],
    split: str,
) -> torch.Tensor | None:
    tensors: list[torch.Tensor] = []
    for symbol in symbols:
        path = layout.tokenizer_corpus_path(symbol, split)
        if not path.exists():
            continue
        data = load_file(str(path))["inputs"]
        if data.numel() > 0:
            tensors.append(data)
    return None if not tensors else torch.cat(tensors, dim=0)


def _load_symbol_split_tensor(layout: StorageLayout, symbol: str, split: str) -> torch.Tensor | None:
    path = layout.tokenizer_corpus_path(symbol, split)
    if not path.exists():
        return None
    return load_file(str(path))["inputs"]


def _token_metrics(token_ids: torch.Tensor, codebook_size: int) -> tuple[float, float]:
    flat = token_ids.reshape(-1)
    counts = torch.bincount(flat, minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nonzero = probs[probs > 0]
    perplexity = float(torch.exp(-(nonzero * nonzero.log()).sum()).item()) if nonzero.numel() else 0.0
    utilization = float((counts > 0).float().mean().item())
    return perplexity, utilization


def train_fsq_tokenizer(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    run_name: str = "fsq_tokenizer",
    context_length: int = DEFAULT_TOKENIZER_CONTEXT_LENGTH,
    patch_size: int = DEFAULT_TOKENIZER_PATCH_SIZE,
    model_dim: int = 256,
    latent_dim: int = 5,
    levels: tuple[int, ...] = (8, 8, 8, 8, 8),
) -> FsqTokenizerArtifacts | None:
    train_inputs = _load_split_tensors(layout, symbols, "train")
    val_inputs = _load_split_tensors(layout, symbols, "val")
    if train_inputs is None or val_inputs is None:
        return None
    if train_inputs.shape[1] != context_length:
        context_length = int(train_inputs.shape[1])
    model = FsqSequenceAutoencoder(
        input_dim=int(train_inputs.shape[2]),
        context_length=context_length,
        patch_size=patch_size,
        model_dim=model_dim,
        latent_dim=latent_dim,
        levels=levels,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(train_inputs), batch_size=batch_size, shuffle=True)
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_val_tokens: torch.Tensor | None = None
    for _epoch in range(epochs):
        model.train()
        for (batch_x,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_reconstruction, val_tokens = model(val_inputs)
            val_loss = float(criterion(val_reconstruction, val_inputs).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_val_tokens = val_tokens.detach().clone()
    if best_state is None:
        raise RuntimeError("FSQ tokenizer training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_reconstruction, train_tokens = model(train_inputs)
        val_reconstruction, val_tokens = model(val_inputs)
        train_loss = float(criterion(train_reconstruction, train_inputs).item())
        val_loss = float(criterion(val_reconstruction, val_inputs).item())
    perplexity, utilization = _token_metrics(best_val_tokens if best_val_tokens is not None else val_tokens, model.quantizer.codebook_size)
    checkpoint_path = layout.tokenizer_checkpoint_path(run_name)
    metrics_path = layout.tokenizer_metrics_path(run_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "context_length": context_length,
                "patch_size": patch_size,
                "model_dim": model_dim,
                "latent_dim": latent_dim,
                "levels": levels,
                "symbols": symbols,
            },
        },
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "train_loss": train_loss,
                "val_loss_best": best_val_loss,
                "val_loss_final": val_loss,
                "token_perplexity": perplexity,
                "token_utilization": utilization,
                "train_sequences": int(train_inputs.shape[0]),
                "val_sequences": int(val_inputs.shape[0]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return FsqTokenizerArtifacts(checkpoint_path=checkpoint_path, metrics_path=metrics_path)


def export_token_dataset(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    run_name: str = "fsq_tokenizer",
    splits: tuple[str, ...] = ("train", "val", "test"),
    batch_size: int = 128,
) -> TokenExportArtifacts | None:
    checkpoint_path = layout.tokenizer_checkpoint_path(run_name)
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = FsqSequenceAutoencoder(
        input_dim=6,
        context_length=int(config["context_length"]),
        patch_size=int(config["patch_size"]),
        model_dim=int(config["model_dim"]),
        latent_dim=int(config["latent_dim"]),
        levels=tuple(config["levels"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    export_paths: dict[tuple[str, str], Path] = {}
    for symbol in symbols:
        for split in splits:
            inputs = _load_symbol_split_tensor(layout, symbol, split)
            index_path = layout.tokenizer_sequence_index_path(symbol, split)
            if inputs is None or inputs.numel() == 0 or not index_path.exists():
                continue
            token_batches: list[torch.Tensor] = []
            for start in range(0, inputs.shape[0], batch_size):
                batch = inputs[start : start + batch_size]
                with torch.no_grad():
                    token_batches.append(model.encode_tokens(batch))
            token_ids = torch.cat(token_batches, dim=0).tolist()
            index_frame = pl.read_parquet(index_path)
            export_frame = index_frame.with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit(split).alias("split"),
                pl.Series("tokens", token_ids),
            )
            export_path = layout.tokenizer_export_path(run_name, symbol, split)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_frame.write_parquet(export_path)
            export_paths[(symbol, split)] = export_path
    return None if not export_paths else TokenExportArtifacts(export_paths=export_paths)
