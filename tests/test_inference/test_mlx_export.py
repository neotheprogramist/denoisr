"""Tests for the MLX safetensors export script."""

import json
import pathlib

import torch
from safetensors.torch import load_file

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.scripts.config import ModelConfig, save_checkpoint
from denoisr.scripts.export_mlx import export_weights

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


class TestExportWeights:
    def _make_checkpoint(self, tmp_path: pathlib.Path) -> pathlib.Path:
        cfg = ModelConfig(
            num_planes=12,
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        )
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        )
        policy_head = ChessPolicyHead(d_s=SMALL_D_S)
        value_head = ChessValueHead(d_s=SMALL_D_S)
        ckpt_path = tmp_path / "model.pt"
        save_checkpoint(
            ckpt_path,
            cfg,
            encoder=encoder.state_dict(),
            backbone=backbone.state_dict(),
            policy_head=policy_head.state_dict(),
            value_head=value_head.state_dict(),
        )
        return ckpt_path

    def test_export_creates_safetensors(self, tmp_path: pathlib.Path) -> None:
        """export_weights should create a .safetensors file."""
        ckpt_path = self._make_checkpoint(tmp_path)
        out_path = tmp_path / "model.safetensors"
        export_weights(ckpt_path, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_export_creates_config_json(self, tmp_path: pathlib.Path) -> None:
        """export_weights should also create a .json config file."""
        ckpt_path = self._make_checkpoint(tmp_path)
        out_path = tmp_path / "model.safetensors"
        export_weights(ckpt_path, out_path)
        config_path = tmp_path / "model.json"
        assert config_path.exists()
        with open(config_path) as f:
            cfg = json.load(f)
        assert cfg["d_s"] == SMALL_D_S
        assert cfg["num_heads"] == SMALL_NUM_HEADS

    def test_exported_weights_have_expected_keys(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Exported safetensors should contain all model weight prefixes."""
        ckpt_path = self._make_checkpoint(tmp_path)
        out_path = tmp_path / "model.safetensors"
        export_weights(ckpt_path, out_path)
        weights = load_file(str(out_path))
        assert any(k.startswith("encoder.") for k in weights)
        assert any(k.startswith("backbone.") for k in weights)
        assert any(k.startswith("policy_head.") for k in weights)
        assert any(k.startswith("value_head.") for k in weights)

    def test_exported_weights_match_original(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Exported weights should be numerically identical to the originals."""
        cfg = ModelConfig(
            num_planes=12,
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        )
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S)
        original_weight = encoder.square_embed.weight.clone()

        ckpt_path = tmp_path / "model.pt"
        save_checkpoint(
            ckpt_path,
            cfg,
            encoder=encoder.state_dict(),
            backbone=ChessPolicyBackbone(
                d_s=SMALL_D_S,
                num_heads=SMALL_NUM_HEADS,
                num_layers=SMALL_NUM_LAYERS,
                ffn_dim=SMALL_FFN_DIM,
            ).state_dict(),
            policy_head=ChessPolicyHead(d_s=SMALL_D_S).state_dict(),
            value_head=ChessValueHead(d_s=SMALL_D_S).state_dict(),
        )

        out_path = tmp_path / "model.safetensors"
        export_weights(ckpt_path, out_path)
        weights = load_file(str(out_path))

        exported = weights["encoder.square_embed.weight"]
        assert torch.equal(original_weight, exported)
