"""Shared model configuration and construction helpers."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.nn.consistency import ChessConsistencyProjector
from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.nn.world_model import ChessWorldModel


@dataclass(frozen=True)
class ModelConfig:
    d_s: int = 256
    num_heads: int = 8
    num_layers: int = 15
    ffn_dim: int = 1024
    num_timesteps: int = 100
    world_model_layers: int = 12
    diffusion_layers: int = 6
    proj_dim: int = 256


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_encoder(cfg: ModelConfig, num_planes: int = 12) -> ChessEncoder:
    return ChessEncoder(num_planes=num_planes, d_s=cfg.d_s)


def build_backbone(cfg: ModelConfig) -> ChessPolicyBackbone:
    return ChessPolicyBackbone(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_dim=cfg.ffn_dim,
    )


def build_policy_head(cfg: ModelConfig) -> ChessPolicyHead:
    return ChessPolicyHead(d_s=cfg.d_s)


def build_value_head(cfg: ModelConfig) -> ChessValueHead:
    return ChessValueHead(d_s=cfg.d_s)


def build_world_model(cfg: ModelConfig) -> ChessWorldModel:
    return ChessWorldModel(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.world_model_layers,
        ffn_dim=cfg.ffn_dim,
    )


def build_diffusion(cfg: ModelConfig) -> ChessDiffusionModule:
    return ChessDiffusionModule(
        d_s=cfg.d_s,
        num_heads=cfg.num_heads,
        num_layers=cfg.diffusion_layers,
        num_timesteps=cfg.num_timesteps,
    )


def build_consistency(cfg: ModelConfig) -> ChessConsistencyProjector:
    return ChessConsistencyProjector(d_s=cfg.d_s, proj_dim=cfg.proj_dim)


def build_schedule(cfg: ModelConfig) -> CosineNoiseSchedule:
    return CosineNoiseSchedule(num_timesteps=cfg.num_timesteps)


def add_model_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("model")
    g.add_argument("--d-s", type=int, default=256, help="latent dimension")
    g.add_argument("--num-heads", type=int, default=8)
    g.add_argument("--num-layers", type=int, default=15, help="backbone layers")
    g.add_argument("--ffn-dim", type=int, default=1024)
    g.add_argument("--num-timesteps", type=int, default=100)
    g.add_argument("--world-model-layers", type=int, default=12)
    g.add_argument("--diffusion-layers", type=int, default=6)


def config_from_args(args: Namespace) -> ModelConfig:
    return ModelConfig(
        d_s=args.d_s,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        num_timesteps=args.num_timesteps,
        world_model_layers=args.world_model_layers,
        diffusion_layers=args.diffusion_layers,
    )


def save_checkpoint(
    path: Path,
    cfg: ModelConfig,
    **state_dicts: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"config": cfg.__dict__}
    data.update(state_dicts)
    torch.save(data, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[ModelConfig, dict]:
    data = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**data.pop("config"))
    return cfg, data
