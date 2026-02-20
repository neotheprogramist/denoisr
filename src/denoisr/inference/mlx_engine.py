# mypy: ignore-errors
# MLX has no type stubs; every class subclassing mnn.Module and every
# mx.array annotation would require type: ignore, so we suppress file-wide.
"""MLX inference engine for Apple Silicon.

Reimplements the PyTorch inference modules (encoder, backbone, policy head,
value head) in MLX for native Apple Silicon inference. Weights are loaded
from safetensors files exported by denoisr-export-mlx.

MLX is an optional dependency -- this module raises ImportError with a
helpful message if mlx is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from denoisr.inference._weight_remap import _remap_all_keys

if TYPE_CHECKING:
    import chess

try:
    import mlx.core as mx
    import mlx.nn as mnn
except ImportError as e:
    raise ImportError(
        "MLX is required for Apple Silicon inference. "
        "Install with: uv add mlx"
    ) from e


def _mish(x: mx.array) -> mx.array:
    """Mish activation: x * tanh(softplus(x))."""
    return x * mx.tanh(mx.softplus(x))


class MLXChessEncoder(mnn.Module):
    """Encodes board tensor [B, C, 8, 8] into latent tokens [B, 64, d_s]."""

    def __init__(self, num_planes: int, d_s: int) -> None:
        super().__init__()
        self.square_embed = mnn.Linear(num_planes, d_s)
        self.global_embed_0 = mnn.Linear(num_planes * 64, d_s)
        self.global_embed_2 = mnn.Linear(d_s, d_s)
        self.norm = mnn.LayerNorm(d_s)

    def __call__(self, x: mx.array) -> mx.array:
        B, C, _H, _W = x.shape
        squares = x.reshape(B, C, 64).transpose(0, 2, 1)  # [B, 64, C]
        local = self.square_embed(squares)

        flat = x.reshape(B, C * 64)
        glob = _mish(self.global_embed_0(flat))
        glob = self.global_embed_2(glob)
        glob = mx.expand_dims(glob, axis=1)  # [B, 1, d_s]

        return self.norm(local + glob)


class MLXSmolgenBias(mnn.Module):
    """Content-dependent attention bias."""

    def __init__(self, d_s: int, num_heads: int, compress_dim: int = 256) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.compress_0 = mnn.Linear(64 * d_s, compress_dim)
        self.project = mnn.Linear(compress_dim, num_heads * 64 * 64)

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        flat = x.reshape(B, -1)
        compressed = _mish(self.compress_0(flat))
        biases = self.project(compressed)
        return biases.reshape(B, self.num_heads, 64, 64)


class MLXTransformerBlock(mnn.Module):
    """Pre-norm transformer block with attention bias."""

    def __init__(self, d_s: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads
        self.norm1 = mnn.LayerNorm(d_s)
        self.qkv = mnn.Linear(d_s, 3 * d_s)
        self.out_proj = mnn.Linear(d_s, d_s)
        self.norm2 = mnn.LayerNorm(d_s)
        self.ffn_0 = mnn.Linear(d_s, ffn_dim)
        self.ffn_2 = mnn.Linear(ffn_dim, d_s)

    def __call__(self, x: mx.array, attn_bias: mx.array | None = None) -> mx.array:
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0, :, :].transpose(0, 2, 1, 3)  # [B, H, S, D_h]
        k = qkv[:, :, 1, :, :].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2, :, :].transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = mx.softmax(attn, axis=-1)
        h = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + h

        h = self.norm2(x)
        h = _mish(self.ffn_0(h))
        h = self.ffn_2(h)
        return x + h


class MLXPolicyHead(mnn.Module):
    """Bilinear query-key policy head."""

    def __init__(self, d_s: int, d_head: int = 128) -> None:
        super().__init__()
        self.query = mnn.Linear(d_s, d_head)
        self.key = mnn.Linear(d_s, d_head)
        self.scale = d_head ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        q = self.query(x)
        k = self.key(x)
        return (q @ k.transpose(0, 2, 1)) * self.scale


class MLXValueHead(mnn.Module):
    """WDL + ply value head."""

    def __init__(self, d_s: int) -> None:
        super().__init__()
        self.norm = mnn.LayerNorm(d_s)
        self.wdl_linear = mnn.Linear(d_s, 3)
        self.ply_linear = mnn.Linear(d_s, 1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        pooled = self.norm(x.mean(axis=1))
        wdl = mx.softmax(self.wdl_linear(pooled), axis=-1)
        ply = mnn.softplus(self.ply_linear(pooled))
        return wdl, ply


class MLXChessEngine:
    """MLX-native chess inference engine for Apple Silicon.

    Loads weights from safetensors exported by denoisr-export-mlx and
    reimplements the full inference pipeline (encoder -> backbone -> heads)
    using MLX operations for native Apple Silicon performance.
    """

    def __init__(
        self,
        weights_path: Path,
        config_path: Path | None = None,
    ) -> None:
        if config_path is None:
            config_path = weights_path.with_suffix(".json")

        with open(config_path) as f:
            cfg = json.load(f)

        d_s: int = cfg["d_s"]
        num_heads: int = cfg["num_heads"]
        num_layers: int = cfg["num_layers"]
        ffn_dim: int = cfg["ffn_dim"]
        num_planes: int = cfg["num_planes"]

        self._encoder = MLXChessEncoder(num_planes, d_s)
        self._smolgen = MLXSmolgenBias(d_s, num_heads)
        self._layers = [
            MLXTransformerBlock(d_s, num_heads, ffn_dim)
            for _ in range(num_layers)
        ]
        self._final_norm = mnn.LayerNorm(d_s)
        self._policy_head = MLXPolicyHead(d_s)
        self._value_head = MLXValueHead(d_s)

        # Shaw PE buffers
        self._shaw_bias_table = mx.zeros((num_heads, 15, 15))
        self._shaw_rank_idx = mx.zeros((64, 64), dtype=mx.int32)
        self._shaw_file_idx = mx.zeros((64, 64), dtype=mx.int32)

        self._num_planes = num_planes
        self._num_layers = num_layers

        # Load and remap weights
        weights = dict(mx.load(str(weights_path)))
        self._apply_weights(weights)

    def _apply_weights(self, weights: dict[str, mx.array]) -> None:
        """Remap PyTorch keys and load into MLX modules."""
        _remap_all_keys(weights, self._num_layers)

        # Encoder
        self._encoder.load_weights(
            list(self._extract_prefix(weights, "encoder."))
        )

        # Smolgen
        self._smolgen.load_weights(
            list(self._extract_prefix(weights, "backbone.smolgen."))
        )

        # Shaw relative position encoding (buffers, not nn.Module)
        shaw_prefix = "backbone.shaw_relative_pe."
        if f"{shaw_prefix}bias_table" in weights:
            self._shaw_bias_table = weights[f"{shaw_prefix}bias_table"]
        if f"{shaw_prefix}rank_idx" in weights:
            self._shaw_rank_idx = weights[f"{shaw_prefix}rank_idx"].astype(
                mx.int32
            )
        if f"{shaw_prefix}file_idx" in weights:
            self._shaw_file_idx = weights[f"{shaw_prefix}file_idx"].astype(
                mx.int32
            )

        # Transformer layers
        for i, layer in enumerate(self._layers):
            prefix = f"backbone.layers.{i}."
            layer.load_weights(list(self._extract_prefix(weights, prefix)))

        # Final norm
        fn_prefix = "backbone.final_norm."
        self._final_norm.load_weights(
            list(self._extract_prefix(weights, fn_prefix))
        )

        # Policy head
        self._policy_head.load_weights(
            list(self._extract_prefix(weights, "policy_head."))
        )

        # Value head
        self._value_head.load_weights(
            list(self._extract_prefix(weights, "value_head."))
        )

    @staticmethod
    def _extract_prefix(
        weights: dict[str, mx.array], prefix: str
    ) -> list[tuple[str, mx.array]]:
        """Extract weight entries matching a prefix, stripping the prefix."""
        return [
            (k.removeprefix(prefix), v)
            for k, v in weights.items()
            if k.startswith(prefix)
        ]

    def _shaw_bias(self) -> mx.array:
        """Compute Shaw relative position bias [H, 64, 64]."""
        return self._shaw_bias_table[:, self._shaw_rank_idx, self._shaw_file_idx]

    def _forward(
        self, board_tensor: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Full forward pass: encoder -> backbone -> heads."""
        x = self._encoder(board_tensor)

        smolgen_bias = self._smolgen(x)
        shaw_bias = self._shaw_bias()
        combined_bias = smolgen_bias + mx.expand_dims(shaw_bias, axis=0)

        for layer in self._layers:
            x = layer(x, attn_bias=combined_bias)
        x = self._final_norm(x)

        logits = self._policy_head(x)
        wdl, ply = self._value_head(x)
        return logits, wdl, ply

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select best legal move using greedy argmax over policy logits.

        Creates a board encoder on each call -- simplicity over premature
        optimization for inference use.
        """
        import chess as chess_lib

        from denoisr.data.board_encoder import SimpleBoardEncoder
        from denoisr.data.extended_board_encoder import ExtendedBoardEncoder

        if self._num_planes == 12:
            board_encoder: SimpleBoardEncoder | ExtendedBoardEncoder = (
                SimpleBoardEncoder()
            )
        else:
            board_encoder = ExtendedBoardEncoder()

        board_tensor = board_encoder.encode(board).data
        x = mx.array(board_tensor.numpy())[None]  # [1, C, 8, 8]

        logits, _wdl, _ply = self._forward(x)
        logits_2d = logits[0]  # [64, 64]

        # Build legal move mask: -inf for illegal, 0 for legal
        import numpy as np

        mask_np = np.full((64, 64), -1e9, dtype=np.float32)
        for move in board.legal_moves:
            mask_np[move.from_square, move.to_square] = 0.0
        mask = mx.array(mask_np)

        masked = logits_2d + mask
        flat = masked.reshape(-1)
        idx = int(mx.argmax(flat).item())

        from_sq = idx // 64
        to_sq = idx % 64

        promotion = None
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess_lib.PAWN:
            to_rank = chess_lib.square_rank(to_sq)
            if (piece.color == chess_lib.WHITE and to_rank == 7) or (
                piece.color == chess_lib.BLACK and to_rank == 0
            ):
                promotion = chess_lib.QUEEN

        return chess_lib.Move(from_sq, to_sq, promotion)
