from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from denoisr.nn.drop_path import DropPath
from denoisr.nn.relative_pos import ShawRelativePositionBias
from denoisr.nn.smolgen import SmolgenBias


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with additive attention bias support.

    Accepts combined smolgen + Shaw PE biases.
    """

    def __init__(
        self,
        d_s: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads
        self.norm1 = nn.LayerNorm(d_s)
        self.qkv = nn.Linear(d_s, 3 * d_s)
        self.out_proj = nn.Linear(d_s, d_s)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_s)
        self.ffn = nn.Sequential(
            nn.Linear(d_s, ffn_dim),
            nn.Mish(),
            nn.Linear(ffn_dim, d_s),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.attn_dropout(self.out_proj(h))
        x = x + self.drop_path(h)

        x = x + self.drop_path(self.ffn_dropout(self.ffn(self.norm2(x))))
        return x


class ChessPolicyBackbone(nn.Module):
    """Encoder-only transformer with smolgen + Shaw relative PE.

    Two types of attention bias are combined additively:
    - Smolgen: content-dependent (adapts to specific position)
    - Shaw relative PE: topology-aware (captures spatial relationships)

    Both are computed once and shared across all transformer layers.
    """

    def __init__(
        self,
        d_s: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        gradient_checkpointing: bool = False,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self._gradient_checkpointing = gradient_checkpointing
        self.smolgen = SmolgenBias(d_s, num_heads)
        self.shaw_relative_pe = ShawRelativePositionBias(num_heads)
        dpr = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_s, num_heads, ffn_dim,
                    dropout=dropout, drop_path_rate=dpr[i],
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_s)

    def forward(self, x: Tensor) -> Tensor:
        smolgen_bias = self.smolgen(x)  # [B, H, 64, 64]
        shaw_bias = self.shaw_relative_pe()  # [H, 64, 64]
        combined_bias = smolgen_bias + shaw_bias.unsqueeze(0)
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch_checkpoint(layer, x, combined_bias, use_reentrant=False)
            else:
                x = layer(x, attn_bias=combined_bias)
        out: Tensor = self.final_norm(x)
        return out
