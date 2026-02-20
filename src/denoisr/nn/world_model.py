import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalTransformerBlock(nn.Module):
    """Pre-norm transformer block with causal (autoregressive) masking."""

    def __init__(self, d_s: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads
        self.norm1 = nn.LayerNorm(d_s)
        self.qkv = nn.Linear(d_s, 3 * d_s)
        self.out_proj = nn.Linear(d_s, d_s)
        self.norm2 = nn.LayerNorm(d_s)
        self.ffn = nn.Sequential(
            nn.Linear(d_s, ffn_dim),
            nn.Mish(),
            nn.Linear(ffn_dim, d_s),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).reshape(B, T, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class ChessWorldModel(nn.Module):
    """Causal transformer world model (UniZero-style).

    Each timestep's (state, action) pair is compressed into a single
    token via mean-pooling + action embedding + MLP fusion (STORM approach).
    The causal transformer processes the token sequence, predicting
    next latent states and rewards.
    """

    def __init__(
        self, d_s: int, num_heads: int, num_layers: int, ffn_dim: int
    ) -> None:
        super().__init__()
        self.d_s = d_s
        self.state_compress = nn.Sequential(
            nn.Linear(d_s, d_s),
            nn.Mish(),
        )
        self.from_embed = nn.Embedding(64, d_s // 2)
        self.to_embed = nn.Embedding(64, d_s // 2)
        self.fuse = nn.Sequential(
            nn.Linear(d_s * 2, d_s),
            nn.Mish(),
            nn.Linear(d_s, d_s),
        )
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(d_s, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_s)
        self.state_pred = nn.Linear(d_s, 64 * d_s)
        self.reward_pred = nn.Linear(d_s, 1)

    def forward(
        self,
        states: Tensor,
        action_from: Tensor,
        action_to: Tensor,
    ) -> tuple[Tensor, Tensor]:
        B, T = states.shape[:2]

        compressed = self.state_compress(states.mean(dim=2))
        act_emb = torch.cat(
            [self.from_embed(action_from), self.to_embed(action_to)], dim=-1
        )
        fused = self.fuse(torch.cat([compressed, act_emb], dim=-1))

        h = fused
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)

        next_states = self.state_pred(h).reshape(B, T, 64, self.d_s)
        rewards = self.reward_pred(h).squeeze(-1)
        return next_states, rewards
