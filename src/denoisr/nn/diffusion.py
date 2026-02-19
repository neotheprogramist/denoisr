import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class CosineNoiseSchedule(nn.Module):
    """Cosine noise schedule for continuous DDPM (Nichol & Dhariwal 2021).

    Produces alpha_bar_t values that follow a cosine curve, giving
    a gentler noise schedule than linear beta scheduling.
    """

    alpha_bar: Tensor

    def __init__(self, num_timesteps: int, s: float = 0.008) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f = (
            torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2)
            ** 2
        )
        alpha_bar = f / f[0]
        self.register_buffer(
            "alpha_bar",
            alpha_bar[:num_timesteps].float().clamp(min=1e-5, max=0.9999),
        )

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: add noise at timestep t."""
        ab = self.alpha_bar[t]
        while ab.ndim < x_0.ndim:
            ab = ab.unsqueeze(-1)
        return ab.sqrt() * x_0 + (1 - ab).sqrt() * noise


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN-Zero conditioning."""

    def __init__(self, d_s: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads
        self.norm1 = nn.LayerNorm(d_s, elementwise_affine=False)
        self.qkv = nn.Linear(d_s, 3 * d_s)
        self.out_proj = nn.Linear(d_s, d_s)
        self.norm2 = nn.LayerNorm(d_s, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_s, d_s * 4),
            nn.Mish(),
            nn.Linear(d_s * 4, d_s),
        )
        self.adaln = nn.Sequential(
            nn.Mish(),
            nn.Linear(d_s, 6 * d_s),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        B, S, D = x.shape
        params = self.adaln(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(
            6, dim=-1
        )

        h = self.norm1(x) * (1 + scale1) + shift1
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + gate1 * h

        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.ffn(h)
        x = x + gate2 * h
        return x


class ChessDiffusionModule(nn.Module):
    """DiT-based diffusion module for latent-space trajectory imagination.

    Uses continuous DDPM (Gaussian noise) in the latent space of
    board representations. Conditioned on the current board state
    and diffusion timestep via AdaLN-Zero modulation.

    The final projection is zero-initialized so each block initially
    acts as identity, ensuring stable early training.
    """

    def __init__(
        self,
        d_s: int,
        num_heads: int,
        num_layers: int,
        num_timesteps: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self._gradient_checkpointing = gradient_checkpointing
        self.time_embed = nn.Sequential(
            nn.Embedding(num_timesteps, d_s),
            nn.Mish(),
            nn.Linear(d_s, d_s),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(d_s, d_s),
            nn.Mish(),
        )
        self.layers = nn.ModuleList(
            [DiTBlock(d_s, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_s)
        self.final_proj = nn.Linear(d_s, d_s)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.time_embed(t)
        c_emb = self.cond_proj(cond.mean(dim=1))
        c = (t_emb + c_emb).unsqueeze(1).expand(-1, 64, -1)

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch_checkpoint(layer, x, c, use_reentrant=False)
            else:
                x = layer(x, c)

        out: Tensor = self.final_proj(self.final_norm(x))
        return out
