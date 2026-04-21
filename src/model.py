"""Score network: CSDI-style Transformer (paper §3.1.1).

Architecture (Tashiro et al. 2021, adapted):
  - 1D Conv projects (1, L) → (C, L)
  - Three embeddings added to the latent: diffusion-step, positional, feature
  - Transformer block captures global temporal dependencies
  - n_layers gated residual blocks (gate ⊙ tanh(filter)) with skip connections
  - Final 1D Conv projects back to (1, L)
"""
from __future__ import annotations

import math

import torch
from torch import nn


def diffusion_step_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for continuous t ∈ [0, 1]. (B,) → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


def positional_embedding(L: int, dim: int, device) -> torch.Tensor:
    """Fixed sinusoidal positional embedding (1, dim, L)."""
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = pos * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (L, dim)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb.transpose(0, 1).unsqueeze(0)                      # (1, dim, L)


class TransformerLayer(nn.Module):
    def __init__(self, channels: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, x):  # x: (B, C, L)
        h = x.transpose(1, 2)                     # (B, L, C)
        h_norm = self.ln(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + attn_out
        h = h + self.ff(self.ln2(h))
        return h.transpose(1, 2)


class GatedResidualBlock(nn.Module):
    """CSDI-style gated residual block: conv → split(filter, gate) → ⊗ → conv."""

    def __init__(self, channels: int, diff_emb_dim: int, feat_emb_dim: int):
        super().__init__()
        self.conv_in = nn.Conv1d(channels, 2 * channels, kernel_size=3, padding=1)
        self.diff_proj = nn.Linear(diff_emb_dim, channels)
        self.feat_proj = nn.Conv1d(feat_emb_dim, 2 * channels, kernel_size=1)
        self.conv_mid = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_res = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, diff_emb, feat_emb):
        # x:(B,C,L), diff_emb:(B,Dd), feat_emb:(B,Df,L)
        h = x + self.diff_proj(diff_emb).unsqueeze(-1)
        h = self.conv_in(h) + self.feat_proj(feat_emb)
        filt, gate = h.chunk(2, dim=1)
        h = torch.tanh(filt) * torch.sigmoid(gate)
        h = self.conv_mid(h)
        r, s = h.chunk(2, dim=1)
        res = (x + self.conv_res(r)) / math.sqrt(2.0)
        skip = self.conv_skip(s)
        return res, skip


class ScoreNet(nn.Module):
    """Transformer-CSDI score network. Input (B, 1, L) → output (B, 1, L)."""

    def __init__(self, channels: int = 128, diff_emb_dim: int = 256,
                 feat_emb_dim: int = 64, n_heads: int = 8, n_layers: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.feat_emb_dim = feat_emb_dim
        self.diff_emb_dim = diff_emb_dim

        self.input_proj = nn.Conv1d(1, channels, kernel_size=1)
        self.diff_mlp = nn.Sequential(
            nn.Linear(diff_emb_dim, diff_emb_dim),
            nn.SiLU(),
            nn.Linear(diff_emb_dim, diff_emb_dim),
        )
        self.feat_emb = nn.Parameter(torch.randn(1, feat_emb_dim, 1) * 0.02)
        self.transformer = TransformerLayer(channels, n_heads, dropout=dropout)
        self.blocks = nn.ModuleList([
            GatedResidualBlock(channels, diff_emb_dim, feat_emb_dim)
            for _ in range(n_layers)
        ])
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L); t: (B,)."""
        B, _, L = x.shape
        d_emb = self.diff_mlp(diffusion_step_embedding(t, self.diff_emb_dim))
        p_emb = positional_embedding(L, self.channels, x.device)          # (1,C,L)
        f_emb = self.feat_emb.expand(B, -1, L)                            # (B,Df,L)

        h = self.input_proj(x) + p_emb
        h = self.transformer(h)

        skip_sum = 0
        for blk in self.blocks:
            h, skip = blk(h, d_emb, f_emb)
            skip_sum = skip_sum + skip
        out = self.output(skip_sum / math.sqrt(len(self.blocks)))
        return out
