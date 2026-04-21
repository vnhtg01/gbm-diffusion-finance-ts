"""Training (DSM loss) and sampling (reverse SDE Euler) utilities.

DSM loss (paper §2.1):
  L = E_{t ~ U(0, T), x_0, ε} [ λ(t) || s_θ(x_t, t) - ∇ log p_{t|0}(x_t | x_0) ||^2 ]

For Gaussian transition kernels (VE, VP, GBM):
  ∇ log p_{t|0}(x_t | x_0) = -(x_t - mean) / std^2 = -ε / std
so the DSM loss is equivalent to matching s_θ to -ε / std, and we use
λ(t) = std^2 to balance scales across t (standard choice).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .sde import SDE


def dsm_loss(model: nn.Module, sde: SDE, x0: torch.Tensor,
             eps_t: float = 1e-3) -> torch.Tensor:
    """Standard DSM with λ(t) = std^2."""
    B = x0.size(0)
    t = torch.rand(B, device=x0.device) * (sde.cfg.T - eps_t) + eps_t
    xt, eps = sde.sample_xt(x0, t)
    score = model(xt, t)
    _, std = sde.marginal(x0, t)
    std = std.view(-1, *([1] * (x0.ndim - 1)))
    # target = -ε / std; loss uses λ = std^2 to remove the 1/std^2 scaling
    loss = ((score * std + eps) ** 2).mean()
    return loss


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")


def train(model: nn.Module, sde: SDE, loader: DataLoader, *, lr: float,
          epochs: int, device: str, ckpt_path: Path, log_every: int = 50,
          grad_clip: float = 1.0) -> TrainState:
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    state = TrainState()
    pbar = tqdm(range(epochs), desc="training", unit="epoch")
    for epoch in pbar:
        state.epoch = epoch
        running = 0.0
        for i, batch in enumerate(loader):
            x0 = batch.to(device).float()
            if x0.ndim == 2:
                x0 = x0.unsqueeze(1)                   # (B, 1, L)
            loss = dsm_loss(model, sde, x0)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            state.step += 1
            if state.step % log_every == 0:
                tqdm.write(f"[e{epoch:4d} s{state.step:6d}] loss={loss.item():.4f}")

        avg = running / max(1, len(loader))
        if avg < state.best_loss:
            state.best_loss = avg
            torch.save({"model": model.state_dict(), "state": state.__dict__},
                       ckpt_path)
        pbar.set_postfix(loss=f"{avg:.4f}", best=f"{state.best_loss:.4f}")
    pbar.close()
    return state


@torch.no_grad()
def sample(model: nn.Module, sde: SDE, shape: tuple[int, int, int], *,
           n_steps: int = 2000, device: str = "cuda",
           x0_cond: torch.Tensor | None = None) -> torch.Tensor:
    """Reverse-time Euler sampler.

    shape = (B, 1, L). For VE/GBM, initial sample ~ N(0, σ_T^2 I); we set
    the first log-price entry to x0_cond[:, 0, 0] if provided, so the
    generated trajectory anchors to a realistic starting log-price.
    """
    model.eval().to(device)
    B, C, L = shape
    t_T = torch.ones(B, device=device) * sde.cfg.T
    _, std_T = sde.marginal(torch.zeros(B, C, L, device=device), t_T)
    std_T = std_T.view(-1, 1, 1)
    x = std_T * torch.randn(B, C, L, device=device)
    if x0_cond is not None:
        x[:, :, 0] = x0_cond[:, :, 0].to(device)

    ts = torch.linspace(sde.cfg.T, 1e-3, n_steps + 1, device=device)
    for i in tqdm(range(n_steps), desc="sampling"):
        t_cur = ts[i].expand(B)
        dt = (ts[i + 1] - ts[i])                       # negative
        score = model(x, t_cur)
        drift, g = sde.reverse_drift_diffusion(x, t_cur, score)
        noise = torch.randn_like(x)
        x = x + drift * dt + g * torch.sqrt(-dt) * noise
    return x.cpu()
