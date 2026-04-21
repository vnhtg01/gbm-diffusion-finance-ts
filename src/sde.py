"""Forward SDEs and noise schedules (paper §2.1, §3, §4).

Noise schedules σ_t on t ∈ [0, 1]:
  - linear:      σ_t^2 = σ_min^2 + t (σ_max^2 - σ_min^2)
  - exponential: σ_t   = σ_min (σ_max/σ_min)^t
  - cosine:      σ_t   = σ_min + (σ_max - σ_min) (1 - cos(π t)) / 2

Forward SDEs operate on log-prices X = log S (for GBM/CEV the forward is derived
via Itô and drift cancellation μ_t = σ_t^2 / 2 → reduces to VE SDE in log-space).
  - VE:  dX = σ_t dW                      → X_t ~ N(X_0, ∫_0^t σ_s^2 ds · I)
  - VP:  dX = -½ σ_t^2 X dt + σ_t dW      → X_t = √α X_0 + √(1-α) ε
  - GBM: identical to VE in log-space after drift cancellation (paper eq. 3.3)
  - CEV (extension): dS = μ S dt + σ S^γ dW; log-space:
         dX = (μ e^{(γ-1)X} - ½ σ^2 e^{2(γ-1)X}) dt + σ e^{(γ-1)X} dW
         With γ=1 we recover GBM. Training uses discrete Euler–Maruyama
         instead of a closed-form transition kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# ---------- Noise schedules ----------

def sigma_linear(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    return torch.sqrt(sigma_min ** 2 + t * (sigma_max ** 2 - sigma_min ** 2))


def sigma_exponential(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    return sigma_min * (sigma_max / sigma_min) ** t


def sigma_cosine(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    return sigma_min + (sigma_max - sigma_min) * (1 - torch.cos(math.pi * t)) / 2


SCHEDULES = {
    "linear": sigma_linear,
    "exponential": sigma_exponential,
    "cosine": sigma_cosine,
}


def integrated_variance(t: torch.Tensor, schedule: str,
                         sigma_min: float, sigma_max: float,
                         n_quad: int = 256) -> torch.Tensor:
    """∫_0^t σ_s^2 ds (trapezoidal)."""
    fn = SCHEDULES[schedule]
    # quadrature grid (0, t) per batch element
    grid = torch.linspace(0, 1, n_quad, device=t.device).view(1, -1)   # (1, Q)
    s = grid * t.view(-1, 1)                                            # (B, Q)
    sig2 = fn(s, sigma_min, sigma_max) ** 2
    dx = s[:, 1:] - s[:, :-1]
    return ((sig2[:, 1:] + sig2[:, :-1]) * 0.5 * dx).sum(dim=1)         # (B,)


# ---------- Forward SDE interface ----------

@dataclass
class SDEConfig:
    type: str
    schedule: str
    sigma_min: float
    sigma_max: float
    T: float = 1.0
    cev_gamma: float = 1.0


class SDE:
    """Common interface for VE / VP / GBM / CEV forward SDEs in log-price space."""

    def __init__(self, cfg: SDEConfig):
        self.cfg = cfg

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return SCHEDULES[self.cfg.schedule](t, self.cfg.sigma_min, self.cfg.sigma_max)

    def marginal(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) of p_{t|0}(x_t | x_0). t shape (B,)."""
        raise NotImplementedError

    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t and return (x_t, noise ε). Used for DSM training."""
        mean, std = self.marginal(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + std.view(-1, *([1] * (x0.ndim - 1))) * eps
        return xt, eps

    def reverse_drift_diffusion(self, x: torch.Tensor, t: torch.Tensor,
                                 score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (drift, g) for reverse-time Euler step dx = drift dt + g dW̄."""
        raise NotImplementedError


class VESDE(SDE):
    """dX = σ_t dW (equivalent to GBM on log-prices after drift cancellation)."""

    def marginal(self, x0, t):
        var = integrated_variance(t, self.cfg.schedule, self.cfg.sigma_min, self.cfg.sigma_max)
        return x0, torch.sqrt(var + 1e-12)

    def reverse_drift_diffusion(self, x, t, score):
        sig = self.sigma(t).view(-1, *([1] * (x.ndim - 1)))
        # d[σ_t^2]/dt via finite diff (could derive analytically per schedule)
        dt = 1e-3
        sig_p = self.sigma((t + dt).clamp(max=1.0)).view_as(sig)
        d_sig2 = (sig_p ** 2 - sig ** 2) / dt
        drift = -d_sig2 * score
        g = torch.sqrt(d_sig2.clamp(min=0))
        return drift, g


class VPSDE(SDE):
    """dX = -½ σ_t^2 X dt + σ_t dW. Marginal: N(√α X_0, (1-α) I)."""

    def _alpha(self, t):
        # α(t) = exp(-∫_0^t σ_s^2 ds)
        return torch.exp(-integrated_variance(t, self.cfg.schedule,
                                               self.cfg.sigma_min, self.cfg.sigma_max))

    def marginal(self, x0, t):
        a = self._alpha(t).view(-1, *([1] * (x0.ndim - 1)))
        mean = torch.sqrt(a) * x0
        std = torch.sqrt((1 - a).clamp(min=1e-12)).squeeze()
        if std.ndim == 0:
            std = std.view(1)
        return mean, std.view(-1)

    def reverse_drift_diffusion(self, x, t, score):
        sig = self.sigma(t).view(-1, *([1] * (x.ndim - 1)))
        drift = sig ** 2 * (score - 0.5 * x)
        g = sig
        return drift, g


class GBMSDE(VESDE):
    """Paper eq. 3.3: after drift cancellation, GBM → VE in log-price space.

    Kept as a distinct class so that experiment bookkeeping is explicit.
    In price-space this corresponds to dS = μ_t S dt + σ_t S dW with
    μ_t = σ_t^2 / 2.
    """


class CEVSDE(SDE):
    """Extension: CEV forward in price space, operates on log-price with
    drift cancellation analogue. γ=1 ≡ GBM.

    Training uses marginal approximation via Euler–Maruyama on a fine grid;
    for γ≠1 there is no simple closed-form transition — caller should pass
    pre-sampled (x_t, ε) pairs generated by `euler_forward` below.
    """

    def marginal(self, x0, t):
        # fallback: treat as VE for training unless euler_forward is used
        var = integrated_variance(t, self.cfg.schedule, self.cfg.sigma_min, self.cfg.sigma_max)
        return x0, torch.sqrt(var + 1e-12)

    def euler_forward(self, x0: torch.Tensor, t_target: torch.Tensor,
                      n_steps: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate forward SDE in log-price space via Euler–Maruyama.

        dX = (μ e^{(γ-1)X} - ½ σ^2 e^{2(γ-1)X}) dt + σ e^{(γ-1)X} dW
        with μ chosen so the drift reduces to VE at γ=1.
        """
        gamma = self.cfg.cev_gamma
        dt = t_target / n_steps
        x = x0.clone()
        for k in range(n_steps):
            t = (k + 1) / n_steps * t_target
            sig = self.sigma(t).view(-1, *([1] * (x.ndim - 1)))
            scale = torch.exp((gamma - 1) * x)
            mu = 0.5 * (sig * scale) ** 2
            drift = mu - 0.5 * (sig * scale) ** 2
            diff = sig * scale
            x = x + drift * dt.view(-1, *([1] * (x.ndim - 1))) \
                  + diff * torch.sqrt(dt.view(-1, *([1] * (x.ndim - 1)))) * torch.randn_like(x)
        return x, None

    def reverse_drift_diffusion(self, x, t, score):
        # same shape as VE with local scaling
        gamma = self.cfg.cev_gamma
        sig = self.sigma(t).view(-1, *([1] * (x.ndim - 1)))
        scale = torch.exp((gamma - 1) * x)
        eff = sig * scale
        dt = 1e-3
        sig_p = self.sigma((t + dt).clamp(max=1.0)).view_as(sig)
        d_sig2 = ((sig_p * scale) ** 2 - eff ** 2) / dt
        drift = -d_sig2 * score
        g = torch.sqrt(d_sig2.clamp(min=0))
        return drift, g


def build_sde(cfg: SDEConfig) -> SDE:
    mapping = {"ve": VESDE, "vp": VPSDE, "gbm": GBMSDE, "cev": CEVSDE}
    return mapping[cfg.type](cfg)
