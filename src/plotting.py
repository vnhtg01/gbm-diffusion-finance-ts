"""Plot utilities mirroring paper Figures 3, 5, 6, 7, 8."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_return_density(returns_dict: dict[str, np.ndarray], ax=None,
                        n_bins: int = 80) -> None:
    """Log-log histogram of |r| normalized by std (paper Fig. 3a / 8a)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    for name, r in returns_dict.items():
        r = r.ravel()
        r = r[np.isfinite(r)]
        r = r / (r.std() + 1e-12)
        r = np.abs(r)
        r = r[r > 0]
        bins = np.logspace(np.log10(r.min() + 1e-6), np.log10(r.max()), n_bins)
        hist, edges = np.histogram(r, bins=bins, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        mask = hist > 0
        ax.loglog(centers[mask], hist[mask], ".", label=name, markersize=4)
    ax.set_xlabel("Normalized price return")
    ax.set_ylabel("Probability density")
    ax.legend()


def plot_vol_clustering(acf_dict: dict[str, np.ndarray], ax=None) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    for name, acf in acf_dict.items():
        lags = np.arange(1, len(acf) + 1)
        ax.loglog(lags, np.clip(acf, 1e-6, None), ".", label=name, markersize=3)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()


def plot_leverage(lev_dict: dict[str, np.ndarray], ax=None) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    for name, lev in lev_dict.items():
        ax.plot(np.arange(len(lev)), lev, label=name, linewidth=0.9)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("L(k)")
    ax.legend()


def plot_stylized_grid(summaries: dict[str, dict], out_path: str | None = None):
    """summaries: name → {alpha, vol_clustering, leverage}."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ret = {k: v.get("returns") for k, v in summaries.items() if "returns" in v}
    if ret:
        plot_return_density(ret, ax=axes[0])
    plot_vol_clustering({k: v["vol_clustering"] for k, v in summaries.items()}, ax=axes[1])
    plot_leverage({k: v["leverage"] for k, v in summaries.items()}, ax=axes[2])
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
