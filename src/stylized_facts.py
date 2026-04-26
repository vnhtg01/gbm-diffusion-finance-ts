"""Stylized-fact metrics (paper §2.2, §4).

(i) Heavy-tail tail exponent α:
    fit power law P(|r|) ~ |r|^{-α} on the upper tail (top q% of |r|).
    Using powerlaw package if available, else MLE on the Pareto tail.

(ii) Volatility clustering:
    autocorrelation of |r_t| for lag k = 1..max_lag. Decays ~ k^{-β}
    with β ∈ [0.1, 0.5] in real data.

(iii) Leverage effect:
    L(k) = E[r_t (r_{t+k})^2 - r_t |r_t|^2] / E[|r_t|^2]^2   (paper eq. §2.2)
    Negative and slowly decaying to 0.
"""
from __future__ import annotations

import numpy as np


def tail_exponent(returns: np.ndarray, tail_fraction: float = 0.05) -> float:
    """Return α estimated on the top-|r| tail. Accepts 1D or 2D (concat 2D)."""
    r = np.abs(returns.ravel())
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) == 0:
        return np.nan
    r_sorted = np.sort(r)
    k = max(10, int(len(r) * tail_fraction))
    tail = r_sorted[-k:]
    xmin = tail[0]
    try:
        import powerlaw
        fit = powerlaw.Fit(tail, xmin=xmin, verbose=False)
        return float(fit.alpha)
    except Exception:
        # Hill estimator
        return float(1 + k / np.sum(np.log(tail / xmin + 1e-12)))


def autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    if lag == 0:
        return 1.0
    num = np.mean(x[:-lag] * x[lag:])
    den = np.var(x)
    return float(num / (den + 1e-12))


def volatility_clustering(returns: np.ndarray, max_lag: int = 1000) -> np.ndarray:
    """ACF of |r| up to max_lag. Input (N, L) or (L,)."""
    if returns.ndim == 1:
        r = np.abs(returns)
        return np.array([autocorr(r, k) for k in range(1, max_lag + 1)])
    # averaging over series
    lags = np.arange(1, max_lag + 1)
    acfs = []
    for row in returns:
        r = np.abs(row)
        acfs.append([autocorr(r, k) for k in lags])
    return np.nanmean(np.array(acfs), axis=0)


def leverage_effect(returns: np.ndarray, max_lag: int = 100) -> np.ndarray:
    if returns.ndim == 1:
        returns = returns[None, :]

    L_vals = np.zeros(max_lag + 1)

    for row in returns:
        r = row - row.mean()
        denom = (r**2).mean()**2

        base_term = (r * r**2).mean()

        for k in range(max_lag + 1):
            if k == 0:
                term = base_term
            else:
                term = (r[:-k] * (r[k:]**2)).mean()

            L_vals[k] += (term - base_term) / (denom + 1e-12)

    return L_vals / returns.shape[0]

def summarize(returns: np.ndarray, tail_fraction: float = 0.05,
              max_lag_vc: int = 1000, max_lag_lev: int = 100) -> dict:
    return {
        "alpha": tail_exponent(returns, tail_fraction),
        "vol_clustering": volatility_clustering(returns, max_lag_vc),
        "leverage": leverage_effect(returns, max_lag_lev),
    }
