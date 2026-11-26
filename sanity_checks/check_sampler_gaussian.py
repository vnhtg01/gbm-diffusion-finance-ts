"""Sanity check of the reverse samplers in src/sde.py with an ANALYTIC score.

Idea: if the data are x0 ~ N(0, s0^2), the exact score of the training marginal
p_t is known in closed form. Plugging that exact score into a correct reverse
sampler must give back std ~= s0. Any deviation comes from the sampler itself,
not from the neural network.

Run:  python sanity_checks/check_sampler_gaussian.py      (numpy only, ~10 s)
"""
import math
import numpy as np

rng = np.random.default_rng(0)
SMIN, SMAX, T, S0, B, N = 0.01, 1.0, 1.0, 1.0, 20000, 2000


def sig(t, kind):
    if kind == "cosine":
        return SMIN + (SMAX - SMIN) * (1 - np.cos(math.pi * t)) / 2
    if kind == "exponential":
        return SMIN * (SMAX / SMIN) ** t
    return np.sqrt(SMIN ** 2 + t * (SMAX ** 2 - SMIN ** 2))


def intvar(t, kind, n=2000):            # same as integrated_variance in src/sde.py
    s = np.linspace(0, t, n)
    return np.trapezoid(sig(s, kind) ** 2, s)


def ve_sampler(kind, g2_rule, init):
    ts = np.linspace(T, 1e-3, N + 1)
    vT = intvar(T, kind)
    x = (np.sqrt(vT) if init == "repo" else np.sqrt(S0 ** 2 + vT)) * rng.standard_normal(B)
    for i in range(N):
        t, dt = ts[i], ts[i + 1] - ts[i]
        score = -x / (S0 ** 2 + intvar(t, kind))                 # exact score of training marginal
        if g2_rule == "repo":                                    # d(sigma_t^2)/dt, as in VESDE
            tp = min(t + 1e-3, 1.0)
            g2 = max((sig(tp, kind) ** 2 - sig(t, kind) ** 2) / 1e-3, 0.0)
        else:                                                    # g^2 = sigma_t^2, consistent with marginal
            g2 = sig(t, kind) ** 2
        x = x - g2 * score * dt + np.sqrt(g2 * -dt) * rng.standard_normal(B)
    return x.std()


def vp_sampler(kind, sign_as_repo):
    ts = np.linspace(T, 1e-3, N + 1)
    x = rng.standard_normal(B)
    for i in range(N):
        t, dt = ts[i], ts[i + 1] - ts[i]
        a = math.exp(-intvar(t, kind))
        score = -x / (a * S0 ** 2 + 1 - a)
        g2 = sig(t, kind) ** 2
        drift = g2 * (score - 0.5 * x) if sign_as_repo else -g2 * (score + 0.5 * x)
        x = x + drift * dt + np.sqrt(g2 * -dt) * rng.standard_normal(B)
    return x.std()


if __name__ == "__main__":
    print(f"target std = {S0}")
    for k in ["linear", "exponential", "cosine"]:
        print(f"VE/GBM {k:11s} int.var(T)={intvar(T, k):.3f} | repo sampler: {ve_sampler(k, 'repo', 'repo'):.3f}"
              f" | repo drift + full prior: {ve_sampler(k, 'repo', 'full'):.3f}"
              f" | consistent drift + full prior: {ve_sampler(k, 'ok', 'full'):.3f}")
    print(f"VP cosine: repo sign {vp_sampler('cosine', True):.3f} | corrected sign {vp_sampler('cosine', False):.3f}")
