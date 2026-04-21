"""Compute 3 stylized facts for a set of runs + real data, render comparison figures.

Example:
  python scripts/evaluate.py \
    --real data/processed/sp500_L2048_S400.npz \
    --runs gbm_cosine=results/samples_gbm_cosine.npy \
           ve_linear=results/samples_ve_linear.npy \
    --out results/figures/stylized.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import plot_stylized_grid
from src.stylized_facts import summarize


# Paper benchmarks (Kim, Choi, Kim 2025 §4.2, Figure 5). Keys: (sde, schedule).
PAPER_ALPHA = {
    ("real", "sp500"): 4.35,
    ("ve", "linear"): 8.96, ("ve", "exponential"): 8.49, ("ve", "cosine"): 4.14,
    ("gbm", "linear"): 3.06, ("gbm", "exponential"): 4.62, ("gbm", "cosine"): 3.78,
}


def load_array(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        return np.load(path)["windows"]
    return np.load(path)


def paper_alpha_for(name: str) -> float | None:
    parts = name.split("_", 1)
    if len(parts) == 2:
        return PAPER_ALPHA.get((parts[0], parts[1]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--runs", nargs="+", default=[],
                    help="name=path pairs")
    ap.add_argument("--tail-fraction", type=float, default=0.05)
    ap.add_argument("--max-lag-vc", type=int, default=1000)
    ap.add_argument("--max-lag-lev", type=int, default=100)
    ap.add_argument("--out", default="results/figures/stylized.png")
    args = ap.parse_args()

    summaries = {}
    real = load_array(args.real)
    s_real = summarize(real, args.tail_fraction, args.max_lag_vc, args.max_lag_lev)
    s_real["returns"] = real
    summaries["real"] = s_real
    print(f"\n{'name':<20s} {'α (ours)':>10s} {'α (paper)':>10s} {'Δ':>8s}")
    print("-" * 52)
    print(f"{'real sp500':<20s} {s_real['alpha']:>10.2f} {PAPER_ALPHA[('real','sp500')]:>10.2f} "
          f"{s_real['alpha'] - PAPER_ALPHA[('real','sp500')]:>+8.2f}")

    for spec in args.runs:
        name, path = spec.split("=", 1)
        arr = load_array(path)
        s = summarize(arr, args.tail_fraction, args.max_lag_vc, args.max_lag_lev)
        s["returns"] = arr
        summaries[name] = s
        ref = paper_alpha_for(name)
        ref_str = f"{ref:>10.2f}" if ref is not None else f"{'n/a':>10s}"
        delta_str = f"{s['alpha'] - ref:>+8.2f}" if ref is not None else f"{'n/a':>8s}"
        print(f"{name:<20s} {s['alpha']:>10.2f} {ref_str} {delta_str}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_stylized_grid(summaries, out_path=args.out)
    print(f"saved figure → {args.out}")


if __name__ == "__main__":
    main()
