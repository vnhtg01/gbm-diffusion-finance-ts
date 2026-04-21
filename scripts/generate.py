"""Generate synthetic log-return series from a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion import sample
from src.model import ScoreNet
from src.sde import SDEConfig, build_sde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-samples", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sde = build_sde(SDEConfig(
        type=cfg["sde"]["type"], schedule=cfg["sde"]["schedule"],
        sigma_min=cfg["sde"]["sigma_min"], sigma_max=cfg["sde"]["sigma_max"],
        T=cfg["sde"]["T"], cev_gamma=cfg["sde"]["cev_gamma"],
    ))
    model = ScoreNet(
        channels=cfg["model"]["channels"],
        diff_emb_dim=cfg["model"]["diffusion_emb_dim"],
        feat_emb_dim=cfg["model"]["feature_emb_dim"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
    )
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = cfg["data"]["seq_len"]
    out = sample(model, sde, (args.n_samples, 1, L),
                 n_steps=cfg["sde"]["N_reverse"], device=device)
    out_np = out.squeeze(1).numpy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out_np)
    print(f"saved {out_np.shape} → {args.out}")


if __name__ == "__main__":
    main()
