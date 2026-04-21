"""Training entry-point.

Usage:
  python scripts/train.py --config configs/default.yaml \
                          --sde gbm --schedule cosine \
                          --universe sp500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DatasetSpec, load_or_build
from src.diffusion import train
from src.model import ScoreNet
from src.sde import SDEConfig, build_sde


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override(cfg: dict, args) -> dict:
    if args.sde:
        cfg["sde"]["type"] = args.sde
    if args.schedule:
        cfg["sde"]["schedule"] = args.schedule
    if args.universe:
        cfg["data"]["universe"] = args.universe
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sde", default=None, choices=["ve", "vp", "gbm", "cev"])
    ap.add_argument("--schedule", default=None,
                    choices=["linear", "exponential", "cosine"])
    ap.add_argument("--universe", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = override(load_cfg(args.config), args)

    spec = DatasetSpec(
        universe=cfg["data"]["universe"],
        tickers=cfg["data"]["tickers"],
        min_years=cfg["data"]["min_years"],
        seq_len=cfg["data"]["seq_len"],
        stride=cfg["data"]["stride"],
    )
    windows = load_or_build(spec, Path("data/raw"), Path("data/processed"))
    print(f"loaded windows {windows.shape} from {spec.universe}")

    ds = TensorDataset(torch.from_numpy(windows.astype(np.float32)))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        drop_last=True, num_workers=cfg["train"]["num_workers"],
                        collate_fn=lambda batch: torch.stack([b[0] for b in batch]))

    sde_cfg = SDEConfig(
        type=cfg["sde"]["type"],
        schedule=cfg["sde"]["schedule"],
        sigma_min=cfg["sde"]["sigma_min"],
        sigma_max=cfg["sde"]["sigma_max"],
        T=cfg["sde"]["T"],
        cev_gamma=cfg["sde"]["cev_gamma"],
    )
    sde = build_sde(sde_cfg)
    model = ScoreNet(
        channels=cfg["model"]["channels"],
        diff_emb_dim=cfg["model"]["diffusion_emb_dim"],
        feat_emb_dim=cfg["model"]["feature_emb_dim"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
    )

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{spec.universe}_{sde_cfg.type}_{sde_cfg.schedule}"
    ckpt_path = ckpt_dir / f"{tag}.pt"

    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"training on {device} → {ckpt_path}")

    train(model, sde, loader,
          lr=cfg["train"]["lr"],
          epochs=cfg["train"]["epochs"],
          device=device,
          ckpt_path=ckpt_path,
          log_every=cfg["train"]["log_every"],
          grad_clip=cfg["train"]["grad_clip"])


if __name__ == "__main__":
    main()
