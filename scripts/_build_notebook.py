"""Generate notebooks/reproduction.ipynb end-to-end paper reproduction on CPU."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "reproduction.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip()))


md("""
# Tái hiện paper Kim, Choi, Kim (2025) — GBM-based Diffusion cho chuỗi thời gian tài chính

**arXiv:2507.19003** — *A diffusion-based generative model for financial time series via geometric Brownian motion*.

Notebook này chạy end-to-end toàn bộ Pha 1 reproduction **scaled cho CPU** (i5-1145G7):

1. Tải 20 cổ phiếu S&P 500 lịch sử lâu nhất
2. Tính 3 stylized facts trên dữ liệu thật (benchmark)
3. Train 3 configs: `gbm+cosine`, `gbm+exponential`, `ve+cosine`
4. Sinh 120 synthetic series cho mỗi config
5. So sánh metrics với paper

**Thời gian dự kiến**: ~3 giờ trên CPU 8 threads.

**Checkpoint**: mỗi bước cache trên disk — nếu ngắt giữa chừng, chạy lại sẽ skip các bước đã xong.
""")

code(r"""
# Setup
import os, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

# Thread count cho CPU
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
torch.set_num_threads(8)

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
    os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.data import DatasetSpec, load_or_build
from src.sde import SDEConfig, build_sde
from src.model import ScoreNet
from src.diffusion import dsm_loss, train, sample
from src.stylized_facts import summarize, tail_exponent
from src.plotting import plot_return_density, plot_vol_clustering, plot_leverage

print(f"root: {ROOT}")
print(f"device: cpu | threads: {torch.get_num_threads()}")
""")

md("## 1. Load config CPU-scaled")

code(r"""
with open("configs/cpu_small.yaml") as f:
    cfg = yaml.safe_load(f)

print("Universe:", cfg["data"]["universe"], "| tickers:", len(cfg["data"]["tickers"]))
print("Seq len:", cfg["data"]["seq_len"], "| stride:", cfg["data"]["stride"])
print("Model: C={channels} De={diffusion_emb_dim} Df={feature_emb_dim} layers={n_layers}".format(**cfg["model"]))
print("Train:", cfg["train"]["epochs"], "epochs |", cfg["train"]["batch_size"], "batch")
""")

md("## 2. Tải dữ liệu S&P 500 (20 ticker lịch sử dài, ≥30 năm)")

code(r"""
spec = DatasetSpec(
    universe=cfg["data"]["universe"],
    tickers=cfg["data"]["tickers"],
    min_years=cfg["data"]["min_years"],
    seq_len=cfg["data"]["seq_len"],
    stride=cfg["data"]["stride"],
)
windows = load_or_build(spec, Path("data/raw"), Path("data/processed"))
print(f"Real dataset: {windows.shape}  (N_windows, L)")
print(f"Mean |r| = {np.mean(np.abs(windows)):.4f}, std = {windows.std():.4f}")
""")

md("## 3. Stylized facts trên dữ liệu THẬT (benchmark)")

code(r"""
real_summary = summarize(
    windows,
    tail_fraction=cfg["eval"]["tail_fraction"],
    max_lag_vc=cfg["eval"]["max_lag_vc"],
    max_lag_lev=cfg["eval"]["max_lag_lev"],
)
print(f"Tail exponent α (real) = {real_summary['alpha']:.2f}   (paper S&P500 benchmark ≈ 4.35)")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
plot_return_density({"real": windows}, ax=axes[0])
plot_vol_clustering({"real": real_summary["vol_clustering"]}, ax=axes[1])
plot_leverage({"real": real_summary["leverage"]}, ax=axes[2])
axes[0].set_title("(a) Heavy-tailed returns")
axes[1].set_title("(b) Volatility clustering")
axes[2].set_title("(c) Leverage effect")
fig.tight_layout()
plt.show()
""")

md("""
## 4. Train 3 configs

| Config | Ý nghĩa |
|---|---|
| `gbm + cosine` | Đóng góp chính của paper |
| `gbm + exponential` | Best α match với empirical |
| `ve + cosine` | Baseline (cùng schedule) — isolate SDE effect |

Mỗi config ~50 phút trên CPU 8 thread. **Nếu đã có checkpoint, skip.**
""")

code(r"""
from torch.utils.data import DataLoader, TensorDataset

def train_one(sde_type: str, schedule: str, cfg: dict, windows: np.ndarray,
              epochs: int | None = None):
    tag = f"cpu_{sde_type}_{schedule}"
    ckpt_path = Path(cfg["train"]["ckpt_dir"]) / f"{tag}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists():
        print(f"[{tag}] checkpoint exists, skipping training")
        return ckpt_path

    ds = TensorDataset(torch.from_numpy(windows.astype(np.float32)))
    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True,
        num_workers=0,  # 0 for CPU — avoids multiprocessing overhead
        collate_fn=lambda batch: torch.stack([b[0] for b in batch]),
    )
    sde = build_sde(SDEConfig(
        type=sde_type, schedule=schedule,
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
    print(f"[{tag}] training {epochs or cfg['train']['epochs']} epochs...")
    t0 = time.time()
    train(
        model, sde, loader,
        lr=cfg["train"]["lr"],
        epochs=epochs or cfg["train"]["epochs"],
        device="cpu",
        ckpt_path=ckpt_path,
        log_every=cfg["train"]["log_every"],
        grad_clip=cfg["train"]["grad_clip"],
    )
    print(f"[{tag}] DONE in {(time.time()-t0)/60:.1f} min -> {ckpt_path}")
    return ckpt_path
""")

code(r"""
# Config 1: GBM + cosine
ckpt_gbm_cos = train_one("gbm", "cosine", cfg, windows)
""")

code(r"""
# Config 2: GBM + exponential
ckpt_gbm_exp = train_one("gbm", "exponential", cfg, windows)
""")

code(r"""
# Config 3: VE + cosine (baseline)
ckpt_ve_cos = train_one("ve", "cosine", cfg, windows)
""")

md("## 5. Sinh synthetic series (120 mẫu × 500 reverse steps)")

code(r"""
def sample_one(sde_type: str, schedule: str, ckpt_path: Path, cfg: dict,
               n_samples: int | None = None) -> np.ndarray:
    tag = f"cpu_{sde_type}_{schedule}"
    out_path = Path("results") / f"samples_{tag}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[{tag}] samples exist, loading")
        return np.load(out_path)

    sde = build_sde(SDEConfig(
        type=sde_type, schedule=schedule,
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
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])

    n = n_samples or cfg["eval"]["n_samples"]
    L = cfg["data"]["seq_len"]
    print(f"[{tag}] sampling {n}x{L} with {cfg['sde']['N_reverse']} reverse steps...")
    t0 = time.time()
    out = sample(model, sde, (n, 1, L),
                 n_steps=cfg["sde"]["N_reverse"], device="cpu")
    arr = out.squeeze(1).numpy()
    np.save(out_path, arr)
    print(f"[{tag}] DONE in {(time.time()-t0)/60:.1f} min -> {out_path}")
    return arr
""")

code(r"""
samples = {
    "gbm_cosine":      sample_one("gbm", "cosine",      ckpt_gbm_cos, cfg),
    "gbm_exponential": sample_one("gbm", "exponential", ckpt_gbm_exp, cfg),
    "ve_cosine":       sample_one("ve",  "cosine",      ckpt_ve_cos,  cfg),
}
for k, v in samples.items():
    print(f"{k:20s} {v.shape}  mean|r|={np.abs(v).mean():.4f}  std={v.std():.4f}")
""")

md("## 6. Đánh giá: tính 3 stylized facts cho mỗi sample set")

code(r"""
summaries = {"real": {**real_summary, "returns": windows}}
for name, arr in samples.items():
    s = summarize(
        arr,
        tail_fraction=cfg["eval"]["tail_fraction"],
        max_lag_vc=cfg["eval"]["max_lag_vc"],
        max_lag_lev=cfg["eval"]["max_lag_lev"],
    )
    s["returns"] = arr
    summaries[name] = s
    print(f"{name:20s} α = {s['alpha']:.2f}")
""")

md("## 7. So sánh với benchmark paper (Table 1 — Figure 5)")

code(r"""
PAPER_ALPHA = {
    "real":            4.35,
    "gbm_cosine":      3.78,
    "gbm_exponential": 4.62,
    "ve_cosine":       4.14,
}

print(f"{'config':<20s}{'α (ours)':>12s}{'α (paper)':>12s}{'Δ':>10s}")
print("-" * 54)
for name, s in summaries.items():
    ref = PAPER_ALPHA.get(name)
    our = s["alpha"]
    delta = f"{our - ref:+.2f}" if ref is not None else "n/a"
    ref_s = f"{ref:.2f}" if ref is not None else "n/a"
    print(f"{name:<20s}{our:>12.2f}{ref_s:>12s}{delta:>10s}")
""")

md("## 8. Plot tổng hợp 3 stylized facts — so sánh model vs real")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Return density log-log
ret_dict = {name: s["returns"] for name, s in summaries.items()}
plot_return_density(ret_dict, ax=axes[0])
axes[0].set_title("(a) Heavy-tailed return distribution")

# (b) Volatility clustering
vc_dict = {name: s["vol_clustering"] for name, s in summaries.items()}
plot_vol_clustering(vc_dict, ax=axes[1])
axes[1].set_title("(b) Volatility clustering (ACF of |r|)")

# (c) Leverage effect
lev_dict = {name: s["leverage"] for name, s in summaries.items()}
plot_leverage(lev_dict, ax=axes[2])
axes[2].set_title("(c) Leverage effect L(k)")

fig.tight_layout()
fig_path = Path("results/figures/cpu_reproduction.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved -> {fig_path}")
plt.show()
""")

md("""
## 9. Kết luận Pha 1

Dựa trên bảng α và biểu đồ trên, các điểm cần kiểm tra (**REPRODUCTION.md §'Khi nào xem Pha 1 là DONE'**):

- [ ] α của dữ liệu real trong [3.8, 4.9]
- [ ] Ít nhất 1 cấu hình GBM có α trong [3, 5]
- [ ] Volatility clustering plot: GBM decay gradual, VE collapse sớm hơn
- [ ] Leverage plot: GBM âm và persistent, VE oscillate quanh 0

**Nếu đạt đủ cả 4** → Pha 1 OK, chuyển sang Pha 2 (extension: crypto hoặc CEV).

**Nếu không đạt**:
- α lệch nhiều → tăng epochs, kiểm tra loss có hội tụ không
- Pattern sai → kiểm tra `src/sde.py` bằng unit test trong REPRODUCTION.md

Ghi chú trong report: do scale CPU (L=256 thay vì 2048, model 10x nhỏ hơn), số liệu tuyệt đối lệch paper là **expected**. Điều quan trọng là **pattern định tính khớp** (GBM > VE, exponential/cosine > linear).
""")


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open(OUT, "w") as f:
    nbf.write(nb, f)

print(f"Wrote {OUT} with {len(cells)} cells")
