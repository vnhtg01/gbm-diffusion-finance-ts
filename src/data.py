"""Data pipeline: download via yfinance, compute log-returns, sliding windows.

Paper §3.1.2: S&P 500 constituents with >=40y history, daily adjusted close,
log-returns r_t = log(p_t/p_{t-1}), sliding windows of length L with stride S.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_GITHUB_CSV = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def fetch_sp500_tickers(source: str = "github") -> list[str]:
    """Fetch current S&P 500 tickers.

    source:
      - "github" (default, stable): datasets/s-and-p-500-companies CSV
      - "wikipedia" (like paper, can 403 without User-Agent)
    """
    if source == "github":
        df = pd.read_csv(SP500_GITHUB_CSV)
    else:
        html = _http_get(SP500_WIKI_URL)
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
    tickers = df["Symbol"].astype(str).tolist()
    # Paper §3.1.2: exclude tickers with non-standard symbols (e.g., BRK.B, BF.B).
    return [t for t in tickers if "." not in t]


def download_prices(tickers: list[str], start: str = "1980-01-01") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def filter_by_history(prices: pd.DataFrame, min_years: int) -> pd.DataFrame:
    min_days = min_years * 252
    keep = prices.notna().sum() >= min_days
    return prices.loc[:, keep]


def sliding_windows(series: np.ndarray, L: int, stride: int) -> np.ndarray:
    """series shape (T,) → (n_windows, L)."""
    n = (len(series) - L) // stride + 1
    if n <= 0:
        return np.empty((0, L))
    out = np.stack([series[i * stride : i * stride + L] for i in range(n)])
    return out


def build_dataset(prices: pd.DataFrame, L: int, stride: int) -> np.ndarray:
    """Return (N, L) array of log-return windows, across tickers."""
    lr = log_returns(prices)
    windows = []
    for col in lr.columns:
        s = lr[col].dropna().to_numpy()
        if len(s) >= L:
            windows.append(sliding_windows(s, L, stride))
    return np.concatenate(windows, axis=0) if windows else np.empty((0, L))


@dataclass
class DatasetSpec:
    universe: str
    tickers: list[str] | None
    min_years: int
    seq_len: int
    stride: int


def load_or_build(spec: DatasetSpec, raw_dir: Path, processed_dir: Path) -> np.ndarray:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{spec.universe}.parquet"
    proc_path = processed_dir / f"{spec.universe}_L{spec.seq_len}_S{spec.stride}.npz"

    if proc_path.exists():
        return np.load(proc_path)["windows"]

    if raw_path.exists():
        prices = pd.read_parquet(raw_path)
    else:
        if spec.universe == "sp500":
            tickers = spec.tickers or fetch_sp500_tickers()
        elif spec.universe == "crypto":
            tickers = spec.tickers or ["BTC-USD", "ETH-USD"]
        elif spec.universe == "fx":
            tickers = spec.tickers or ["EURUSD=X", "USDJPY=X", "GBPUSD=X"]
        elif spec.universe == "commodities":
            tickers = spec.tickers or ["GC=F", "CL=F", "SI=F"]
        else:
            raise ValueError(f"Unknown universe: {spec.universe}")
        prices = download_prices(tickers)
        prices.to_parquet(raw_path)

    prices = filter_by_history(prices, spec.min_years) if spec.min_years > 0 else prices
    windows = build_dataset(prices, spec.seq_len, spec.stride)
    np.savez_compressed(proc_path, windows=windows)
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500",
                    choices=["sp500", "crypto", "fx", "commodities"])
    ap.add_argument("--tickers", default=None, help="comma-separated override")
    ap.add_argument("--min-years", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=400)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()

    spec = DatasetSpec(
        universe=args.universe,
        tickers=args.tickers.split(",") if args.tickers else None,
        min_years=args.min_years,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    w = load_or_build(spec, Path(args.raw_dir), Path(args.processed_dir))
    print(f"{args.universe}: {w.shape} windows")


if __name__ == "__main__":
    main()
