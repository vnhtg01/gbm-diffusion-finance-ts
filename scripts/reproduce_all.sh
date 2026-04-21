#!/usr/bin/env bash
# Run the full 3x3 reproduction grid from paper Kim, Choi, Kim (2025) §4.
# Trains VE/VP/GBM x linear/exponential/cosine, then generates + evaluates.
#
# WARNING: 9 models × 1000 epochs × L=2048 can take MANY GPU-days.
# For a quick sanity check, override epochs via EPOCHS env var:
#   EPOCHS=50 bash scripts/reproduce_all.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/paper.yaml}"
EPOCHS="${EPOCHS:-}"
UNIVERSE="${UNIVERSE:-sp500}"
RESULTS_DIR="results"
CKPT_DIR="experiments/checkpoints"

mkdir -p "$RESULTS_DIR" "$CKPT_DIR" results/figures

SDES=(ve vp gbm)
SCHEDULES=(linear exponential cosine)

# Step 1: prepare data (cached if already downloaded)
echo "=== [1/4] Preparing $UNIVERSE dataset ==="
python -m src.data --universe "$UNIVERSE" --min-years 40 --seq-len 2048 --stride 400

# Step 2: train the 9 models
echo "=== [2/4] Training 3x3 grid ==="
for sde in "${SDES[@]}"; do
  for sch in "${SCHEDULES[@]}"; do
    tag="${UNIVERSE}_${sde}_${sch}"
    ckpt="${CKPT_DIR}/${tag}.pt"
    if [[ -f "$ckpt" ]]; then
      echo "    -> $tag already trained, skipping"
      continue
    fi
    echo "--- training $tag ---"
    cmd=(python scripts/train.py --config "$CONFIG" --sde "$sde" --schedule "$sch" \
         --universe "$UNIVERSE" --tag "$tag")
    if [[ -n "$EPOCHS" ]]; then
      cmd+=(--epochs "$EPOCHS")
    fi
    "${cmd[@]}"
  done
done

# Step 3: generate samples
echo "=== [3/4] Generating 120 samples per model ==="
for sde in "${SDES[@]}"; do
  for sch in "${SCHEDULES[@]}"; do
    tag="${UNIVERSE}_${sde}_${sch}"
    out="${RESULTS_DIR}/samples_${sde}_${sch}.npy"
    if [[ -f "$out" ]]; then
      echo "    -> $tag samples exist, skipping"
      continue
    fi
    python scripts/generate.py --config "$CONFIG" \
      --ckpt "${CKPT_DIR}/${tag}.pt" --n-samples 120 --out "$out"
  done
done

# Step 4: evaluate grid
echo "=== [4/4] Evaluating stylized facts ==="
RUNS=()
for sde in "${SDES[@]}"; do
  for sch in "${SCHEDULES[@]}"; do
    RUNS+=("${sde}_${sch}=${RESULTS_DIR}/samples_${sde}_${sch}.npy")
  done
done

python scripts/evaluate.py \
  --real "data/processed/${UNIVERSE}_L2048_S400.npz" \
  --runs "${RUNS[@]}" \
  --out "results/figures/${UNIVERSE}_grid_3x3.png"

echo ""
echo "=== DONE ==="
echo "Figure: results/figures/${UNIVERSE}_grid_3x3.png"
echo "Compare tail exponents with REPRODUCTION.md Table 1."
