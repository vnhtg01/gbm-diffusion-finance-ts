#!/usr/bin/env bash
# Run the full 3x3 reproduction grid from paper Kim, Choi, Kim (2025) §4.
# For each (SDE, schedule) config: train → generate → evaluate (per-model figure).
# After all 9 configs are done, produce the aggregated 3x3 comparison figure.
#
# Rationale: the old version trained the full grid, then generated, then ran
# a single evaluation at the end. If the run was interrupted mid-grid, no
# figure was ever produced. The end-to-end-per-model layout gives intermediate
# results after each ~1.5–2 h on GPU, and remains fully resumable (checkpoints,
# samples, and per-model figures are all skipped when already present).
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
FIG_DIR="results/figures"

mkdir -p "$RESULTS_DIR" "$CKPT_DIR" "$FIG_DIR"

SDES=(ve vp gbm)
SCHEDULES=(linear exponential cosine)
DATA_FILE="data/processed/${UNIVERSE}_L2048_S400.npz"

# Step 1: prepare data (cached if already downloaded)
echo "=== [1/2] Preparing $UNIVERSE dataset ==="
python -m src.data --universe "$UNIVERSE" --min-years 40 --seq-len 2048 --stride 400

# Step 2: per-model pipeline (train → generate → evaluate). Each iteration is
# self-contained: it saves its own checkpoint, samples, and figure before
# moving to the next config.
echo "=== [2/2] Running 3x3 grid end-to-end per model ==="
total=$(( ${#SDES[@]} * ${#SCHEDULES[@]} ))
i=0
for sde in "${SDES[@]}"; do
  for sch in "${SCHEDULES[@]}"; do
    i=$((i + 1))
    tag="${UNIVERSE}_${sde}_${sch}"
    ckpt="${CKPT_DIR}/${tag}.pt"
    samples="${RESULTS_DIR}/samples_${sde}_${sch}.npy"
    fig="${FIG_DIR}/${tag}.png"
    echo ""
    echo "--- [$i/$total] $tag ---"

    # 2a. train (skip if checkpoint exists)
    if [[ -f "$ckpt" ]]; then
      echo "    train: $ckpt already exists, skipping"
    else
      cmd=(python scripts/train.py --config "$CONFIG" --sde "$sde" --schedule "$sch" \
           --universe "$UNIVERSE" --tag "$tag")
      [[ -n "$EPOCHS" ]] && cmd+=(--epochs "$EPOCHS")
      "${cmd[@]}"
    fi

    # 2b. generate (skip if samples exist)
    if [[ -f "$samples" ]]; then
      echo "    sample: $samples already exists, skipping"
    else
      python scripts/generate.py --config "$CONFIG" \
        --ckpt "$ckpt" --n-samples 120 --out "$samples"
    fi

    # 2c. per-model evaluation (always re-run: cheap vs. training and we want
    # an up-to-date figure after every completed model)
    python scripts/evaluate.py \
      --real "$DATA_FILE" \
      --runs "${sde}_${sch}=${samples}" \
      --out "$fig"
    echo "    -> per-model figure: $fig"
  done
done

# Step 3: aggregate figure over the full 3x3 grid
echo ""
echo "=== Aggregating 3x3 comparison figure ==="
RUNS=()
for sde in "${SDES[@]}"; do
  for sch in "${SCHEDULES[@]}"; do
    RUNS+=("${sde}_${sch}=${RESULTS_DIR}/samples_${sde}_${sch}.npy")
  done
done

python scripts/evaluate.py \
  --real "$DATA_FILE" \
  --runs "${RUNS[@]}" \
  --out "${FIG_DIR}/${UNIVERSE}_grid_3x3.png"

echo ""
echo "=== DONE ==="
echo "Per-model figures: ${FIG_DIR}/${UNIVERSE}_<sde>_<schedule>.png"
echo "Grid figure:       ${FIG_DIR}/${UNIVERSE}_grid_3x3.png"
echo "Compare tail exponents with REPRODUCTION.md Table 1."
