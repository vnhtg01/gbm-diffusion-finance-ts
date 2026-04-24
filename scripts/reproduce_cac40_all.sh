#!/usr/bin/env bash
# Paper-scale GPU reproduction on CAC 40 (French blue-chips).
# Mirrors scripts/reproduce_all.sh but hard-codes universe=cac40 and reads
# configs/paper_cac40.yaml. Produces its own namespaced checkpoints, samples,
# and figures so it coexists with S&P 500 runs without collision.
#
# WARNING: 9 models × 1000 epochs × L=2048 → several GPU-days.
# For a quick sanity check:  EPOCHS=50 bash scripts/reproduce_cac40_all.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/paper_cac40.yaml}"
EPOCHS="${EPOCHS:-}"
UNIVERSE="cac40"
RESULTS_DIR="results"
CKPT_DIR="experiments/checkpoints"
FIG_DIR="results/figures"

mkdir -p "$RESULTS_DIR" "$CKPT_DIR" "$FIG_DIR"

SDES=(ve vp gbm)
SCHEDULES=(linear exponential cosine)

SEQ_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['seq_len'])")
STRIDE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['stride'])")
MIN_YEARS=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['min_years'])")
DATA_FILE="data/processed/${UNIVERSE}_L${SEQ_LEN}_S${STRIDE}.npz"

echo "=== [1/2] Preparing $UNIVERSE dataset (min_years=$MIN_YEARS) ==="
python3 -m src.data --universe "$UNIVERSE" --min-years "$MIN_YEARS" \
  --seq-len "$SEQ_LEN" --stride "$STRIDE"

echo "=== [2/2] Running 3x3 grid end-to-end per model ==="
total=$(( ${#SDES[@]} * ${#SCHEDULES[@]} ))
i=0
for sde in "${SDES[@]}"; do
  for sch in "${SCHEDULES[@]}"; do
    i=$((i + 1))
    tag="${UNIVERSE}_${sde}_${sch}"
    ckpt="${CKPT_DIR}/${tag}.pt"
    samples="${RESULTS_DIR}/samples_${tag}.npy"
    fig="${FIG_DIR}/${tag}.png"
    echo ""
    echo "--- [$i/$total] $tag ---"

    if [[ -f "$ckpt" ]]; then
      echo "    train: $ckpt already exists, skipping"
    else
      cmd=(python3 scripts/train.py --config "$CONFIG" --sde "$sde" --schedule "$sch" \
           --universe "$UNIVERSE" --tag "$tag")
      [[ -n "$EPOCHS" ]] && cmd+=(--epochs "$EPOCHS")
      "${cmd[@]}"
    fi

    if [[ -f "$samples" ]]; then
      echo "    sample: $samples already exists, skipping"
    else
      python3 scripts/generate.py --config "$CONFIG" \
        --ckpt "$ckpt" --n-samples 120 --out "$samples"
    fi

    python3 scripts/evaluate.py \
      --real "$DATA_FILE" \
      --runs "${sde}_${sch}=${samples}" \
      --out "$fig"
    echo "    -> per-model figure: $fig"
  done
done

echo ""
echo "=== Aggregating 3x3 comparison figure ==="
RUNS=()
for sde in "${SDES[@]}"; do
  for sch in "${SCHEDULES[@]}"; do
    RUNS+=("${sde}_${sch}=${RESULTS_DIR}/samples_${UNIVERSE}_${sde}_${sch}.npy")
  done
done

python3 scripts/evaluate.py \
  --real "$DATA_FILE" \
  --runs "${RUNS[@]}" \
  --out "${FIG_DIR}/${UNIVERSE}_grid_3x3.png"

echo ""
echo "=== DONE ==="
echo "Per-model figures: ${FIG_DIR}/${UNIVERSE}_<sde>_<schedule>.png"
echo "Grid figure:       ${FIG_DIR}/${UNIVERSE}_grid_3x3.png"
