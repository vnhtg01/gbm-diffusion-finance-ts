#!/usr/bin/env bash
# CPU-scaled reproduction on CAC 40 (French blue-chips).
# Same 3 configs as scripts/reproduce_cpu.sh but with universe=cac40
# and configs/cpu_cac40.yaml.
#
#   1. GBM + cosine
#   2. GBM + exponential
#   3. VE  + cosine
#
# Usage:
#   bash scripts/reproduce_cac40.sh
#   EPOCHS=50 bash scripts/reproduce_cac40.sh   # quick smoke test

set -euo pipefail

CONFIG="${CONFIG:-configs/cpu_cac40.yaml}"
EPOCHS="${EPOCHS:-}"
UNIVERSE="cac40"
RESULTS_DIR="results"
CKPT_DIR="experiments/checkpoints"
FIG_DIR="results/figures"

mkdir -p "$RESULTS_DIR" "$CKPT_DIR" "$FIG_DIR"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

SEQ_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['seq_len'])")
STRIDE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['stride'])")
MIN_YEARS=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['min_years'])")
TICKERS=$(python3 -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG'))['data']['tickers']))")
DATA_FILE="data/processed/${UNIVERSE}_L${SEQ_LEN}_S${STRIDE}.npz"
TAG_PREFIX="cac40"

CONFIGS=("gbm:cosine" "gbm:exponential" "ve:cosine")

echo "=== [1/2] Preparing CAC 40 dataset ==="
if [[ ! -f "$DATA_FILE" ]]; then
  python3 -m src.data \
    --universe "$UNIVERSE" \
    --tickers "$TICKERS" \
    --min-years "$MIN_YEARS" --seq-len "$SEQ_LEN" --stride "$STRIDE"
fi
ls -la "$DATA_FILE" || { echo "ERROR: data file missing" ; exit 1; }

echo "=== [2/2] Running 3 configs end-to-end per model ==="
total="${#CONFIGS[@]}"
i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i + 1))
  sde="${cfg%%:*}"; sch="${cfg##*:}"
  tag="${TAG_PREFIX}_${sde}_${sch}"
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
    --max-lag-vc 128 --max-lag-lev 64 \
    --out "$fig"
  echo "    -> per-model figure: $fig"
done

echo ""
echo "=== Aggregating comparison figure ==="
RUNS=()
for cfg in "${CONFIGS[@]}"; do
  sde="${cfg%%:*}"; sch="${cfg##*:}"
  RUNS+=("${sde}_${sch}=${RESULTS_DIR}/samples_${TAG_PREFIX}_${sde}_${sch}.npy")
done

python3 scripts/evaluate.py \
  --real "$DATA_FILE" \
  --runs "${RUNS[@]}" \
  --max-lag-vc 128 --max-lag-lev 64 \
  --out "${FIG_DIR}/cac40_reproduction.png"

echo ""
echo "=== DONE ==="
echo "Per-model figures: ${FIG_DIR}/${TAG_PREFIX}_<sde>_<schedule>.png"
echo "Aggregate figure:  ${FIG_DIR}/cac40_reproduction.png"
