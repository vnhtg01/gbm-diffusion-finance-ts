#!/usr/bin/env bash
# CPU-scaled reproduction: 3 configs that isolate both effects in the paper.
# For each config: train → generate → evaluate (per-model figure).
# After all 3 configs are done, produce the aggregated comparison figure.
#
#   1. GBM + cosine       — paper's main contribution, best performer
#   2. GBM + exponential  — best α match to empirical (paper Fig. 5)
#   3. VE  + cosine       — baseline: same schedule as (1), isolates SDE effect
#
# With configs/cpu_small.yaml each config takes ~50 min on i5-1145G7.
# Total: ~2.5 h training + ~30 min sampling + ~5 min eval.
#
# The script is resumable: checkpoints, samples, and per-model figures are
# skipped when already present, so re-running after an interruption picks up
# where it stopped.
#
# Usage:
#   bash scripts/reproduce_cpu.sh
#   EPOCHS=50 bash scripts/reproduce_cpu.sh   # quick smoke test

set -euo pipefail

CONFIG="${CONFIG:-configs/cpu_small.yaml}"
EPOCHS="${EPOCHS:-}"
RESULTS_DIR="results"
CKPT_DIR="experiments/checkpoints"
FIG_DIR="results/figures"

mkdir -p "$RESULTS_DIR" "$CKPT_DIR" "$FIG_DIR"

# Force PyTorch to use all 8 threads on the i5-1145G7
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Universe is embedded in the config; pull seq_len from YAML for file naming
SEQ_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['seq_len'])")
STRIDE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['stride'])")
# L=256/S=64 makes the filename unique vs the paper-scale run (L=2048/S=400)
DATA_FILE="data/processed/sp500_L${SEQ_LEN}_S${STRIDE}.npz"
TAG_PREFIX="cpu"              # prefix for checkpoints/samples so they don't collide

# Configs to train: "sde:schedule"
CONFIGS=("gbm:cosine" "gbm:exponential" "ve:cosine")

# Step 1: prepare data
echo "=== [1/2] Preparing dataset (20 oldest tickers) ==="
if [[ ! -f "$DATA_FILE" ]]; then
  python3 -m src.data \
    --universe sp500 \
    --tickers IBM,KO,PG,XOM,GE,MCD,MMM,JNJ,MRK,PFE,CAT,DIS,WMT,BA,CVX,HON,T,F,HD,DD \
    --min-years 30 --seq-len "$SEQ_LEN" --stride "$STRIDE"
fi
ls -la "$DATA_FILE" || { echo "ERROR: data file missing" ; exit 1; }

# Step 2: per-model pipeline (train → generate → evaluate)
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
         --universe sp500 --tag "$tag")
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

# Step 3: aggregate figure
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
  --out "${FIG_DIR}/cpu_reproduction.png"

echo ""
echo "=== DONE ==="
echo "Per-model figures: ${FIG_DIR}/${TAG_PREFIX}_<sde>_<schedule>.png"
echo "Aggregate figure:  ${FIG_DIR}/cpu_reproduction.png"
echo "Compare α printed above with paper (REPRODUCTION.md Table 1)."
