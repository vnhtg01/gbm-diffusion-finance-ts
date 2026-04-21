#!/usr/bin/env bash
# CPU-scaled reproduction: 3 configs that isolate both effects in the paper.
#
#   1. GBM + cosine       — paper's main contribution, best performer
#   2. GBM + exponential  — best α match to empirical (paper Fig. 5)
#   3. VE  + cosine       — baseline: same schedule as (1), isolates SDE effect
#
# With configs/cpu_small.yaml each config takes ~50 min on i5-1145G7.
# Total: ~2.5 h training + ~30 min sampling + ~5 min eval.
#
# Usage:
#   bash scripts/reproduce_cpu.sh
#   EPOCHS=50 bash scripts/reproduce_cpu.sh   # quick smoke test

set -euo pipefail

CONFIG="${CONFIG:-configs/cpu_small.yaml}"
EPOCHS="${EPOCHS:-}"
RESULTS_DIR="results"
CKPT_DIR="experiments/checkpoints"

mkdir -p "$RESULTS_DIR" "$CKPT_DIR" results/figures

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
echo "=== [1/4] Preparing dataset (20 oldest tickers) ==="
if [[ ! -f "$DATA_FILE" ]]; then
  python3 -m src.data \
    --universe sp500 \
    --tickers IBM,KO,PG,XOM,GE,MCD,MMM,JNJ,MRK,PFE,CAT,DIS,WMT,BA,CVX,HON,T,F,HD,DD \
    --min-years 30 --seq-len "$SEQ_LEN" --stride "$STRIDE"
fi
ls -la "$DATA_FILE" || { echo "ERROR: data file missing" ; exit 1; }

# Step 2: train
echo "=== [2/4] Training 3 configs ==="
for cfg in "${CONFIGS[@]}"; do
  sde="${cfg%%:*}"; sch="${cfg##*:}"
  tag="${TAG_PREFIX}_${sde}_${sch}"
  ckpt="${CKPT_DIR}/${tag}.pt"
  if [[ -f "$ckpt" ]]; then
    echo "    -> $tag already trained, skipping"
    continue
  fi
  echo "--- training $tag ---"
  cmd=(python3 scripts/train.py --config "$CONFIG" --sde "$sde" --schedule "$sch" \
       --universe sp500 --tag "$tag")
  [[ -n "$EPOCHS" ]] && cmd+=(--epochs "$EPOCHS")
  "${cmd[@]}"
done

# Step 3: sample
echo "=== [3/4] Generating samples ==="
for cfg in "${CONFIGS[@]}"; do
  sde="${cfg%%:*}"; sch="${cfg##*:}"
  tag="${TAG_PREFIX}_${sde}_${sch}"
  out="${RESULTS_DIR}/samples_${tag}.npy"
  if [[ -f "$out" ]]; then
    echo "    -> $tag samples exist, skipping"
    continue
  fi
  python3 scripts/generate.py --config "$CONFIG" \
    --ckpt "${CKPT_DIR}/${tag}.pt" --n-samples 120 --out "$out"
done

# Step 4: evaluate
echo "=== [4/4] Evaluating stylized facts ==="
RUNS=()
for cfg in "${CONFIGS[@]}"; do
  sde="${cfg%%:*}"; sch="${cfg##*:}"
  RUNS+=("${sde}_${sch}=${RESULTS_DIR}/samples_${TAG_PREFIX}_${sde}_${sch}.npy")
done

python3 scripts/evaluate.py \
  --real "$DATA_FILE" \
  --runs "${RUNS[@]}" \
  --max-lag-vc 128 --max-lag-lev 64 \
  --out "results/figures/cpu_reproduction.png"

echo ""
echo "=== DONE ==="
echo "Figure: results/figures/cpu_reproduction.png"
echo "Compare α printed above with paper (REPRODUCTION.md Table 1)."
