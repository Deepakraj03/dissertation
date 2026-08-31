#!/bin/bash
# Checkpoint sweep for Stage 4 (dense-Gale-corpus retrain, 1332 pairs).
# Generates and evaluates every saved checkpoint (100-1000) plus the final
# weights against the corpus's own new 134-image test split, same pattern
# as checkpoint_sweep_stage3.sh.
set -uo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

OUTDIR=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage4_dense_gale
TESTDIR=../Data/processed/paired_controlnet_corpus/test
EVAL_ROOT=/scratch/dr00846/Dissertation/ControlNet/Data/eval/stage4_dense_gale_sweep
mkdir -p "$EVAL_ROOT"

echo "checkpoint,step,n_samples,mean_ssim,mean_psnr,kid" > "$EVAL_ROOT/sweep_summary.csv"

for ckpt_dir in "$OUTDIR"/checkpoint-*; do
  [ -d "$ckpt_dir" ] || continue
  name=$(basename "$ckpt_dir")
  step="${name#checkpoint-}"
  echo ""
  echo "=== $name ==="
  gen_dir="$EVAL_ROOT/${name}_generated"
  eval_json="$EVAL_ROOT/${name}_eval.json"
  plots_dir="$EVAL_ROOT/${name}_plots"

  python3 generate_controlnet_translations.py \
    --controlnet-checkpoint "$ckpt_dir/controlnet" \
    --input-dir "$TESTDIR" \
    --output-dir "$gen_dir" \
    --seed 42 || { echo "generation failed for $name, skipping"; continue; }

  python3 evaluate_mars_model.py \
    --condition-dir "$TESTDIR" \
    --generated-dir "$gen_dir" \
    --output "$eval_json" \
    --plots-dir "$plots_dir" || { echo "eval failed for $name, skipping"; continue; }

  python3 -c "
import json
d = json.load(open('$eval_json'))
print(f\"$name,$step,{d['n_samples']},{d['mean_ssim']:.4f},{d['mean_psnr']:.2f},{d['kid']:.5f}\")
" >> "$EVAL_ROOT/sweep_summary.csv"
done

echo ""
echo "=== final ==="
gen_dir="$EVAL_ROOT/final_generated"
eval_json="$EVAL_ROOT/final_eval.json"
plots_dir="$EVAL_ROOT/final_plots"
python3 generate_controlnet_translations.py \
  --controlnet-checkpoint "$OUTDIR" \
  --input-dir "$TESTDIR" \
  --output-dir "$gen_dir" \
  --seed 42 && \
python3 evaluate_mars_model.py \
  --condition-dir "$TESTDIR" \
  --generated-dir "$gen_dir" \
  --output "$eval_json" \
  --plots-dir "$plots_dir" && \
python3 -c "
import json
d = json.load(open('$eval_json'))
print(f\"final,final,{d['n_samples']},{d['mean_ssim']:.4f},{d['mean_psnr']:.2f},{d['kid']:.5f}\")
" >> "$EVAL_ROOT/sweep_summary.csv"

echo ""
echo "=== SWEEP SUMMARY ==="
cat "$EVAL_ROOT/sweep_summary.csv"
echo ""
echo "=== SWEEP DONE ==="
