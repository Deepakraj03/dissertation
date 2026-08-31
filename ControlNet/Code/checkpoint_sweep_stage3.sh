#!/bin/bash
# Waits for launch_pseudo_stage3_combined.sh's training process to exit,
# then sweeps every saved checkpoint-N (generate translations over the
# combined corpus's real 27-pair test split, then evaluate_mars_model.py
# for KID/SSIM/PSNR against those same real targets), writing one row per
# checkpoint to sweep_summary.csv. Mirrors the manual checkpoint-sweep
# process already used to pick Stage 2b/2c's best checkpoint, just
# scripted so it can run unattended for however long training takes.
set -uo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

OUTDIR=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage3_combined
TESTDIR=../Data/processed/paired_controlnet_corpus_combined/test
EVAL_ROOT=/scratch/dr00846/Dissertation/ControlNet/Data/eval/stage3_combined_sweep
SUMMARY="$EVAL_ROOT/sweep_summary.csv"
mkdir -p "$EVAL_ROOT"

echo "Waiting for training process (train_controlnet.py) to finish..."
while pgrep -f "train_controlnet.py.*stage3_combined" > /dev/null; do
  sleep 30
done
echo "Training finished. Sweeping checkpoints in $OUTDIR"

echo "checkpoint,step,n_samples,mean_ssim,mean_psnr,kid" > "$SUMMARY"

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
" >> "$SUMMARY"
done

# Also sweep the final (non-checkpoint-N) output_dir weights, saved after
# the last epoch -- same convention Stage 2c's manual sweep included.
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
" >> "$SUMMARY"

echo ""
echo "=== SWEEP SUMMARY ==="
cat "$SUMMARY"
echo ""
echo "=== SWEEP DONE ==="
