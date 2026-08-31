#!/bin/bash
# Two apples-to-apples cross-checks to isolate whether the combined-corpus
# Stage 3 run's lower KID (0.0603 at checkpoint-200) vs Stage 2c's
# Gale-only run (0.142 at checkpoint-400) reflects a real training
# improvement or just a different, larger, Jezero-heavy test set:
#
#  A) Stage 2c's ckpt-400 (Gale-only trained) evaluated against the
#     COMBINED 27-pair test set -- same eval set as Stage 3's headline
#     number, isolating the training-data effect.
#  B) Stage 3 combined's ckpt-200 (Gale+Jezero trained) evaluated against
#     the GALE-ONLY 8-pair test set -- same eval set as Stage 2c's
#     headline number, isolating the training-data effect from the other
#     direction.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

STAGE2C_400=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage2c/checkpoint-400/controlnet
STAGE3_200=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage3_combined/checkpoint-200/controlnet

COMBINED_TEST=../Data/processed/paired_controlnet_corpus_combined/test
GALE_TEST=../Data/processed/paired_controlnet_corpus/test
JEZERO_TEST=../Data/processed/paired_controlnet_corpus_m2020/test

EVAL_ROOT=/scratch/dr00846/Dissertation/ControlNet/Data/eval/cross_check
mkdir -p "$EVAL_ROOT"

run_check () {
  name="$1"; ckpt="$2"; test_dir="$3"
  echo ""
  echo "=== $name ==="
  gen_dir="$EVAL_ROOT/${name}_generated"
  eval_json="$EVAL_ROOT/${name}_eval.json"
  plots_dir="$EVAL_ROOT/${name}_plots"
  python3 generate_controlnet_translations.py \
    --controlnet-checkpoint "$ckpt" \
    --input-dir "$test_dir" \
    --output-dir "$gen_dir" \
    --seed 42
  python3 evaluate_mars_model.py \
    --condition-dir "$test_dir" \
    --generated-dir "$gen_dir" \
    --output "$eval_json" \
    --plots-dir "$plots_dir"
}

run_check "A_stage2c400_on_combinedtest" "$STAGE2C_400" "$COMBINED_TEST"
run_check "B_stage3_200_on_galetest"     "$STAGE3_200"   "$GALE_TEST"
run_check "C_stage3_200_on_jezerotest"   "$STAGE3_200"   "$JEZERO_TEST"
run_check "D_stage2c400_on_galetest"     "$STAGE2C_400"  "$GALE_TEST"

echo ""
echo "=== CROSS-CHECK SUMMARY ==="
for j in "$EVAL_ROOT"/*_eval.json; do
  python3 -c "
import json
d = json.load(open('$j'))
print(f\"$(basename $j .json.json 2>/dev/null || true)\")
"
done
python3 -c "
import json, glob, os
for j in sorted(glob.glob('$EVAL_ROOT/*_eval.json')):
    d = json.load(open(j))
    name = os.path.basename(j).removesuffix('_eval.json')
    print(f\"{name}: n={d['n_samples']} ssim={d['mean_ssim']:.4f} psnr={d['mean_psnr']:.2f} kid={d['kid']:.5f}\")
"
echo "=== CROSS-CHECK DONE ==="
