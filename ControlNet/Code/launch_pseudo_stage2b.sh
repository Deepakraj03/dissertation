#!/bin/bash
# Stage 2b: extended correction fine-tune, testing the hypothesis that the
# original Stage 2 (50 epochs / 150 steps, lr=2e-6) was too brief/gentle a
# correction pass to stabilize the Stage-1 pseudo-pretrained model against
# real renderer-style condition maps -- diagnosed from real, reproducible
# generation failures (NLB_552938012 broke identically at 3 different
# seeds; NLB_566255628 produced a full wrong-color-regime output) that
# survived the original Stage 2 pass.
#
# Two changes from the original stage 2, both moderate, not extreme, so
# this remains a controlled test of one hypothesis rather than a
# from-scratch new experiment:
#   - 4x more steps: 200 epochs (~600 steps) instead of 50 (~150 steps).
#   - LR raised to 5e-6 (was 2e-6), still well below Stage 1's 1e-5, to
#     give the correction pass more actual pull without risking wholesale
#     overwriting of Stage 1's learned Mars-texture prior.
# Output goes to a NEW directory (controlnet_pseudo_stage2b), not
# overwriting the original controlnet_pseudo_stage2, so both remain
# available to compare.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

STAGE1_CKPT=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage1
CORPUS=../Data/processed/paired_controlnet_corpus
OUTDIR=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage2b
mkdir -p "$OUTDIR"

if [ ! -d "$STAGE1_CKPT" ]; then
  echo "Stage 1 checkpoint not found at $STAGE1_CKPT -- run launch_pseudo_stage1.sh first." >&2
  exit 1
fi

accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --controlnet_model_name_or_path="$STAGE1_CKPT" \
  --output_dir="$OUTDIR" \
  --train_data_dir="$CORPUS" \
  --resolution=512 \
  --learning_rate=5e-6 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=200 \
  --checkpointing_steps=100 \
  --mixed_precision=fp16 \
  --seed=42
