#!/bin/bash
# Stage 2: short correction fine-tune on the 27 REAL pose-matched pairs,
# initialized from Stage 1's final checkpoint (not from
# lllyasviel/sd-controlnet-depth), so the model that has already learned
# broad Mars rover texture from 20,417 pseudo-pairs gets a final calibration
# pass against genuine geometry-render/real-photo correspondence.
#
# Learning rate lowered to 2e-6 (from 1e-5) for this stage deliberately:
# 27 images is small enough that the original LR risks overwriting Stage
# 1's learned prior rather than gently correcting it. Epoch count matches
# the original Track 3 run (50) since the corpus size is the same.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

STAGE1_CKPT=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage1
CORPUS=../Data/processed/paired_controlnet_corpus
OUTDIR=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage2
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
  --learning_rate=2e-6 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=50 \
  --checkpointing_steps=200 \
  --mixed_precision=fp16 \
  --seed=42
