#!/bin/bash
# Stage 3: correction fine-tune on the COMBINED real corpus (244 pairs, 194
# train -- Gale's grown 66-pair corpus + the newly built 178-pair Jezero
# corpus). Same lr=5e-6 and stage1-pseudo-pretrain initialization as Stage
# 2c (the setting a checkpoint sweep validated on the smaller, Gale-only
# corpus), so this run isolates the effect of adding a second landing
# site's real pairs rather than also changing the optimization recipe.
#
# Epoch count: 194 train images / batch 2 = 97 batches/epoch; with
# gradient_accumulation_steps=4 that's ceil(97/4)=25 optimizer steps/epoch,
# so 36 epochs ~ 900 steps -- covers the same step range (100-800/900) the
# prior sweeps searched, so the sweep below can find this corpus's own
# sweet spot rather than assuming it lands at the same step count as the
# smaller corpus did.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

STAGE1_CKPT=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage1
CORPUS=../Data/processed/paired_controlnet_corpus_combined
OUTDIR=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage3_combined
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
  --num_train_epochs=36 \
  --checkpointing_steps=100 \
  --mixed_precision=fp16 \
  --seed=42
