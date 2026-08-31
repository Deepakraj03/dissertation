#!/bin/bash
# Stage 1 of the two-stage pseudo-pairing plan: fine-tune ControlNet on the
# large style-matched pseudo-paired corpus (20,417 train pairs, from
# generate_pseudo_conditions.py --style-match). Starts from the same
# lllyasviel/sd-controlnet-depth initialization Track 3 used, NOT from the
# Track 3 checkpoint, since this is an independent experiment being compared
# against it, not a continuation of it.
#
# Epoch count: 20,417 images / batch 2 = ~10,208 batches/epoch; with
# gradient_accumulation_steps=4 that's ~2,552 optimizer steps/epoch, so 3
# epochs ~ 7,656 steps -- roughly 60x the ~125 total steps the original
# 27-pair run got. Conservative first attempt, not tuned; extend if the
# loss curve still looks like it's improving at the end.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

CORPUS=/scratch/dr00846/Dissertation/ControlNet/Data/processed/pseudo_paired_controlnet_corpus
OUTDIR=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage1
mkdir -p "$OUTDIR"

python3 prepare_controlnet_hf_dataset.py "$CORPUS"

accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --controlnet_model_name_or_path="lllyasviel/sd-controlnet-depth" \
  --output_dir="$OUTDIR" \
  --train_data_dir="$CORPUS" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=3 \
  --checkpointing_steps=1000 \
  --mixed_precision=fp16 \
  --seed=42
