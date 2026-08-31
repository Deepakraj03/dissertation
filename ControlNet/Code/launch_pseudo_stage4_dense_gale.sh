#!/bin/bash
# Stage 4: correction fine-tune on the DENSE-sampled Gale corpus (1,332
# pairs, 1065 train -- up from Stage 2c's 66-pair corpus, via every-sol
# rather than step-20 sampling against the same 25 real HiRISE DTM
# footprints). Same lr=5e-6, batch=2, grad_accum=4, and stage1
# pseudo-pretrain initialization as Stage 2c, so this run isolates the
# effect of the 20x larger real corpus, not a change in recipe.
#
# Epoch count: 1065 train images / batch 2 = 533 batches/epoch; with
# gradient_accumulation_steps=4 that's ceil(533/4)=134 optimizer
# steps/epoch, so 8 epochs ~ 1072 steps -- checkpointed every 100 steps
# gives a comparable-resolution sweep (~10 checkpoints) to Stage 2c's.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

STAGE1_CKPT=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage1
CORPUS=../Data/processed/paired_controlnet_corpus
OUTDIR=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage4_dense_gale
mkdir -p "$OUTDIR"

if [ ! -d "$STAGE1_CKPT" ]; then
  echo "Stage 1 checkpoint not found at $STAGE1_CKPT -- run launch_pseudo_stage1.sh first." >&2
  exit 1
fi

python3 prepare_controlnet_hf_dataset.py "$CORPUS"

accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --controlnet_model_name_or_path="$STAGE1_CKPT" \
  --output_dir="$OUTDIR" \
  --train_data_dir="$CORPUS" \
  --resolution=512 \
  --learning_rate=5e-6 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=8 \
  --checkpointing_steps=100 \
  --mixed_precision=fp16 \
  --seed=42
