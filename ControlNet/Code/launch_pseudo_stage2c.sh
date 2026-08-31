#!/bin/bash
# Stage 2c: correction fine-tune on the GROWN real corpus (66 pairs, 52
# train -- up from the original 27/21, via denser sol sampling in
# assemble_paired_corpus.py). Same lr=5e-6 as Stage 2b (the setting that
# let us find a real sweet spot via checkpoint sweep on the smaller corpus),
# initialized from Stage 1's pseudo-pretrained checkpoint as before.
#
# Epoch count: 52 train images / batch 2 = 26 batches/epoch; with
# gradient_accumulation_steps=4 that's ~6.5 optimizer steps/epoch, so 120
# epochs ~ 780 steps -- covers roughly the same step range Stage 2b's sweep
# did (100-600), giving room to find this larger corpus's own sweet spot
# via the same checkpoint-sweep-then-evaluate approach, rather than
# assuming it lands at the same step count as the 27-pair corpus did.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/scratch/dr00846/hf_cache

STAGE1_CKPT=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage1
CORPUS=../Data/processed/paired_controlnet_corpus
OUTDIR=/scratch/dr00846/Dissertation/CycleGAN/checkpoints/controlnet_pseudo_stage2c
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
  --num_train_epochs=120 \
  --checkpointing_steps=100 \
  --mixed_precision=fp16 \
  --seed=42
