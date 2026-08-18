#!/bin/bash
#SBATCH --job-name=dtm_estimator_eval
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimator_eval_%j.out
#SBATCH --error=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimator_eval_%j.err

set -euo pipefail

mkdir -p /users/dr00846/Dissertation/CycleGAN/logs

module load CUDA/12.2.2

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/CycleGAN}/Code"
source ../.venv/bin/activate

export HF_HOME=/parallel_scratch/dr00846/hf_cache

CKPT_ROOT=../checkpoints/dtm_estimator
CONVERTED_ROOT=../checkpoints/dtm_estimator_converted

python3 -u convert_dtm_estimator_checkpoint.py \
  --checkpoint-dir="$CKPT_ROOT/final" \
  --out-controlnet-dir="$CONVERTED_ROOT/controlnet" \
  --out-unet-lora-dir="$CONVERTED_ROOT/unet_lora"

python3 -u evaluate_dtm_estimator.py \
  --controlnet-checkpoint="$CONVERTED_ROOT/controlnet" \
  --unet-lora-checkpoint="$CONVERTED_ROOT/unet_lora" \
  --test-dir="../Data/processed/dtm_estimation_corpus/test"
