#!/bin/bash
#SBATCH --job-name=dtm_estimator_visualize
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimator_visualize_%j.out
#SBATCH --error=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimator_visualize_%j.err

set -euo pipefail

mkdir -p /users/dr00846/Dissertation/CycleGAN/logs

module load CUDA/12.2.2

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/CycleGAN}/Code"
source ../.venv/bin/activate

export HF_HOME=/parallel_scratch/dr00846/hf_cache

# Reuses the checkpoint conversion already done by
# submit_evaluate_dtm_estimator.sbatch -- this job is submitted with
# --dependency=afterok on that job, so the converted checkpoint is
# guaranteed to already exist here.
python3 -u visualize_dtm_estimates.py \
  --controlnet-checkpoint="../checkpoints/dtm_estimator_converted/controlnet" \
  --unet-lora-checkpoint="../checkpoints/dtm_estimator_converted/unet_lora" \
  --test-dir="../Data/processed/dtm_estimation_corpus/test" \
  --out-dir="../Data/dtm_estimator_visualize_output"
