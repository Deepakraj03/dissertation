#!/bin/bash
#SBATCH --job-name=e2e_pipeline
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/users/dr00846/Dissertation/CycleGAN/logs/e2e_pipeline_%j.out
#SBATCH --error=/users/dr00846/Dissertation/CycleGAN/logs/e2e_pipeline_%j.err

set -euo pipefail

mkdir -p /users/dr00846/Dissertation/CycleGAN/logs

module load CUDA/12.2.2

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/CycleGAN}/Code"
source ../.venv/bin/activate

export HF_HOME=/parallel_scratch/dr00846/hf_cache

python3 -u run_end_to_end_pipeline.py \
  --dtm-controlnet-checkpoint="../checkpoints/dtm_estimator_converted/controlnet" \
  --dtm-unet-lora-checkpoint="../checkpoints/dtm_estimator_converted/unet_lora" \
  --track3-controlnet-checkpoint="../checkpoints/controlnet_phase1" \
  --scratch-dir="../Data/_e2e_pipeline_scratch" \
  --reference-photo="../Data/reference_photos/NLB_552937981EDR_F0642154NCAM00256M1_target.jpg" \
  --out-dir="../Data/e2e_pipeline_output"
