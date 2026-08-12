#!/bin/bash
#SBATCH --job-name=evaluate_mars_model
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/users/dr00846/Dissertation/ControlNet/logs/evaluate_mars_model_%j.out
#SBATCH --error=/users/dr00846/Dissertation/ControlNet/logs/evaluate_mars_model_%j.err

set -euo pipefail
module load CUDA/12.2.2
cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/ControlNet}/Code"
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/parallel_scratch/dr00846/hf_cache

python3 evaluate_mars_model.py \
  --condition-dir ../Data/processed/paired_controlnet_corpus/test \
  --generated-dir ../Data/eval/generated_controlnet_rgb \
  --output ../Data/eval/mars_model_eval.json \
  --plots-dir ../evaluation_plots \
  --normalize mean_std
