#!/bin/bash
#SBATCH --job-name=controlnet_phase1
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/users/dr00846/Dissertation/ControlNet/logs/controlnet_phase1_%j.out
#SBATCH --error=/users/dr00846/Dissertation/ControlNet/logs/controlnet_phase1_%j.err

set -euo pipefail

module load CUDA/12.2.2

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/ControlNet}/Code"
# The Python venv is shared with CycleGAN (avoids duplicating a multi-GB
# torch/diffusers install) -- lives in CycleGAN/, a sibling of ControlNet/.
source ../../CycleGAN/.venv/bin/activate

# Home dir is slow NFS (see project_dissertation_eureka2_infra memory) --
# keep the HF model cache on parallel_scratch like everything else.
export HF_HOME=/parallel_scratch/dr00846/hf_cache

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --controlnet_model_name_or_path="lllyasviel/sd-controlnet-depth" \
  --output_dir="../checkpoints/controlnet_phase1" \
  --train_data_dir="../Data/processed/paired_controlnet_corpus" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=50 \
  --checkpointing_steps=200 \
  --mixed_precision=fp16 \
  --seed=42
