#!/bin/bash
#SBATCH --job-name=dtm_estimator_train
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/users/dr00846/Dissertation/ControlNet/logs/dtm_estimator_train_%j.out
#SBATCH --error=/users/dr00846/Dissertation/ControlNet/logs/dtm_estimator_train_%j.err

# Run after submit_dtm_estimation_corpus.sbatch + finalize_dtm_estimation_corpus.py
# have produced a real manifest.csv and train/val/test split. Real
# 2026-08-16 run: 1,274 products processed, 815 productive, split
# train=354,616 / val=34,164 / test=25,048 patches (post-augmentation).
#
# No --out-root override needed on this host: CycleGAN/Data is already a
# symlink to /parallel_scratch/dr00846/Dissertation/CycleGAN/Data, so the
# corpus landed there directly -- same convention as the corpus job.

set -euo pipefail

mkdir -p /users/dr00846/Dissertation/ControlNet/logs

module load CUDA/12.2.2

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/CycleGAN}/Code"
# Shared venv with the corpus-assembly job -- same convention as
# submit_controlnet_phase1.sbatch. peft was added to
# ControlNet/requirements.txt for this run -- confirm it's installed in
# this venv before submitting (pip install peft>=0.20 from the venv if not).
source ../.venv/bin/activate

export HF_HOME=/parallel_scratch/dr00846/hf_cache

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Real run on Eureka2 (job 748677) reached checkpoint-2000 before dying
# when the home-dir NFS quota filled mid-write -- checkpoints now live on
# /parallel_scratch (symlinked in), and --resume_from_checkpoint=latest
# picks up wherever the last complete checkpoint left off.
accelerate launch train_dtm_estimator.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --controlnet_model_name_or_path="lllyasviel/sd-controlnet-depth" \
  --train_data_dir="../Data/processed/dtm_estimation_corpus" \
  --output_dir="../checkpoints/dtm_estimator" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --max_train_steps=20000 \
  --checkpointing_steps=1000 \
  --lambda_height=0.5 \
  --lora_rank=16 \
  --lora_alpha=16 \
  --mixed_precision=bf16 \
  --seed=42 \
  --resume_from_checkpoint=latest
