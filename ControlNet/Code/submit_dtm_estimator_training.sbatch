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

# Run submit_dtm_estimation_corpus.sbatch first and confirm it produced
# real train/ data under $DTM_CORPUS_ROOT before submitting this job --
# there is no automatic --dependency wiring here since the corpus job's
# real duration/product yield isn't known yet. To chain them once both
# are validated: sbatch --dependency=afterok:<corpus_job_id> submit_dtm_estimator_training.sbatch

set -euo pipefail

mkdir -p /users/dr00846/Dissertation/ControlNet/logs

module load CUDA/12.2.2

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/ControlNet}/Code"
# Shared venv with CycleGAN (avoids duplicating a multi-GB torch/diffusers
# install) -- same convention as submit_controlnet_phase1.sbatch. peft was
# added to ControlNet/requirements.txt for this run -- confirm it's
# installed in this venv before submitting (pip install -r
# ControlNet/requirements.txt from the venv if not).
source ../../CycleGAN/.venv/bin/activate

# Home dir is slow NFS (see project_dissertation_eureka2_infra memory) --
# same scratch location the corpus job wrote to.
export DTM_CORPUS_ROOT=/parallel_scratch/dr00846/dtm_estimation_corpus
export HF_HOME=/parallel_scratch/dr00846/hf_cache

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# NOT YET RUN END-TO-END: train_dtm_estimator.py's core math (x0-estimate,
# height-regression loss, dataset loading, LoRA config) is unit-tested on
# CPU against real torch/diffusers/peft, but the full accelerate training
# loop has never executed against real GPU hardware or the real
# pretrained SD1.5 weights. Recommend a short first run here
# (--max_train_steps=50, drop --checkpointing_steps to e.g. 25) to confirm
# it actually runs end-to-end before committing a 48h job to the values
# below. Resolution isn't a training-script flag -- it's whatever
# patch-size the corpus job was built with (256 by default, see
# submit_dtm_estimation_corpus.sbatch).
accelerate launch train_dtm_estimator.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --controlnet_model_name_or_path="lllyasviel/sd-controlnet-depth" \
  --train_data_dir="$DTM_CORPUS_ROOT/Data/processed/dtm_estimation_corpus" \
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
  --seed=42
