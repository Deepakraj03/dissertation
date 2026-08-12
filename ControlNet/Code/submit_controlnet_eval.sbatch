#!/bin/bash
#SBATCH --job-name=controlnet_eval
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/users/dr00846/Dissertation/ControlNet/logs/controlnet_eval_%j.out
#SBATCH --error=/users/dr00846/Dissertation/ControlNet/logs/controlnet_eval_%j.err

set -euo pipefail
module load CUDA/12.2.2
cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/ControlNet}/Code"
source ../../CycleGAN/.venv/bin/activate
export HF_HOME=/parallel_scratch/dr00846/hf_cache

python3 generate_controlnet_translations.py \
  --controlnet-checkpoint ../checkpoints/controlnet_phase1 \
  --base-model stable-diffusion-v1-5/stable-diffusion-v1-5

python3 prepare_controlnet_real_rgb.py

# compute_fid.py is shared/generic -- stays in CycleGAN/Code.
python3 ../../CycleGAN/Code/compute_fid.py \
  --generated-dir ../Data/eval/generated_controlnet_rgb \
  --real-dir ../Data/eval/real_controlnet_rgb \
  --checkpoint controlnet_phase1 \
  --output ../Data/eval/fid_results_controlnet.json
