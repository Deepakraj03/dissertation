#!/bin/bash
#SBATCH --job-name=dtm_estimation_finalize
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimation_finalize_%j.out
#SBATCH --error=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimation_finalize_%j.err

# Run once, after every submit_dtm_estimation_corpus.sbatch array task has
# reached a real terminal state (check with `sacct -j <array_job_id>`) --
# merges the per-shard manifests and builds the real train/val/test split.
# Running this before all shards finish would build a split missing
# whatever shards hadn't completed yet.
#
# Chain automatically once the array job ID is known:
#   sbatch --dependency=afterany:<array_job_id> submit_finalize_dtm_estimation_corpus.sbatch

set -euo pipefail

mkdir -p /users/dr00846/Dissertation/CycleGAN/logs

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/CycleGAN}/Code"
source ../.venv/bin/activate

export DTM_CORPUS_ROOT=/parallel_scratch/dr00846/dtm_estimation_corpus

python3 -u finalize_dtm_estimation_corpus.py --out-root "$DTM_CORPUS_ROOT"
