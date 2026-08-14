#!/bin/bash
#SBATCH --job-name=dtm_estimation_corpus
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimation_corpus_%j.out
#SBATCH --error=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimation_corpus_%j.err

# No --gres=gpu: this job is pure network fetch + numpy/PIL patch tiling
# (assemble_dtm_estimation_corpus.py imports no torch/diffusers/peft), so
# it doesn't need one. Still on the gpu partition/cvssp account because
# that's the only combination confirmed working on this account so far
# (see submit_controlnet_phase1.sbatch) -- if a real CPU-only partition
# exists on Eureka2, switch to it so this doesn't compete with GPU jobs
# for scheduling.

set -euo pipefail

mkdir -p /users/dr00846/Dissertation/CycleGAN/logs

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/CycleGAN}/Code"
source ../.venv/bin/activate

# Home dir is slow NFS (see project_dissertation_eureka2_infra memory) --
# stage the corpus on parallel_scratch, not the default Data/ under home.
export DTM_CORPUS_ROOT=/parallel_scratch/dr00846/dtm_estimation_corpus
mkdir -p "$DTM_CORPUS_ROOT"

# RECOMMENDATION, not enforced: this pipeline (global catalog query,
# fetch/tile/screen/augment) has never been run for real against the full
# ~1,319-product global catalog -- only unit-tested with mocked I/O.
# Consider a smoke test first (`--limit 20`) to confirm real throughput
# and disk usage before committing a 24h job to the full run. Full run
# below as requested.

python3 assemble_dtm_estimation_corpus.py \
  --patch-size 256 \
  --stride 256 \
  --alignment-th 0.4 \
  --out-root "$DTM_CORPUS_ROOT"

echo "Corpus assembly finished. Output under $DTM_CORPUS_ROOT/Data/processed/dtm_estimation_corpus/"
