#!/bin/bash
#SBATCH --job-name=dtm_estimation_corpus
#SBATCH --partition=gpu
#SBATCH --account=cvssp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimation_corpus_%A_%a.out
#SBATCH --error=/users/dr00846/Dissertation/CycleGAN/logs/dtm_estimation_corpus_%A_%a.err

# No --gres=gpu: this job is pure network fetch + numpy/PIL patch tiling
# (assemble_dtm_estimation_corpus.py imports no torch/diffusers/peft), so
# it doesn't need one. Still on the gpu partition/cvssp account because
# that's the only combination confirmed working on this account so far
# (see submit_controlnet_phase1.sbatch) -- if a real CPU-only partition
# exists on Eureka2, switch to it so this doesn't compete with GPU jobs
# for scheduling.
#
# --array=0-19 (20 shards): the real 2026-08-14 smoke test measured
# ~8 min/product on 20 products; the full catalog (~1,274 products as of
# that run, grows over time) at that rate is ~7 days single-threaded --
# past any realistic job time limit. 20 parallel shards brings that down
# to roughly 8-9 hours each. Check this cluster's real per-user/per-account
# array-size and concurrent-job limits before submitting (not confirmed
# here) and adjust --array/--time accordingly.
#
# Each shard writes its own manifest incrementally and resumes from it if
# interrupted (see assemble_dtm_estimation_corpus.py) -- a shard that hits
# the time limit can be resubmitted with the same --array index and will
# skip products it already finished, not restart from scratch.
#
# After ALL array tasks finish, run finalize_dtm_estimation_corpus.py once
# (separately, not part of this array) to merge the per-shard manifests
# and build the real train/val/test split -- see
# submit_finalize_dtm_estimation_corpus.sbatch.

set -euo pipefail

mkdir -p /users/dr00846/Dissertation/CycleGAN/logs

cd "${SLURM_SUBMIT_DIR:-$HOME/Dissertation/CycleGAN}/Code"
source ../.venv/bin/activate

# Home dir is slow NFS (see project_dissertation_eureka2_infra memory) --
# stage the corpus on parallel_scratch, not the default Data/ under home.
export DTM_CORPUS_ROOT=/parallel_scratch/dr00846/dtm_estimation_corpus
mkdir -p "$DTM_CORPUS_ROOT"

# -u: unbuffered stdout -- the 2026-08-14 smoke test's log stayed empty
# for the entire ~2h41m run because Python block-buffers stdout when it's
# not a TTY, only flushing at exit. Without -u there is no way to tell a
# slow-but-healthy shard apart from a stuck one from the log alone.
python3 -u assemble_dtm_estimation_corpus.py \
  --patch-size 256 \
  --stride 256 \
  --alignment-th 0.4 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards 20 \
  --out-root "$DTM_CORPUS_ROOT"

echo "Shard $SLURM_ARRAY_TASK_ID finished."
