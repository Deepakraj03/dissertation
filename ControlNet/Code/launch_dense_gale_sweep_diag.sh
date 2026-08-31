#!/bin/bash
# Re-run of the dense per-sol Gale sweep, now instrumented: caches every
# gathered candidate pose to disk (--cache-poses-to) so any future
# filter-tuning experiment can replay against them with --load-poses-from
# and no network cost, and records a per-DTM-group rejection-reason
# breakdown (pitch / render_failed / low_entropy / fetch_failed) to
# diagnose why 4 of 5 covered DTM groups yield far fewer pairs per
# candidate than DTEEC_040770 does.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate

python3 assemble_paired_corpus.py \
  --sols $(seq 1 1 4699) \
  --per-sol 8 \
  --region gale_crater \
  --cache-poses-to /scratch/dr00846/gale_candidate_poses_cache.json
