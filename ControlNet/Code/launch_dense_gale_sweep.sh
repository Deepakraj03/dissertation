#!/bin/bash
# Dense per-sol Gale Crater corpus sweep: same real assemble_paired_corpus.py
# pipeline (same DTM catalog -- 25 real HiRISE DTM products, cross-verified
# against both PDS ODE and USGS STAC -- same pitch/entropy filters), but
# sampling EVERY sol (1..4699) instead of the step-20/-50 sampling the
# 66-pair corpus used. Rationale: the existing corpus draws 65/66 pairs
# from a single DTM group (DTEEC_040770); the other 24 real, cataloged DTM
# footprints sit at 0-1 pairs each, which may reflect sparse sol sampling
# missing narrow rover-traverse/DTM-footprint overlap windows rather than
# those crossings not existing. This is a full rebuild (assemble_paired_corpus.py
# has no incremental/append mode -- it clears and rewrites its output dir
# each run), which is why the prior 66-pair corpus was tarred as a backup
# first.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate

python3 assemble_paired_corpus.py \
  --sols $(seq 1 1 4699) \
  --per-sol 8 \
  --region gale_crater
