#!/bin/bash
# Fast, network-cheap replay of the dense Gale sweep from cached candidate
# poses (no per-sol/per-candidate label re-fetch needed), testing whether
# loosening ENTROPY_TH (default 4.5 bits) recovers real, usable pairs from
# the underperforming DTM groups (Section on corpus diversity diagnosis:
# entropy, not pitch, is what differentiates high- and low-yield groups).
# Writes to a separate output dir so the live 1332-pair corpus is untouched.
set -euo pipefail
cd ~/Dissertation/ControlNet/Code
source ../../CycleGAN/.venv/bin/activate

python3 assemble_paired_corpus.py \
  --load-poses-from /scratch/dr00846/gale_candidate_poses_cache.json \
  --region gale_crater \
  --entropy-th 3.5 \
  --output-dir ../Data/processed/paired_controlnet_corpus_entropy35_test
