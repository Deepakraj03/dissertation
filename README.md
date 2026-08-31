# Generative Modelling for Realistic Mars Landing Site Visualisation

A project addressing the nadir-to-perspective domain gap in Mars orbital
imagery: translating HiRISE nadir orthoimages into physically consistent,
ground-level rover-perspective views, motivated by immersive VR
reconstruction of candidate landing sites (ESA's ExoMars Rosalind Franklin
mission).

The core architecture separates viewpoint from appearance explicitly, using a
single-image digital terrain model (DTM) estimator, a deterministic
ray-marching renderer, and a paired conditional-diffusion (ControlNet)
texture-synthesis stage, rather than asking an end-to-end network to infer
both from a raw nadir crop. A controlled CycleGAN ablation (Tracks 1 and 2)
validates that this geometric intermediate is what makes viewpoint
translation possible in the first place.

Every reported result is backed by a committed training log, checkpoint, or
evaluation file in this repository.

## Repository layout

| Directory | Contents |
|---|---|
| `CycleGAN/` | Tracks 1 and 2: the unpaired restyling ablation (raw nadir crop vs. geometry-rendered input), the geometry-mediated corpus assembler, and the real-DTM ground-view renderer. |
| `ControlNet/` | The paired texture-synthesis pipeline: real pose-matched corpus assembly (Gale Crater/PDS3 and Jezero Crater/PDS4), the two-stage pseudo-pairing + checkpoint-sweep training procedure, the single-image DTM estimator, and end-to-end pipeline orchestration. |
| `Review papers/` | Source PDFs for the literature review's reviewed publications. |

Bulk data (raw downloads, processed image patches, model checkpoints) is
deliberately excluded from version control — see `.gitignore` for the exact
policy — and is regenerated locally via each subproject's `Code/` scripts.
Small provenance records (corpus manifests, evaluation result JSON files,
training logs) are the tracked exception, so every reported result traces
back to a committed artefact.

## Reproducing a result

Each pipeline stage is a standalone script under the relevant `Code/`
directory. Broadly:

1. **Data acquisition** — `download_hirise.py`, `download_rover.py`,
   `hirise_fullres.py` fetch and preprocess HiRISE/rover imagery from the
   NASA Planetary Data System.
2. **Geometry-mediated corpus** — `assemble_geometry_corpus.py` (CycleGAN
   Track 2) and `assemble_paired_corpus.py` / `assemble_paired_corpus_m2020.py`
   (ControlNet) build the rendered/paired training corpora from real stereo
   DTM coverage and real rover pose metadata.
3. **Training** — `train_cyclegan.py` (Tracks 1/2), `train_dtm_estimator.py`,
   and `train_controlnet.py` (driven by the `launch_pseudo_stage*.sh`
   scripts for the two-stage pseudo-pairing procedure).
4. **Evaluation** — `compute_fid.py`, `evaluate_mars_model.py`, and
   `evaluate_dtm_estimator.py` compute the FID/KID/SSIM/PSNR/RMSE figures
   for each pipeline stage.
