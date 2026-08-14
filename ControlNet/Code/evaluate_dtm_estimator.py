#!/usr/bin/env python
"""Evaluate a trained single-image DTM estimator against real Gale Crater
ground truth: our one held-out real stereo DTM
(DTEEC_040770_1755_039280_1755_A01, Task 8's sole productive DTM for the
paired ControlNet corpus). Computes RMSE between the estimated height map
and the real stereo DTM, in real meters -- matches the paper's own RMSE
reporting convention (rs13214220: RMSE ~1.05m on its held-out global test
set) so this result is directly comparable in units.

The established visual/numeric shelf-artifact spot-check from Task 4/8
(render a ground-level view from the estimated DTM at a real rover pose,
compare against the real target photo that originally exposed the shelf
artifact) is real follow-up work once a trained checkpoint and a matched
real pose both exist -- not built here, since it needs real integration
with render_ground_view.py and a real RoverPose beyond what this
evaluation script's own inputs provide. Deliberately not stubbed out.

NOT YET RUN: requires a real trained checkpoint from train_dtm_estimator.py
(GPU infra, not yet executed -- see that module's docstring) as input.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from peft import PeftModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))
from height_encoding import decode_height_from_unit_range

from train_dtm_estimator import DTM_ESTIMATOR_CAPTION

DEFAULT_GALE_DTM_PRODUCT_ID = "DTEEC_040770_1755_039280_1755_A01"


def compute_rmse(estimated: np.ndarray, real: np.ndarray) -> float:
    """RMSE between estimated and real height arrays (same shape), ignoring
    any NaN (nodata) cells in the real ground truth."""
    valid = ~np.isnan(real)
    if not valid.any():
        return float("nan")
    diff = estimated[valid] - real[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


def load_trained_pipeline(base_model: str, controlnet_checkpoint_dir: str,
                          unet_lora_checkpoint_dir: str, device: str = "cuda",
                          ) -> StableDiffusionControlNetPipeline:
    """Load a trained ControlNet + LoRA-wrapped UNet checkpoint from
    train_dtm_estimator.py into a standard inference pipeline."""
    controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint_dir)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, safety_checker=None,
    )
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_lora_checkpoint_dir)
    return pipeline.to(device)


def run_inference(pipeline: StableDiffusionControlNetPipeline,
                  ortho_condition_image: Image.Image,
                  num_inference_steps: int = 50, seed: int = 0) -> np.ndarray:
    """Run the trained pipeline on a real HiRISE orthoimage, returning the
    predicted height map still in [-1, 1] unit range -- caller decodes to
    real meters via height_encoding.decode_height_from_unit_range, using
    the real ground truth's own min/max as the evaluation-time reference
    range (a convenience for comparing against known ground truth here,
    not something available at real deployment time on an uncovered site)."""
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    result = pipeline(
        prompt=DTM_ESTIMATOR_CAPTION, image=ortho_condition_image,
        num_inference_steps=num_inference_steps, generator=generator,
    ).images[0]
    return np.array(result.convert("L"), dtype=np.float32) / 127.5 - 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-checkpoint", type=str, required=True)
    parser.add_argument("--unet-lora-checkpoint", type=str, required=True)
    parser.add_argument("--gale-ortho", type=str, required=True,
                        help="Real Gale HiRISE orthoimage patch to run inference on")
    parser.add_argument("--gale-real-dtm", type=str, required=True,
                        help="Real stereo DTM .npy patch (same footprint as --gale-ortho)"
                             " to compare against")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pipeline = load_trained_pipeline(
        args.base_model, args.controlnet_checkpoint, args.unet_lora_checkpoint,
        device=args.device,
    )

    ortho_image = Image.open(args.gale_ortho).convert("L")
    real_height = np.load(args.gale_real_dtm)

    estimated_unit_range = run_inference(pipeline, ortho_image)
    valid_real = real_height[~np.isnan(real_height)]
    reference_meta = {"min": float(valid_real.min()), "max": float(valid_real.max())}
    estimated_height = decode_height_from_unit_range(estimated_unit_range, reference_meta)

    rmse = compute_rmse(estimated_height, real_height)
    print(f"RMSE vs. real Gale stereo DTM: {rmse:.4f} m "
         f"(paper's own reference: ~1.0545 m on its held-out global test set)")


if __name__ == "__main__":
    main()
