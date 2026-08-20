#!/usr/bin/env python
"""Real end-to-end test of the full geometry-mediated Stage 1+2 chain:

    HiRISE ortho (nadir, 25cm/px)
      -> DTM estimator (train_dtm_estimator.py checkpoint)   [Stage 1]
      -> render_ground_view.py, synthetic camera pose         [geometry, no network]
      -> Track 3 ControlNet (train_controlnet.py checkpoint)  [Stage 2, texture]
      -> photorealistic perspective output

Nobody has run these three pieces back-to-back before -- the DTM estimator
was evaluated against real ground truth on its own (evaluate_dtm_estimator.py,
visualize_dtm_estimates.py), and separately spotcheck_dtm_estimator_render.py
compared crude renders from the real vs. estimated DTM, but neither of those
passes a render through the Track 3 ControlNet's texture synthesis. This
script closes that gap: it re-runs spotcheck_dtm_estimator_render.py's exact
real-vs-estimated-DTM render logic, then feeds BOTH renders through the same
Track 3 ControlNet checkpoint, so the real question -- "does the DTM
estimator's ~2m RMSE error visibly degrade the final photorealistic output
relative to using the real DTM?" -- has a real image answer instead of an
inference from separate pieces.

Uses the same fixed synthetic camera pose as spotcheck_dtm_estimator_render.py
(crop center, heading=0, pitch=-15deg) for the same reason documented there:
recovering one of the 27 real paired-corpus rover poses exactly isn't
feasible from the product IDs alone.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))

from dtm_coverage import query_dtm_coverage, signed_lon_to_0_360
from dtm_arrays import load_dtm_arrays
from fetch_hirise_dtm import fetch_dtm_and_orthos
from render_ground_view import render_ground_view
from height_encoding import decode_height_from_unit_range

from evaluate_dtm_estimator import load_trained_pipeline as load_dtm_estimator_pipeline
from evaluate_dtm_estimator import run_inference as run_dtm_estimator_inference
from spotcheck_dtm_estimator_render import (
    GALE_PRODUCT_ID, GALE_LAT_MIN, GALE_LAT_MAX, GALE_LON_MIN, GALE_LON_MAX,
    SPOTCHECK_PITCH_DEG, crop_window_centered, find_gale_dtm_record,
)

CONTROLNET_CAPTION = "a photorealistic Mars rover Navcam photograph of the martian surface"


def load_controlnet_pipeline(controlnet_checkpoint: Path, base_model: str,
                             device: torch.device) -> StableDiffusionControlNetPipeline:
    """Same loading pattern as generate_controlnet_translations.py's
    load_pipeline -- Track 3's checkpoint is a plain diffusers ControlNetModel
    (no LoRA), unlike the DTM estimator's ControlNet+LoRA combination."""
    controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


def run_controlnet_texture(pipe: StableDiffusionControlNetPipeline, condition_render: np.ndarray,
                           num_inference_steps: int = 30, seed: int = 42) -> Image.Image:
    condition_image = Image.fromarray(condition_render).convert("RGB")
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        CONTROLNET_CAPTION, image=condition_image,
        num_inference_steps=num_inference_steps, generator=generator,
    ).images[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtm-controlnet-checkpoint", type=str, required=True)
    parser.add_argument("--dtm-unet-lora-checkpoint", type=str, required=True)
    parser.add_argument("--track3-controlnet-checkpoint", type=str, required=True)
    parser.add_argument("--base-model", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--precomputed-crop-npz", type=str, default=None,
                        help="Skip the real DTM/ortho fetch (~940MB, both"
                             " source-observation orthos + the DTM) and load"
                             " real_height_crop/albedo_crop/pixel_scale_m"
                             " from this .npz instead (see prepare_e2e_crop.py)"
                             " -- for GPU machines too disk-constrained to"
                             " fetch the raw product themselves.")
    parser.add_argument("--reference-photo", type=str, default=None,
                        help="Optional real target photo (any real pose on"
                             " this product) copied alongside for loose,"
                             " NOT camera-matched visual reference")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--pitch-deg", type=float, default=SPOTCHECK_PITCH_DEG,
                        help="Synthetic camera pitch for both crude renders"
                             " (default matches spotcheck_dtm_estimator_render.py's"
                             " -15deg steep down-look; a shallower value, e.g."
                             " -5, shows less of the known shelf artifact and"
                             " is a fairer test of whether the DTM estimator's"
                             " error visibly propagates to the final image)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtm_path = None
    fetch_result = {}
    try:
        if args.precomputed_crop_npz:
            print(f"Loading precomputed crop from {args.precomputed_crop_npz}...")
            npz = np.load(args.precomputed_crop_npz)
            real_height_crop = npz["real_height_crop"]
            albedo_crop = npz["albedo_crop"]
            pixel_scale_m = float(npz["pixel_scale_m"])
        else:
            print("Fetching real Gale DTM/ortho...")
            record = find_gale_dtm_record()
            fetch_result = fetch_dtm_and_orthos(record, scratch_dir, scratch_dir)
            if fetch_result["status"] != "ok":
                raise RuntimeError(f"fetch_dtm_and_orthos failed: {fetch_result['status']}")
            dtm_path = Path(fetch_result["dtm_path"])
            ortho_path = Path(next(iter(fetch_result["ortho_paths"].values())))

            arrays = load_dtm_arrays(dtm_path, ortho_path)
            rows, cols = arrays.heightmap.shape
            row_start, col_start = crop_window_centered(
                (rows, cols), rows / 2, cols / 2, args.patch_size)
            row_end, col_end = row_start + args.patch_size, col_start + args.patch_size

            real_height_crop = arrays.heightmap[row_start:row_end, col_start:col_end]
            albedo_crop = arrays.albedo[row_start:row_end, col_start:col_end]
            pixel_scale_m = arrays.pixel_scale_m

        camera_row = camera_col = args.patch_size / 2

        print("Rendering crude perspective view from the REAL stereo DTM...")
        real_render = render_ground_view(
            real_height_crop, albedo_crop, pixel_scale_m,
            camera_row=camera_row, camera_col=camera_col, heading_deg=0.0,
            pitch_deg=args.pitch_deg,
        )

        print("Stage 1: running the DTM estimator on the HiRISE ortho patch...")
        dtm_pipeline = load_dtm_estimator_pipeline(
            args.base_model, args.dtm_controlnet_checkpoint, args.dtm_unet_lora_checkpoint,
            device=str(device),
        )
        ortho_image = Image.fromarray(albedo_crop).convert("L")
        estimated_unit_range = run_dtm_estimator_inference(dtm_pipeline, ortho_image)
        valid_real = real_height_crop[~np.isnan(real_height_crop)]
        reference_meta = {"min": float(valid_real.min()), "max": float(valid_real.max())}
        estimated_height_crop = decode_height_from_unit_range(
            estimated_unit_range, reference_meta)
        del dtm_pipeline
        torch.cuda.empty_cache()

        print("Rendering crude perspective view from the ESTIMATED DTM...")
        estimated_render = render_ground_view(
            estimated_height_crop, albedo_crop, pixel_scale_m,
            camera_row=camera_row, camera_col=camera_col, heading_deg=0.0,
            pitch_deg=args.pitch_deg,
        )
    finally:
        if dtm_path is not None:
            dtm_path.unlink(missing_ok=True)
        for p in fetch_result.get("ortho_paths", {}).values():
            Path(p).unlink(missing_ok=True)

    print("Stage 2: running Track 3 ControlNet on both crude renders...")
    controlnet_pipeline = load_controlnet_pipeline(
        args.track3_controlnet_checkpoint, args.base_model, device)
    real_chain_output = run_controlnet_texture(controlnet_pipeline, real_render)
    estimated_chain_output = run_controlnet_texture(controlnet_pipeline, estimated_render)

    Image.fromarray(albedo_crop).save(out_dir / "01_hirise_ortho_input.png")
    Image.fromarray(real_render).save(out_dir / "02a_real_dtm_crude_render.png")
    Image.fromarray(estimated_render).save(out_dir / "02b_estimated_dtm_crude_render.png")
    real_chain_output.save(out_dir / "03a_real_dtm_chain_final_output.png")
    estimated_chain_output.save(out_dir / "03b_estimated_dtm_chain_final_output.png")

    side_by_side = Image.new("RGB", (256 * 4, 256))
    for i, img in enumerate([
        Image.fromarray(albedo_crop).convert("RGB").resize((256, 256)),
        Image.fromarray(real_render).convert("RGB").resize((256, 256)),
        real_chain_output.resize((256, 256)),
        estimated_chain_output.resize((256, 256)),
    ]):
        side_by_side.paste(img, (256 * i, 0))
    side_by_side.save(out_dir / "04_pipeline_hirise_realrender_realchain_estimatedchain.png")

    if args.reference_photo:
        shutil.copy(args.reference_photo,
                    out_dir / "reference_real_photo_NOT_camera_matched.jpg")

    print(f"End-to-end pipeline outputs written to {out_dir}")


if __name__ == "__main__":
    main()
