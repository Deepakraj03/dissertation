"""Build a pseudo-paired ControlNet corpus from the large unpaired rover
corpus, using a monocular depth estimator as a self-supervised stand-in for
the real ray-march renderer's condition map.

Rationale: the real pose-matched corpus (assemble_paired_corpus.py) has only
27 genuinely paired (condition, target) examples, because it requires a real
Navcam pose to fall inside real HiRISE stereo DTM coverage. This script
sidesteps that constraint entirely by generating a condition map directly
FROM each real rover photo via a pretrained depth model, the same paradigm
the project's own sd-controlnet-depth initialization checkpoint was itself
trained with (MiDaS-style depth maps paired with the photo they came from).
This turns the ~20,417-image unpaired rover corpus into an equally large
pseudo-paired one.

Known, documented risk (not resolved by this script): the depth estimator's
output style (continuous relative depth, no explicit sky/nodata region) does
not exactly match the real renderer's style (ray-marched height field,
albedo-draped, explicit blank sky at sky_value=0). Whether this domain gap
actually hurts generalisation from pseudo-pairs to real renderer output at
inference time is an open empirical question this script does not answer;
see the smoke-test comparison this is paired with before committing to the
full run.

Output layout matches paired_controlnet_corpus exactly (<id>_condition.png +
<id>_target.jpg per split), so prepare_controlnet_hf_dataset.py and
train_controlnet.py work against it completely unmodified.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

CAPTION = "a photorealistic Mars rover Navcam photograph of the martian surface"
DEPTH_MODEL = "LiheYoung/depth-anything-small-hf"


def style_match_to_renderer(depth_map, n_bands: int, sky_top_frac: float, sky_abs_threshold: float):
    """Reshape a raw monocular-depth output to resemble the real ray-march
    renderer's condition-map style, so a ControlNet trained on pseudo-pairs
    sees a training-time input distribution closer to what it is actually
    fed at inference (the real renderer's output).

    Two operations, both driven by tunable parameters so they can be
    calibrated against a real example before committing to a full run:

    1. Quantise into n_bands discrete grey levels, approximating the
       renderer's terraced, low-poly step-shading (the real renderer's
       facets come from a discretised height field lit by one directional
       source; a smooth continuous depth map has no equivalent).
    2. Hard sky cutoff: pixels in the top sky_top_frac of the image AND
       below an ABSOLUTE brightness threshold (sky_abs_threshold, on the
       model's own 0-255 per-image normalisation) are forced to 0,
       approximating the renderer's explicit sky_value=0 for rays that
       never hit terrain. Deliberately absolute, not a percentile within
       the top region: a percentile-of-whatever-is-there always finds
       "the bottom N%" even when the crop has no real sky at all (a
       close-range ground patch with a shadow band, say), which was
       confirmed to wrongly zero real terrain during calibration. An
       absolute threshold only fires when something is genuinely near the
       model's own far-depth floor, so a crop with no real sky triggers
       nothing.
    """
    import numpy as np

    arr = np.array(depth_map, dtype=np.float32)
    h, w = arr.shape

    # Quantise to n_bands discrete levels.
    band_edges = np.linspace(0, 255, n_bands + 1)
    band_centres = (band_edges[:-1] + band_edges[1:]) / 2
    band_idx = np.clip(np.digitize(arr, band_edges[1:-1]), 0, n_bands - 1)
    quantised = band_centres[band_idx]

    # Hard sky cutoff: absolute threshold, restricted to the top portion of the frame.
    top_h = int(h * sky_top_frac)
    if top_h > 0:
        sky_mask = np.zeros_like(arr, dtype=bool)
        sky_mask[:top_h, :] = arr[:top_h, :] <= sky_abs_threshold
        quantised[sky_mask] = 0

    from PIL import Image

    return Image.fromarray(quantised.astype(np.uint8), mode="L")


def build_split(
    pipe,
    src_dir: Path,
    dst_dir: Path,
    limit: int | None = None,
    style_match: bool = False,
    n_bands: int = 10,
    sky_top_frac: float = 0.4,
    sky_abs_threshold: float = 25.0,
) -> int:
    """Run depth estimation over every image in src_dir, writing
    <stem>_condition.png (normalised depth map, optionally style-matched to
    the renderer's convention) and <stem>_target.jpg (a copy of the source
    photo) into dst_dir. Returns the number of pairs written. Skips a file
    if the pair already exists, so this is safely resumable after an
    interrupted run."""
    from PIL import Image

    dst_dir.mkdir(parents=True, exist_ok=True)
    src_files = sorted(src_dir.glob("*.png"))
    if limit is not None:
        src_files = src_files[:limit]

    n_written = 0
    for i, src_path in enumerate(src_files):
        stem = src_path.stem
        condition_path = dst_dir / f"{stem}_condition.png"
        target_path = dst_dir / f"{stem}_target.jpg"
        if condition_path.exists() and target_path.exists():
            n_written += 1
            continue

        image = Image.open(src_path).convert("RGB")
        depth_out = pipe(image)
        depth_map = depth_out["depth"]  # PIL Image, mode L, already normalised 0-255 by the pipeline
        if style_match:
            depth_map = style_match_to_renderer(depth_map, n_bands, sky_top_frac, sky_abs_threshold)
        depth_map.save(condition_path)
        image.save(target_path, quality=95)
        n_written += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(src_files)} done", flush=True)

    return n_written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rover-corpus",
        type=Path,
        default=Path(__file__).parent.parent.parent / "CycleGAN" / "Data" / "processed" / "rover",
        help="Root of the existing {train,val,test} unpaired rover corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write the new pseudo_paired_controlnet_corpus "
        "(point this at /scratch, not the home-quota Data/ tree).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap images per split, for a smoke test before the full run.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--style-match",
        action="store_true",
        help="Quantise + sky-cutoff the depth map to resemble the real "
        "renderer's condition-map style (see style_match_to_renderer).",
    )
    parser.add_argument("--n-bands", type=int, default=10, help="Discrete grey levels for quantisation.")
    parser.add_argument(
        "--sky-top-frac", type=float, default=0.4, help="Fraction of image height eligible for the sky cutoff."
    )
    parser.add_argument(
        "--sky-abs-threshold",
        type=float,
        default=25.0,
        help="Absolute depth value (0-255) at/below which a top-region pixel becomes sky. "
        "Deliberately absolute, not a percentile, so a crop with no real sky triggers nothing.",
    )
    args = parser.parse_args()

    import torch
    from transformers import pipeline

    print(f"Loading depth model {DEPTH_MODEL} on {args.device}...", flush=True)
    pipe = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=0 if args.device == "cuda" and torch.cuda.is_available() else -1,
    )

    t0 = time.time()
    for split in ("train", "val", "test"):
        src = args.rover_corpus / split
        dst = args.output_dir / split
        if not src.exists():
            print(f"  skipping {split}: {src} does not exist")
            continue
        print(f"Processing split '{split}' ({src}) -> {dst}", flush=True)
        n = build_split(
            pipe,
            src,
            dst,
            limit=args.limit,
            style_match=args.style_match,
            n_bands=args.n_bands,
            sky_top_frac=args.sky_top_frac,
            sky_abs_threshold=args.sky_abs_threshold,
        )
        print(f"  wrote {n} pairs to {dst}", flush=True)

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
