"""
Run the fine-tuned ControlNet (Phase 1 geometry-mediated corpus) over the
held-out paired-corpus test split's condition maps, producing RGB
translations for FID evaluation against the same real target photos --
mirrors generate_translations.py's CycleGAN role, but for the ControlNet
model in the three-way comparison (Track 1 / Track 2 / ControlNet).

Usage:
    python generate_controlnet_translations.py \\
        --controlnet-checkpoint ../checkpoints/controlnet_phase1 \\
        --base-model stable-diffusion-v1-5/stable-diffusion-v1-5
"""

import argparse
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

ROOT = Path(__file__).parent.parent
DEFAULT_INPUT_DIR = ROOT / "Data" / "processed" / "paired_controlnet_corpus" / "test"
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "eval" / "generated_controlnet_rgb"
CAPTION = "a photorealistic Mars rover Navcam photograph of the martian surface"


def load_pipeline(controlnet_checkpoint: Path, base_model: str,
                  device: torch.device) -> StableDiffusionControlNetPipeline:
    controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe


def generate_all(pipe: StableDiffusionControlNetPipeline, input_dir: Path,
                 output_dir: Path, num_inference_steps: int = 30,
                 seed: int = 42, limit: int | None = None) -> int:
    """Run pipe over every *_condition.png in input_dir, save RGB outputs
    to output_dir named to match the source condition map. Clears any
    pre-existing PNGs first -- same stale-output convention as
    generate_translations.py (a prior contamination bug in this project's
    history)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()

    files = sorted(input_dir.glob("*_condition.png"))
    if limit:
        files = files[:limit]

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    count = 0
    for path in files:
        condition_image = Image.open(path).convert("RGB")
        result = pipe(
            CAPTION, image=condition_image,
            num_inference_steps=num_inference_steps, generator=generator,
        ).images[0]
        out_name = path.name.removesuffix("_condition.png") + ".png"
        result.save(output_dir / out_name)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet-checkpoint", type=str, required=True)
    parser.add_argument("--base-model", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipe = load_pipeline(Path(args.controlnet_checkpoint), args.base_model, device)
    count = generate_all(pipe, Path(args.input_dir), Path(args.output_dir), limit=args.limit)
    print(f"Generated {count} translations -> {args.output_dir}")


if __name__ == "__main__":
    main()
