"""
Run the trained CycleGAN HiRISE->Rover generator (G_H2R) over the HiRISE
test set, producing RGB-converted translations for FID evaluation.

Usage:
    python generate_translations.py --checkpoint ../checkpoints/epoch_025.pt
    python generate_translations.py --checkpoint ../checkpoints/epoch_025.pt --limit 20
"""

import argparse
import sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from cyclegan import Generator
from image_utils import denormalize_to_uint8, save_rgb_png, to_rgb

ROOT = Path(__file__).parent.parent
DEFAULT_INPUT_DIR = ROOT / "Data" / "processed" / "hirise" / "test"
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "eval" / "generated_rover_rgb"

# Matches Code/train_cyclegan.py's UnpairedDataset.tf, minus Resize/
# CenterCrop (patches are already 256x256) and RandomHorizontalFlip
# (inference should be deterministic, not augmented).
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),                 # [0, 1]
    transforms.Normalize([0.5], [0.5]),    # -> [-1, 1]
])


def load_generator(checkpoint_path: Path, device: torch.device) -> Generator:
    """Load G_H2R weights from a train_cyclegan.py checkpoint."""
    generator = Generator(in_ch=1, out_ch=1, ngf=64).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt["G_H2R"])
    generator.eval()
    return generator


def generate_all(generator: Generator, input_dir: Path, output_dir: Path,
                 device: torch.device, limit: int | None = None) -> int:
    """Run generator over every PNG in input_dir, save RGB outputs to
    output_dir. Returns the number of images processed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.png"))
    if limit:
        files = files[:limit]

    count = 0
    with torch.no_grad():
        for path in files:
            img = Image.open(path)
            tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)
            fake = generator(tensor)[0]  # (1, H, W)
            array = denormalize_to_uint8(fake)
            rgb = to_rgb(array)
            save_rgb_png(rgb, output_dir / path.name)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N images (smoke test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    generator = load_generator(Path(args.checkpoint), device)
    count = generate_all(generator, Path(args.input_dir), Path(args.output_dir),
                         device, limit=args.limit)
    print(f"Generated {count} translations -> {args.output_dir}")


if __name__ == "__main__":
    main()
