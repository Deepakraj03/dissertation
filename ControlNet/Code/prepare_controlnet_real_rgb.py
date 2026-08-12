"""
Convert the paired-corpus test split's real target photos (JPEGs) to RGB
PNGs, named to match generate_controlnet_translations.py's output naming
(<product_id>.png) -- the real-image side of the ControlNet FID comparison.

Usage:
    python prepare_controlnet_real_rgb.py
"""

import argparse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
DEFAULT_INPUT_DIR = ROOT / "Data" / "processed" / "paired_controlnet_corpus" / "test"
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "eval" / "real_controlnet_rgb"


def convert_directory_to_rgb(input_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(input_dir.glob("*_target.jpg")):
        out_name = path.name.removesuffix("_target.jpg") + ".png"
        Image.open(path).convert("RGB").save(output_dir / out_name)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    count = convert_directory_to_rgb(Path(args.input_dir), Path(args.output_dir))
    print(f"Converted {count} real images -> {args.output_dir}")


if __name__ == "__main__":
    main()
