"""
Convert the real rover test set (greyscale PNGs) to RGB copies, so both
sides of the FID comparison are in the 3-channel format clean-fid expects.

Usage:
    python prepare_real_rgb.py
    python prepare_real_rgb.py --limit 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from image_utils import save_rgb_png, to_rgb

ROOT = Path(__file__).parent.parent
DEFAULT_INPUT_DIR = ROOT / "Data" / "processed" / "rover" / "test"
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "eval" / "real_rover_rgb"


def convert_directory_to_rgb(input_dir: Path, output_dir: Path,
                             limit: int | None = None) -> int:
    """Convert every PNG in input_dir from greyscale to RGB (channel
    replication), saving to output_dir. Returns the count converted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.png"))
    if limit:
        files = files[:limit]

    count = 0
    for path in files:
        array = np.array(Image.open(path).convert("L"))
        rgb = to_rgb(array)
        save_rgb_png(rgb, output_dir / path.name)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    count = convert_directory_to_rgb(Path(args.input_dir), Path(args.output_dir),
                                     limit=args.limit)
    print(f"Converted {count} real images -> {args.output_dir}")


if __name__ == "__main__":
    main()
