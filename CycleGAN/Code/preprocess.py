"""
Data preprocessing pipeline for Mars I2I translation training.

Converts raw downloaded images (rover NAVCAM + HiRISE browse)
into normalised 256×256 and 512×512 PNG patches ready for training.

Steps:
  1. Validate images (remove corrupt/tiny files)
  2. Convert to grayscale (HiRISE RED channel is monochrome)
  3. Histogram-normalise (Mars images have variable illumination)
  4. Extract overlapping patches with configurable stride
  5. Filter low-information patches (sky, uniform sand)
  6. Save train / val / test splits

Usage:
    python preprocess.py                          # process both domains
    python preprocess.py --domain rover           # rover only
    python preprocess.py --patch-size 512 --stride 256
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent / "Data"
ROVER_RAW  = BASE / "Rover"  / "raw"
HIRISE_RAW = BASE / "HiRISE" / "raw"
PROC_DIR   = BASE / "processed"

SPLITS     = {"train": 0.80, "val": 0.10, "test": 0.10}
MIN_DIM    = 256          # discard images smaller than this
ENTROPY_TH = 4.5          # bits — discard near-uniform patches (sky, blank sand)


# ── Image utilities ───────────────────────────────────────────────────────────

def load_gray(path: Path) -> np.ndarray | None:
    """Load image, convert to uint8 grayscale. Returns None on error."""
    try:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def normalise(arr: np.ndarray) -> np.ndarray:
    """Stretch histogram to [0, 255] with 1% percentile clipping."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi == lo:
        return arr
    arr = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    return arr.astype(np.uint8)


def entropy(arr: np.ndarray) -> float:
    """Shannon entropy of the pixel value histogram (bits)."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 255), density=True)
    hist    = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def extract_patches(arr: np.ndarray, size: int, stride: int) -> list[np.ndarray]:
    """Extract fixed-size patches with given stride."""
    patches = []
    h, w = arr.shape
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            patches.append(arr[y:y+size, x:x+size])
    return patches


def save_patch(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, format="PNG", optimize=False)


def split_and_move(patches_all: list[Path], out_dir: Path,
                   splits: dict = SPLITS) -> dict:
    """Shuffle patches_all and move them into out_dir/{train,val,test} per
    the given ratios. Returns counts per split. Shared by process_domain
    (browse-resolution pipeline) and hirise_fullres.py (real-resolution
    pipeline) so both use identical split behavior."""
    random.shuffle(patches_all)
    n = len(patches_all)
    n_train = int(n * splits["train"])
    n_val   = int(n * splits["val"])

    split_map = {
        "train": patches_all[:n_train],
        "val":   patches_all[n_train:n_train + n_val],
        "test":  patches_all[n_train + n_val:],
    }

    for split_name, paths in split_map.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.move(str(p), split_dir / p.name)

    return {
        "total_patches": n,
        "train":         len(split_map["train"]),
        "val":           len(split_map["val"]),
        "test":          len(split_map["test"]),
    }


# ── Domain processing ─────────────────────────────────────────────────────────

def process_domain(raw_dir: Path, out_dir: Path, domain: str,
                   patch_size: int, stride: int,
                   max_patches: int) -> dict:
    """Process all images in raw_dir, extract patches, save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".jp2"}
    )
    print(f"\n[{domain}] Found {len(img_paths)} source images in {raw_dir}")

    patches_all: list[Path] = []
    skipped_small, skipped_entropy, saved = 0, 0, 0

    for img_path in img_paths:
        if saved >= max_patches:
            break

        arr = load_gray(img_path)
        if arr is None or arr.shape[0] < MIN_DIM or arr.shape[1] < MIN_DIM:
            skipped_small += 1
            continue

        arr = normalise(arr)
        patches = extract_patches(arr, patch_size, stride)

        for i, patch in enumerate(patches):
            if saved >= max_patches:
                break
            if entropy(patch) < ENTROPY_TH:
                skipped_entropy += 1
                continue
            stem = img_path.stem
            # Save to a temp staging dir; moved into splits below
            tmp_dir = out_dir / "_staging"
            tmp_dir.mkdir(exist_ok=True)
            out_path = tmp_dir / f"{stem}_p{i:04d}.png"
            save_patch(patch, out_path)
            patches_all.append(out_path)
            saved += 1

    print(f"  Saved {saved} patches | Skipped (too small): {skipped_small} | "
          f"Skipped (low entropy): {skipped_entropy}")

    # ── Train / val / test split ──────────────────────────────────────────────
    split_counts = split_and_move(patches_all, out_dir)

    # Remove empty staging dir
    tmp_dir = out_dir / "_staging"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    return {
        "domain":        domain,
        "total_patches": split_counts["total_patches"],
        "train":         split_counts["train"],
        "val":           split_counts["val"],
        "test":          split_counts["test"],
        "patch_size":    patch_size,
        "stride":        stride,
        "out_dir":       str(out_dir),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",     choices=["rover", "hirise", "both"],
                        default="both")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride",     type=int, default=128)
    parser.add_argument("--max-patches", type=int, default=50_000,
                        help="Max patches per domain")
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    stats = []

    if args.domain in ("rover", "both"):
        s = process_domain(
            raw_dir     = ROVER_RAW,
            out_dir     = PROC_DIR / "rover",
            domain      = "rover",
            patch_size  = args.patch_size,
            stride      = args.stride,
            max_patches = args.max_patches,
        )
        stats.append(s)

    if args.domain in ("hirise", "both"):
        s = process_domain(
            raw_dir     = HIRISE_RAW,
            out_dir     = PROC_DIR / "hirise",
            domain      = "hirise",
            patch_size  = args.patch_size,
            stride      = args.stride,
            max_patches = args.max_patches,
        )
        stats.append(s)

    report_path = PROC_DIR / "preprocessing_report.json"
    report_path.write_text(json.dumps(stats, indent=2))
    print(f"\nPreprocessing complete. Report: {report_path}")
    for s in stats:
        print(f"  [{s['domain']}] train={s['train']}, val={s['val']}, test={s['test']}")


if __name__ == "__main__":
    main()
