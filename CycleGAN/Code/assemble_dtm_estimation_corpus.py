"""Orchestrate the single-image DTM estimation training corpus build: for
every real HiRISE stereo DTM product in the global catalog, download it,
tile it into aligned (orthoimage, height-map) patches, keep only patches
that pass the hillshade-vs-ortho alignment screen, augment survivors via
horizontal/vertical flip, and assemble a train/val/test split -- same
one-product-at-a-time-then-delete lifecycle as assemble_geometry_corpus.py.

Deliberately not roughness-filtered: the corpus is meant to be as large as
real global coverage allows (2026-08-14 design decision), so every product
that fetches successfully contributes patches regardless of terrain type.
Each product's terrain roughness is still recorded in the manifest, for
potential future loss-weighting rather than corpus-inclusion filtering."""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dtm_coverage import DtmCoverageRecord, query_global_dtm_coverage
from dtm_arrays import load_dtm_arrays
from dtm_quality_screen import patch_alignment_score
from fetch_hirise_dtm import fetch_dtm_and_orthos
from preprocess import extract_patches, save_patch
from terrain_roughness import compute_terrain_roughness


def augment_patch_pair(height_patch: np.ndarray, ortho_patch: np.ndarray,
                       ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The paper's own augmentation: horizontal + vertical flip, applied in
    lockstep to both the height and ortho patch so they stay aligned."""
    return {
        "orig": (height_patch, ortho_patch),
        "hflip": (np.fliplr(height_patch), np.fliplr(ortho_patch)),
        "vflip": (np.flipud(height_patch), np.flipud(ortho_patch)),
        "hvflip": (np.flipud(np.fliplr(height_patch)), np.flipud(np.fliplr(ortho_patch))),
    }


def process_dtm_product_for_estimation(record: DtmCoverageRecord, scratch_dir: Path,
                                       staging_dir: Path, patch_size: int = 256,
                                       stride: int = 256,
                                       alignment_th: float = 0.4) -> dict:
    """Download record's DTM+ortho, tile into aligned patches, keep only
    patches whose hillshade genuinely resembles the real ortho (screens out
    misaligned/degenerate pairs), augment survivors 4x, save to
    staging_dir, delete the raw downloaded files, and return a status dict
    including the product's terrain roughness (metadata only, not a
    filter — see module docstring)."""
    fetch_result = fetch_dtm_and_orthos(record, scratch_dir, scratch_dir)
    if fetch_result["status"] != "ok":
        return {"product_id": record.product_id, "status": fetch_result["status"],
               "patches_saved": 0, "roughness": None}

    dtm_path = Path(fetch_result["dtm_path"])
    ortho_path = Path(next(iter(fetch_result["ortho_paths"].values())))

    try:
        arrays = load_dtm_arrays(dtm_path, ortho_path)
        roughness = compute_terrain_roughness(arrays.heightmap, arrays.pixel_scale_m)

        staging_dir.mkdir(parents=True, exist_ok=True)
        height_patches = extract_patches(arrays.heightmap, patch_size, stride)
        ortho_patches = extract_patches(arrays.albedo, patch_size, stride)

        saved = 0
        for i, (h_patch, o_patch) in enumerate(zip(height_patches, ortho_patches)):
            if patch_alignment_score(h_patch, o_patch, arrays.pixel_scale_m) < alignment_th:
                continue
            for aug_name, (h_aug, o_aug) in augment_patch_pair(h_patch, o_patch).items():
                base = f"{record.product_id}_p{i:04d}_{aug_name}"
                np.save(staging_dir / f"{base}_target.npy", h_aug.astype(np.float32))
                save_patch(o_aug, staging_dir / f"{base}_condition.png")
                saved += 1

        return {"product_id": record.product_id, "status": "ok",
               "patches_saved": saved, "roughness": roughness}
    finally:
        dtm_path.unlink(missing_ok=True)
        for p in fetch_result["ortho_paths"].values():
            Path(p).unlink(missing_ok=True)


def select_shard(records: list, shard_index: int, num_shards: int) -> list:
    """Round-robin (interleaved) partition of records across num_shards --
    every record appears in exactly one shard, and shard sizes never
    differ by more than one even when the count doesn't divide evenly.
    Used to split the global catalog across parallel SLURM array tasks;
    a contiguous chunking would work too, but interleaving means a shard
    isn't accidentally starved if the catalog happens to have a productive
    run of products clustered together."""
    return records[shard_index::num_shards]


def load_completed_product_ids(manifest_path: Path) -> set[str]:
    """Real product IDs already recorded in an existing manifest CSV --
    used to skip re-processing them on a resumed/restarted run, since a
    killed job would otherwise redo already-fetched work from scratch."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="") as f:
        return {row["product_id"] for row in csv.DictReader(f)}


def append_manifest_row(manifest_path: Path, row: dict) -> None:
    """Append one product's result to the manifest CSV immediately after
    it's processed, so partial progress survives an interrupted run --
    rather than only writing the whole manifest at the very end, which
    loses everything if the job is killed partway (the 2026-08-14 smoke
    test's real ~7-day full-catalog extrapolation makes this a real risk,
    not a hypothetical one)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["product_id", "status", "patches_saved", "roughness"])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def split_patches_by_product(product_ids: list[str], staging_dir: Path,
                             out_dir: Path, seed: int = 0,
                             train_frac: float = 0.8, val_frac: float = 0.1,
                             ) -> dict[str, int]:
    """Shuffle DTM PRODUCTS (not individual patches) into train/val/test,
    then move every patch file whose filename is prefixed by that
    product's ID into the same split. Patches from one DTM are spatially
    correlated, and flip-augmented siblings are near-duplicates -- splitting
    at the patch level would leak correlated content across splits."""
    ids = list(product_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }

    counts = {}
    for split_name, split_ids in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for pid in split_ids:
            for f in staging_dir.glob(f"{pid}_*"):
                shutil.move(str(f), str(split_dir / f.name))
                moved += 1
        counts[split_name] = moved
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--alignment-th", type=float, default=0.4)
    parser.add_argument("--band-deg", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Consider at most this many DTM products globally,"
                             " before sharding (smoke testing)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="This task's shard, for parallel SLURM array runs")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shards the catalog is split across")
    parser.add_argument("--out-root", type=str, default=None,
                        help="Root directory for Data/ output (default: this repo's"
                             " CycleGAN/ directory). Override to a fast scratch"
                             " filesystem on clusters where the default is slow NFS"
                             " -- this corpus can run to tens of thousands of patches.")
    args = parser.parse_args()

    root = Path(args.out_root) if args.out_root else Path(__file__).parent.parent

    print("Querying global HiRISE stereo DTM catalog…")
    records = query_global_dtm_coverage(band_deg=args.band_deg)
    print(f"Found {len(records)} DTM product(s) globally")
    if args.limit:
        records = records[:args.limit]
        print(f"Limited to {len(records)} product(s) for this run")
    records = select_shard(records, args.shard_index, args.num_shards)
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(records)} product(s) assigned")

    scratch_dir = root / "Data" / "_dtm_estimation_scratch" / f"shard{args.shard_index}"
    staging_dir = root / "Data" / "processed" / "dtm_estimation_corpus" / "_staging"
    out_dir = root / "Data" / "processed" / "dtm_estimation_corpus"
    manifest_path = out_dir / f"manifest_shard{args.shard_index}of{args.num_shards}.csv"

    already_done = load_completed_product_ids(manifest_path)
    if already_done:
        print(f"Resuming: {len(already_done)} product(s) already in this shard's"
             f" manifest, skipping them")

    for record in records:
        if record.product_id in already_done:
            continue
        print(f"Processing {record.product_id}…")
        result = process_dtm_product_for_estimation(
            record, scratch_dir, staging_dir,
            patch_size=args.patch_size, stride=args.stride,
            alignment_th=args.alignment_th,
        )
        print(f"  {result['status']} — {result.get('patches_saved', 0)} patches"
             f" (roughness={result.get('roughness')})")
        append_manifest_row(manifest_path, result)

    print(f"\nShard {args.shard_index}/{args.num_shards} finished. Manifest at {manifest_path}")
    print("Run finalize_dtm_estimation_corpus.py once every shard has finished"
         " to merge manifests and build the train/val/test split.")


if __name__ == "__main__":
    main()
