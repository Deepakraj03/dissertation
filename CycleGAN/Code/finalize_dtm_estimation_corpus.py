"""Merge all per-shard manifests from assemble_dtm_estimation_corpus.py's
--num-shards runs into one manifest.csv, and build the real train/val/test
split across the union of every shard's productive products -- run once,
after every shard has finished (or been resumed to completion)."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from assemble_dtm_estimation_corpus import split_patches_by_product


def find_shard_manifests(out_dir: Path) -> list[Path]:
    """Real per-shard manifest files, matching the naming convention
    assemble_dtm_estimation_corpus.py writes (manifest_shard<i>of<n>.csv)
    -- excludes the merged manifest.csv this script itself produces, so
    re-running finalization doesn't fold a stale merged output back in."""
    return sorted(out_dir.glob("manifest_shard*of*.csv"))


def merge_shard_manifests(shard_manifest_paths: list[Path],
                          ) -> tuple[list[dict], list[str]]:
    """Combine every shard's rows into one list, and return the real
    product IDs that saved at least one patch anywhere across all shards
    -- the set split_patches_by_product needs to build the final
    train/val/test split."""
    all_rows = []
    products_with_patches = []
    for path in shard_manifest_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
                if int(row.get("patches_saved") or 0) > 0:
                    products_with_patches.append(row["product_id"])
    return all_rows, products_with_patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default=None,
                        help="Same --out-root the shard jobs were run with")
    args = parser.parse_args()

    root = Path(args.out_root) if args.out_root else Path(__file__).parent.parent
    out_dir = root / "Data" / "processed" / "dtm_estimation_corpus"
    staging_dir = out_dir / "_staging"

    shard_manifests = find_shard_manifests(out_dir)
    print(f"Found {len(shard_manifests)} shard manifest(s): "
         f"{[p.name for p in shard_manifests]}")
    if not shard_manifests:
        print("No shard manifests found -- nothing to finalize.")
        return

    all_rows, products_with_patches = merge_shard_manifests(shard_manifests)
    print(f"{len(all_rows)} total product result(s) across all shards, "
         f"{len(products_with_patches)} product(s) with at least one patch")

    split_counts = split_patches_by_product(products_with_patches, staging_dir, out_dir)
    print(f"Final split: {split_counts}")

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["product_id", "status", "patches_saved", "roughness"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Merged manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
