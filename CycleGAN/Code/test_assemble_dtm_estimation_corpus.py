from unittest.mock import patch

import numpy as np
import pytest

from assemble_dtm_estimation_corpus import (
    append_manifest_row, augment_patch_pair, load_completed_product_ids,
    process_dtm_product_for_estimation, select_shard, split_patches_by_product,
)
from dtm_arrays import DtmArrays
from dtm_coverage import DtmCoverageRecord
from dtm_quality_screen import compute_shaded_relief

SAMPLE_RECORD = DtmCoverageRecord(
    product_id="DTEEC_TEST_0000_0001_L01", dtm_url="https://example/dtm.IMG",
    obs_id_a="ESP_000000_1985", obs_id_b="ESP_000001_1985",
    min_lat=18.0, max_lat=18.1, min_lon=335.0, max_lon=335.1,
    comment="test", files_url="https://example/files",
)


def _fetch_mock_return(scratch_dir):
    return {
        "product_id": SAMPLE_RECORD.product_id, "status": "ok",
        "dtm_path": str(scratch_dir / "d.IMG"),
        "ortho_paths": {
            "ESP_000000_1985": str(scratch_dir / "o1.JP2"),
            "ESP_000001_1985": str(scratch_dir / "o2.JP2"),
        },
    }


def _bump_heightmap(size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2, size / 2
    return 10.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (size * 2))


def test_augment_patch_pair_produces_four_correct_flips():
    height = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ortho = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    augmented = augment_patch_pair(height, ortho)

    assert set(augmented.keys()) == {"orig", "hflip", "vflip", "hvflip"}
    np.testing.assert_array_equal(augmented["orig"][0], height)
    np.testing.assert_array_equal(augmented["hflip"][0], np.array([[2.0, 1.0], [4.0, 3.0]]))
    np.testing.assert_array_equal(augmented["vflip"][0], np.array([[3.0, 4.0], [1.0, 2.0]]))
    np.testing.assert_array_equal(augmented["hvflip"][0], np.array([[4.0, 3.0], [2.0, 1.0]]))
    # ortho gets the same transform applied in lockstep
    np.testing.assert_array_equal(augmented["hflip"][1], np.array([[20, 10], [40, 30]]))


def test_process_dtm_product_for_estimation_saves_aligned_patches_augmented(tmp_path):
    scratch_dir = tmp_path / "scratch"
    staging_dir = tmp_path / "staging"
    scratch_dir.mkdir()
    (scratch_dir / "d.IMG").write_bytes(b"fake dtm")

    size = 256
    heightmap = _bump_heightmap(size)
    # Ortho constructed FROM the heightmap's own shaded relief so the
    # alignment screen genuinely passes -- a real correlated pair, not an
    # arbitrary "hope it passes" fixture.
    ortho = compute_shaded_relief(heightmap, pixel_scale_m=1.0)

    with patch("assemble_dtm_estimation_corpus.fetch_dtm_and_orthos") as mock_fetch, \
         patch("assemble_dtm_estimation_corpus.load_dtm_arrays") as mock_load:
        mock_fetch.return_value = _fetch_mock_return(scratch_dir)
        mock_load.return_value = DtmArrays(heightmap=heightmap, albedo=ortho, pixel_scale_m=1.0)

        result = process_dtm_product_for_estimation(
            SAMPLE_RECORD, scratch_dir, staging_dir,
            patch_size=size, stride=size, alignment_th=0.4,
        )

    assert result["status"] == "ok"
    # One 256x256 patch from a 256x256 source, x4 augmentations.
    assert result["patches_saved"] == 4
    assert result["roughness"] is not None and result["roughness"] > 0.0

    condition_files = list(staging_dir.glob("*_condition.png"))
    target_files = list(staging_dir.glob("*_target.npy"))
    assert len(condition_files) == 4
    assert len(target_files) == 4
    for f in target_files:
        arr = np.load(f)
        assert arr.dtype == np.float32
        assert arr.shape == (size, size)

    assert not (scratch_dir / "d.IMG").exists()


def test_process_dtm_product_for_estimation_skips_misaligned_patches(tmp_path):
    scratch_dir = tmp_path / "scratch"
    staging_dir = tmp_path / "staging"
    scratch_dir.mkdir()
    (scratch_dir / "d.IMG").write_bytes(b"fake dtm")

    size = 256
    heightmap = _bump_heightmap(size)
    rng = np.random.default_rng(0)
    unrelated_ortho = rng.integers(0, 255, size=(size, size), dtype=np.uint8)

    with patch("assemble_dtm_estimation_corpus.fetch_dtm_and_orthos") as mock_fetch, \
         patch("assemble_dtm_estimation_corpus.load_dtm_arrays") as mock_load:
        mock_fetch.return_value = _fetch_mock_return(scratch_dir)
        mock_load.return_value = DtmArrays(
            heightmap=heightmap, albedo=unrelated_ortho, pixel_scale_m=1.0)

        result = process_dtm_product_for_estimation(
            SAMPLE_RECORD, scratch_dir, staging_dir,
            patch_size=size, stride=size, alignment_th=0.4,
        )

    assert result["status"] == "ok"
    assert result["patches_saved"] == 0
    assert list(staging_dir.glob("*_condition.png")) == []


def test_process_dtm_product_for_estimation_cleans_up_on_fetch_failure(tmp_path):
    scratch_dir = tmp_path / "scratch"
    staging_dir = tmp_path / "staging"
    scratch_dir.mkdir()

    with patch("assemble_dtm_estimation_corpus.fetch_dtm_and_orthos") as mock_fetch:
        mock_fetch.return_value = {
            "product_id": SAMPLE_RECORD.product_id,
            "status": "dtm_download_failed",
        }
        result = process_dtm_product_for_estimation(
            SAMPLE_RECORD, scratch_dir, staging_dir, patch_size=256, stride=256,
        )

    assert result["status"] == "dtm_download_failed"
    assert result["patches_saved"] == 0
    assert result["roughness"] is None


def test_split_patches_by_product_keeps_one_products_patches_together(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "out"

    product_ids = [f"PROD{i}" for i in range(10)]
    for pid in product_ids:
        for aug in ["orig", "hflip", "vflip", "hvflip"]:
            (staging / f"{pid}_p0000_{aug}_condition.png").write_bytes(b"c")
            (staging / f"{pid}_p0000_{aug}_target.npy").write_bytes(b"t")

    result = split_patches_by_product(product_ids, staging, out_dir, seed=0)

    assert sum(result.values()) == 10 * 4 * 2  # condition + target per aug

    for split in ["train", "val", "test"]:
        split_dir = out_dir / split
        present_pids = {f.name.split("_p")[0] for f in split_dir.glob("*")}
        # Every file found under a product's ID prefix in this split must
        # belong to a product actually assigned to this split -- i.e. no
        # product's patches leaked into a different split than the rest of
        # that product's own patches.
        for pid in present_pids:
            all_this_products_files_here = list(split_dir.glob(f"{pid}_*"))
            # 4 augmentations x 2 file types (condition + target) = 8
            assert len(all_this_products_files_here) == 8


def test_split_patches_by_product_no_product_appears_in_two_splits(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "out"

    product_ids = [f"PROD{i}" for i in range(10)]
    for pid in product_ids:
        (staging / f"{pid}_p0000_orig_condition.png").write_bytes(b"c")
        (staging / f"{pid}_p0000_orig_target.npy").write_bytes(b"t")

    split_patches_by_product(product_ids, staging, out_dir, seed=0)

    seen_in = {}
    for split in ["train", "val", "test"]:
        for f in (out_dir / split).glob("*_condition.png"):
            pid = f.name.split("_p")[0]
            assert pid not in seen_in, f"{pid} appeared in both {seen_in.get(pid)} and {split}"
            seen_in[pid] = split


def test_select_shard_partitions_without_overlap_or_gaps():
    records = [f"R{i}" for i in range(23)]
    num_shards = 4

    shards = [select_shard(records, i, num_shards) for i in range(num_shards)]

    reconstructed = sorted(r for shard in shards for r in shard)
    assert reconstructed == sorted(records)  # every record appears exactly once
    # Interleaved (round-robin) assignment keeps shard sizes balanced --
    # no shard should be starved just because 23 doesn't divide evenly by 4.
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_select_shard_single_shard_returns_everything():
    records = [f"R{i}" for i in range(10)]
    assert select_shard(records, 0, 1) == records


def test_load_completed_product_ids_empty_when_manifest_missing(tmp_path):
    assert load_completed_product_ids(tmp_path / "does_not_exist.csv") == set()


def test_load_completed_product_ids_reads_real_written_rows(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    append_manifest_row(manifest_path,
                        {"product_id": "A", "status": "ok",
                         "patches_saved": 4, "roughness": 0.1})
    append_manifest_row(manifest_path,
                        {"product_id": "B", "status": "dtm_download_failed",
                         "patches_saved": 0, "roughness": None})

    assert load_completed_product_ids(manifest_path) == {"A", "B"}


def test_append_manifest_row_writes_header_only_once(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    append_manifest_row(manifest_path,
                        {"product_id": "A", "status": "ok",
                         "patches_saved": 4, "roughness": 0.1})
    append_manifest_row(manifest_path,
                        {"product_id": "B", "status": "ok",
                         "patches_saved": 2, "roughness": 0.2})

    lines = manifest_path.read_text().splitlines()
    assert lines[0] == "product_id,status,patches_saved,roughness"
    assert len(lines) == 3  # header + 2 data rows, no duplicate headers


def test_append_manifest_row_survives_across_two_separate_calls_simulating_resume(tmp_path):
    # Simulates a killed-and-restarted run: first "session" writes one row,
    # then a fresh call (as a resumed process would make) appends another --
    # the header must not be duplicated and both rows must be readable.
    manifest_path = tmp_path / "manifest.csv"
    append_manifest_row(manifest_path,
                        {"product_id": "A", "status": "ok",
                         "patches_saved": 4, "roughness": 0.1})

    # "Restart": a fresh process re-opens the same path and appends more.
    append_manifest_row(manifest_path,
                        {"product_id": "C", "status": "ok",
                         "patches_saved": 9, "roughness": 0.3})

    assert load_completed_product_ids(manifest_path) == {"A", "C"}
    lines = manifest_path.read_text().splitlines()
    assert lines.count("product_id,status,patches_saved,roughness") == 1
