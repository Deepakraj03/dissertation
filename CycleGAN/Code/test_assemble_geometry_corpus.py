from unittest.mock import patch

import numpy as np

from assemble_geometry_corpus import process_dtm_product, sample_camera_positions
from dtm_arrays import DtmArrays
from dtm_coverage import DtmCoverageRecord


def test_sample_camera_positions_count_and_distinctness():
    positions = sample_camera_positions((1000, 1000), margin_px=50, n=30, seed=0)
    assert len(positions) == 30
    assert len(set((r, c) for r, c, h in positions)) == 30  # rows/cols distinct


def test_sample_camera_positions_respects_margin():
    positions = sample_camera_positions((1000, 1000), margin_px=50, n=50, seed=0)
    for row, col, heading in positions:
        assert 50 <= row <= 950
        assert 50 <= col <= 950
        assert 0 <= heading < 360


def test_sample_camera_positions_reproducible_with_same_seed():
    a = sample_camera_positions((500, 500), margin_px=20, n=10, seed=42)
    b = sample_camera_positions((500, 500), margin_px=20, n=10, seed=42)
    assert a == b


def test_sample_camera_positions_too_small_heightmap_returns_empty():
    # margin leaves no valid interior — matches
    # hirise_fullres.sample_patch_positions's convention of returning []
    # rather than raising when the request can't be satisfied.
    positions = sample_camera_positions((80, 80), margin_px=50, n=10, seed=0)
    assert positions == []


SAMPLE_RECORD = DtmCoverageRecord(
    product_id="DTEEC_TEST_0000_0001_L01", dtm_url="https://example/dtm.IMG",
    obs_id_a="ESP_000000_1985", obs_id_b="ESP_000001_1985",
    min_lat=18.0, max_lat=18.1, min_lon=335.0, max_lon=335.1,
    comment="test", files_url="https://example/files",
)


def test_process_dtm_product_saves_qualifying_crops_and_cleans_up(tmp_path):
    scratch_dir = tmp_path / "scratch"
    staging_dir = tmp_path / "staging"

    # High-entropy synthetic terrain (random noise) so entropy filtering
    # doesn't reject every crop, and enough size that camera margin sampling
    # succeeds.
    rng = np.random.default_rng(0)
    heightmap = rng.uniform(0, 5, size=(300, 300)).astype(np.float32)
    albedo = rng.integers(0, 255, size=(300, 300), dtype=np.uint8)

    with patch("assemble_geometry_corpus.fetch_dtm_and_orthos") as mock_fetch, \
         patch("assemble_geometry_corpus.load_dtm_arrays") as mock_load:
        mock_fetch.return_value = {
            "product_id": SAMPLE_RECORD.product_id, "status": "ok",
            "dtm_path": str(scratch_dir / "d.IMG"),
            "ortho_paths": {"ESP_000000_1985": str(scratch_dir / "o.JP2")},
        }
        mock_load.return_value = DtmArrays(
            heightmap=heightmap, albedo=albedo, pixel_scale_m=1.0,
        )
        # process_dtm_product must delete whatever fetch_dtm_and_orthos
        # downloaded — simulate that by actually creating the files it
        # claims to have fetched, so cleanup has something real to remove.
        scratch_dir.mkdir(parents=True)
        (scratch_dir / "d.IMG").write_bytes(b"fake dtm")
        (scratch_dir / "o.JP2").write_bytes(b"fake ortho")

        result = process_dtm_product(SAMPLE_RECORD, scratch_dir, staging_dir,
                                     n_crops=10, seed=0)

    assert result["status"] == "ok"
    assert result["crops_saved"] > 0
    saved_files = list(staging_dir.glob(f"{SAMPLE_RECORD.product_id}_*.png"))
    assert len(saved_files) == result["crops_saved"]
    # Raw downloaded files must be gone once this product's crops are extracted.
    assert not (scratch_dir / "d.IMG").exists()
    assert not (scratch_dir / "o.JP2").exists()


def test_process_dtm_product_reports_fetch_failure(tmp_path):
    with patch("assemble_geometry_corpus.fetch_dtm_and_orthos") as mock_fetch:
        mock_fetch.return_value = {
            "product_id": SAMPLE_RECORD.product_id,
            "status": "dtm_download_failed",
        }
        result = process_dtm_product(SAMPLE_RECORD, tmp_path / "s", tmp_path / "st")
    assert result["status"] == "dtm_download_failed"
    assert result["crops_saved"] == 0
