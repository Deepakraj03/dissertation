import json
from pathlib import Path

import pytest

from coregister_ctx_hirise import find_downloaded_images, process_one
from hirise_index import Footprint


def test_find_downloaded_images_reads_manifest(tmp_path):
    # download_hirise.py writes a single flat manifest.json directly in
    # Data/HiRISE/raw/, not one per region subdirectory — verified
    # against Code/download_hirise.py's actual DATA_DIR/manifest_path
    # logic after the live smoke test found 0 images with the wrong
    # (nested) assumption.
    region_dir = tmp_path / "oxia_planum"
    region_dir.mkdir()
    jpeg_path = region_dir / "ESP_037235_1985_RED.browse.jpg"
    jpeg_path.write_bytes(b"\xff\xd8\xff fake jpeg")
    manifest = [{"obs_id": "ESP_037235_1985", "path": str(jpeg_path), "size_kb": 500}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    results = find_downloaded_images(tmp_path)

    assert results == [("ESP_037235_1985", jpeg_path)]


def test_find_downloaded_images_skips_missing_files(tmp_path):
    region_dir = tmp_path / "oxia_planum"
    region_dir.mkdir()
    missing_path = region_dir / "ESP_999999_1985_RED.browse.jpg"
    manifest = [{"obs_id": "ESP_999999_1985", "path": str(missing_path), "size_kb": 500}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    results = find_downloaded_images(tmp_path)

    assert results == []


def test_process_one_reports_no_index_match(tmp_path):
    ok, status = process_one(
        obs_id="ESP_037235_1985",
        hirise_path=tmp_path / "fake.jpg",
        index={},  # empty index -> no match, regardless of file existence
        tile_dir=tmp_path / "tiles",
        pairs_dir=tmp_path / "pairs",
    )
    assert ok is False
    assert status == "no_index_match"


def test_lonlat_to_projected_matches_real_tile_bounds():
    # Empirically verified during planning: tile
    # MurrayLab_GlobalCTXMosaic_V01_E-060_N-56 (downloaded and inspected
    # with rasterio) has bounds left=-3556481.85, bottom=-3319383.06 for
    # its nominal SW corner at lon=-60, lat=-56 degrees. Confirms the
    # projected-CRS conversion formula before it's used inside
    # crop_ctx_to_footprint.
    from coregister_ctx_hirise import lonlat_to_projected

    x, y = lonlat_to_projected(-60.0, -56.0)
    assert abs(x - (-3556481.85)) < 200  # within ~40 pixels at 5m/px
    assert abs(y - (-3319383.06)) < 200


def test_process_one_reports_multi_tile_footprint(tmp_path):
    # Footprint straddling the lon=-60 cell boundary -> covers 2 tiles.
    footprint = Footprint(
        obs_id="ESP_TEST_0001",
        min_lat=-55.9, max_lat=-55.1,
        min_lon=-60.5, max_lon=-59.5,
        projection="EQUIRECTANGULAR",
        file_name_spec="RDR/ESP/ORB_TEST/ESP_TEST_0001/ESP_TEST_0001_RED.JP2",
    )
    ok, status = process_one(
        obs_id="ESP_TEST_0001",
        hirise_path=tmp_path / "fake.jpg",
        index={"ESP_TEST_0001": footprint},
        tile_dir=tmp_path / "tiles",
        pairs_dir=tmp_path / "pairs",
    )
    assert ok is False
    assert status == "spans_multiple_tiles"
