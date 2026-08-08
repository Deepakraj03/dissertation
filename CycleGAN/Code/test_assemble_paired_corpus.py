# test_assemble_paired_corpus.py
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rasterio

from assemble_paired_corpus import (
    group_products_by_covering_dtm, process_dtm_group, split_pairs_and_move,
)
from parse_rover_pose import RoverPose
from dtm_coverage import DtmCoverageRecord


def _make_pose(product_id, lat, lon):
    return RoverPose(product_id=product_id, sol=46, site=4, drive=2100,
                     latitude=lat, longitude=lon, mast_azimuth_deg=10.0,
                     compass_heading_deg=55.0)


def _make_record(product_id, min_lat, max_lat, min_lon, max_lon):
    return DtmCoverageRecord(
        product_id=product_id, dtm_url="", obs_id_a="", obs_id_b="",
        min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,
        comment="", files_url="",
    )


def test_group_products_by_covering_dtm_groups_and_skips_uncovered():
    records = [_make_record("DTM_A", -5.0, -4.0, 137.0, 138.0)]
    poses = [
        _make_pose("P1", lat=-4.5, lon=137.5),  # covered by DTM_A
        _make_pose("P2", lat=-4.2, lon=137.2),  # also covered by DTM_A
        _make_pose("P3", lat=50.0, lon=10.0),   # not covered by anything
    ]

    grouped = group_products_by_covering_dtm(poses, records)

    assert set(grouped.keys()) == {"DTM_A"}
    assert {p.product_id for p in grouped["DTM_A"]} == {"P1", "P2"}


def test_process_dtm_group_renders_and_saves_pairs(tmp_path, monkeypatch):
    dtm_record = _make_record("DTM_A", -5.0, -4.0, 137.0, 138.0)
    poses = [_make_pose("P1", lat=-4.5, lon=137.5)]

    fake_fetch_result = {
        "product_id": "DTM_A", "status": "ok",
        "dtm_path": str(tmp_path / "DTM_A.IMG"),
        "ortho_paths": {"obs": str(tmp_path / "obs_ORTHO.JP2")},
    }
    # process_dtm_group opens the DTM directly (unmocked) to read its
    # affine transform, so this fixture must be a real raster rasterio can
    # open — not opaque bytes. In production this is always a freshly
    # downloaded genuine DTM; only the unit-test fixture needs this.
    dtm_transform = rasterio.transform.from_origin(137.0, -4.0, 0.001, 0.001)
    with rasterio.open(
        tmp_path / "DTM_A.IMG", "w", driver="GTiff", height=50, width=50,
        count=1, dtype="float32", crs="EPSG:4326", transform=dtm_transform,
    ) as dst:
        dst.write(np.zeros((50, 50), dtype=np.float32), 1)
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")

    fake_arrays = Mock(heightmap=np.zeros((50, 50), dtype=np.float32),
                       albedo=np.full((50, 50), 100, dtype=np.uint8),
                       pixel_scale_m=1.0)

    monkeypatch.setattr("assemble_paired_corpus.fetch_dtm_and_orthos",
                        lambda record, scratch, dest: fake_fetch_result)
    monkeypatch.setattr("assemble_paired_corpus.load_dtm_arrays",
                        lambda dtm_path, ortho_path: fake_arrays)
    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: (25.0, 25.0))
    monkeypatch.setattr("assemble_paired_corpus.compass_heading_to_render_heading",
                        lambda compass_deg, transform: 0.0)
    monkeypatch.setattr("assemble_paired_corpus.fetch_target_photo",
                        lambda product_id, sol, dest_path: dest_path.write_bytes(b"\xff\xd8\xff fake jpg") or True)

    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    result = process_dtm_group(dtm_record, poses, scratch_dir, out_dir)

    assert result["status"] == "ok"
    assert result["pairs_saved"] == 1
    assert (out_dir / "P1_condition.png").exists()
    assert (out_dir / "P1_target.jpg").exists()
    # Raw DTM/ortho files deleted after processing.
    assert not (tmp_path / "DTM_A.IMG").exists()


def test_split_pairs_and_move_keeps_condition_and_target_together(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "out"
    for pid in ["P1", "P2", "P3", "P4"]:
        (staging / f"{pid}_condition.png").write_bytes(b"c")
        (staging / f"{pid}_target.jpg").write_bytes(b"t")

    result = split_pairs_and_move(["P1", "P2", "P3", "P4"], staging, out_dir, seed=0)

    total_condition = sum(1 for split in ["train", "val", "test"]
                          for _ in (out_dir / split).glob("*_condition.png"))
    total_target = sum(1 for split in ["train", "val", "test"]
                       for _ in (out_dir / split).glob("*_target.jpg"))
    assert total_condition == 4
    assert total_target == 4
    # Every pair landed in the same split as its partner.
    for split in ["train", "val", "test"]:
        conditions = {p.stem.removesuffix("_condition")
                     for p in (out_dir / split).glob("*_condition.png")}
        targets = {p.stem.removesuffix("_target")
                  for p in (out_dir / split).glob("*_target.jpg")}
        assert conditions == targets
    assert result["total_pairs"] == 4
