# test_assemble_paired_corpus.py
import csv
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rasterio

from assemble_paired_corpus import (
    group_products_by_covering_dtm, process_dtm_group, split_pairs_and_move,
    render_pose_condition_map, gather_candidate_poses, query_region_dtm_coverage,
    build_arg_parser, write_manifest,
)
from parse_rover_pose import RoverPose
from dtm_coverage import DtmCoverageRecord


def _make_pose(product_id, lat, lon, pitch_deg=0.0):
    return RoverPose(product_id=product_id, sol=46, site=4, drive=2100,
                     latitude=lat, longitude=lon,
                     compass_heading_deg=55.0, pitch_deg=pitch_deg)


def _make_record(product_id, min_lat, max_lat, min_lon, max_lon,
                 obs_id_a="", obs_id_b=""):
    return DtmCoverageRecord(
        product_id=product_id, dtm_url="", obs_id_a=obs_id_a, obs_id_b=obs_id_b,
        min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,
        comment="", files_url="",
    )


def _make_varied_arrays(shape=(50, 50), seed=0):
    """A heightmap/albedo pair with enough real variation that render_ground_view's
    output clears ENTROPY_TH — same rng-based construction test_assemble_geometry_corpus.py
    already uses for this. A flat/constant fixture would render as a
    uniform (single-value) condition map, which is exactly what the new
    ground_entropy filter is meant to reject, so tests exercising a normal
    save need real texture, not a flat mock."""
    rng = np.random.default_rng(seed)
    heightmap = rng.uniform(0, 5, size=shape).astype(np.float32)
    albedo = rng.integers(0, 255, size=shape, dtype=np.uint8)
    return Mock(heightmap=heightmap, albedo=albedo, pixel_scale_m=1.0)


def _write_fake_dtm_raster(path):
    """process_dtm_group opens the DTM directly (unmocked) to read its
    affine transform, so any fixture standing in for a real DTM must be a
    raster rasterio can open — not opaque bytes. In production this is
    always a freshly downloaded genuine DTM; only unit-test fixtures need
    this."""
    transform = rasterio.transform.from_origin(137.0, -4.0, 0.001, 0.001)
    with rasterio.open(
        path, "w", driver="GTiff", height=50, width=50,
        count=1, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.zeros((50, 50), dtype=np.float32), 1)


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
    _write_fake_dtm_raster(tmp_path / "DTM_A.IMG")
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")

    fake_arrays = _make_varied_arrays()

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
    assert result["saved_ids"] == ["P1"]
    assert (out_dir / "P1_condition.png").exists()
    assert (out_dir / "P1_target.jpg").exists()
    # Raw DTM/ortho files deleted after processing.
    assert not (tmp_path / "DTM_A.IMG").exists()


def test_process_dtm_group_saved_ids_reflect_actual_successes_not_position(
        tmp_path, monkeypatch):
    """Regression test: an earlier version of main() assumed the first N
    poses in a group were the N poses that got saved (group_poses[:pairs_saved]).
    That's wrong whenever a skip happens before a later success — this test
    puts a pose that fails ahead of one that succeeds and checks saved_ids
    names the actual survivor, not a positional slice."""
    dtm_record = _make_record("DTM_A", -5.0, -4.0, 137.0, 138.0)
    poses = [_make_pose("P1", lat=-4.5, lon=137.5),
            _make_pose("P2", lat=-4.5, lon=137.5)]

    fake_fetch_result = {
        "product_id": "DTM_A", "status": "ok",
        "dtm_path": str(tmp_path / "DTM_A.IMG"),
        "ortho_paths": {"obs": str(tmp_path / "obs_ORTHO.JP2")},
    }
    _write_fake_dtm_raster(tmp_path / "DTM_A.IMG")
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")

    fake_arrays = _make_varied_arrays()

    def fake_fetch_target_photo(product_id, sol, dest_path):
        if product_id == "P1":
            return False  # P1's photo fetch fails — should be skipped
        dest_path.write_bytes(b"\xff\xd8\xff fake jpg")
        return True

    monkeypatch.setattr("assemble_paired_corpus.fetch_dtm_and_orthos",
                        lambda record, scratch, dest: fake_fetch_result)
    monkeypatch.setattr("assemble_paired_corpus.load_dtm_arrays",
                        lambda dtm_path, ortho_path: fake_arrays)
    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: (25.0, 25.0))
    monkeypatch.setattr("assemble_paired_corpus.compass_heading_to_render_heading",
                        lambda compass_deg, transform: 0.0)
    monkeypatch.setattr("assemble_paired_corpus.fetch_target_photo",
                        fake_fetch_target_photo)

    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    result = process_dtm_group(dtm_record, poses, scratch_dir, out_dir)

    assert result["pairs_saved"] == 1
    assert result["saved_ids"] == ["P2"]
    assert not (out_dir / "P1_condition.png").exists()
    assert (out_dir / "P2_condition.png").exists()
    assert (out_dir / "P2_target.jpg").exists()


def test_process_dtm_group_skips_steep_pitch_before_rendering(tmp_path, monkeypatch):
    """2026-08-09 finding: even with correct pitch modeling, a real steep
    down-look pose (close range) renders as a low-detail DTM-resolution-
    limited 'shelf' pattern that doesn't match the fine real-photo detail —
    a resolution mismatch, not something the entropy filter alone catches
    (both real examples passed entropy). Skip poses beyond MAX_ABS_PITCH_DEG
    before spending a render/network call on a pose we know can't work."""
    dtm_record = _make_record("DTM_A", -5.0, -4.0, 137.0, 138.0)
    poses = [
        _make_pose("LEVEL", lat=-4.5, lon=137.5, pitch_deg=-10.0),   # kept
        _make_pose("STEEP", lat=-4.5, lon=137.5, pitch_deg=-40.0),  # skipped
    ]

    fake_fetch_result = {
        "product_id": "DTM_A", "status": "ok",
        "dtm_path": str(tmp_path / "DTM_A.IMG"),
        "ortho_paths": {"obs": str(tmp_path / "obs_ORTHO.JP2")},
    }
    _write_fake_dtm_raster(tmp_path / "DTM_A.IMG")
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")

    rendered_ids = []

    def fake_render_ground_view(*args, **kwargs):
        # Real texture (not a uniform array), so this clears the entropy
        # filter and isolates the pitch-skip decision being tested here.
        return np.random.default_rng(0).integers(0, 255, size=(256, 256), dtype=np.uint8)

    def fake_fetch_target_photo(product_id, sol, dest_path):
        rendered_ids.append(product_id)
        dest_path.write_bytes(b"\xff\xd8\xff fake jpg")
        return True

    monkeypatch.setattr("assemble_paired_corpus.fetch_dtm_and_orthos",
                        lambda record, scratch, dest: fake_fetch_result)
    monkeypatch.setattr("assemble_paired_corpus.load_dtm_arrays",
                        lambda dtm_path, ortho_path: _make_varied_arrays())
    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: (25.0, 25.0))
    monkeypatch.setattr("assemble_paired_corpus.compass_heading_to_render_heading",
                        lambda compass_deg, transform: 0.0)
    monkeypatch.setattr("assemble_paired_corpus.render_ground_view",
                        fake_render_ground_view)
    monkeypatch.setattr("assemble_paired_corpus.fetch_target_photo",
                        fake_fetch_target_photo)

    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    result = process_dtm_group(dtm_record, poses, scratch_dir, out_dir)

    assert result["saved_ids"] == ["LEVEL"]
    assert rendered_ids == ["LEVEL"]  # STEEP never even reached the render/fetch step


def test_process_dtm_group_passes_pose_pitch_to_render(tmp_path, monkeypatch):
    """Regression test for the 2026-08-09 content-mismatch fix: most real
    Navcam poses have a steep non-zero pitch (down-look workspace shots),
    and the renderer only produces a geometrically correct condition map
    if that real pitch reaches render_ground_view — silently dropping it
    (e.g. a stale positional-arg refactor) would reintroduce the bug even
    though every other test here uses pitch_deg=0.0 and wouldn't catch it."""
    dtm_record = _make_record("DTM_A", -5.0, -4.0, 137.0, 138.0)
    # Within MAX_ABS_PITCH_DEG so this pose reaches the render step at all --
    # the steep-pitch-skip behavior itself is covered by
    # test_process_dtm_group_skips_steep_pitch_before_rendering above.
    poses = [_make_pose("P1", lat=-4.5, lon=137.5, pitch_deg=-15.0)]

    fake_fetch_result = {
        "product_id": "DTM_A", "status": "ok",
        "dtm_path": str(tmp_path / "DTM_A.IMG"),
        "ortho_paths": {"obs": str(tmp_path / "obs_ORTHO.JP2")},
    }
    _write_fake_dtm_raster(tmp_path / "DTM_A.IMG")
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")

    received_pitch = []

    def fake_render_ground_view(*args, **kwargs):
        received_pitch.append(kwargs.get("pitch_deg"))
        return np.full((256, 256), 100, dtype=np.uint8)

    monkeypatch.setattr("assemble_paired_corpus.fetch_dtm_and_orthos",
                        lambda record, scratch, dest: fake_fetch_result)
    monkeypatch.setattr("assemble_paired_corpus.load_dtm_arrays",
                        lambda dtm_path, ortho_path: _make_varied_arrays())
    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: (25.0, 25.0))
    monkeypatch.setattr("assemble_paired_corpus.compass_heading_to_render_heading",
                        lambda compass_deg, transform: 0.0)
    monkeypatch.setattr("assemble_paired_corpus.render_ground_view",
                        fake_render_ground_view)
    monkeypatch.setattr("assemble_paired_corpus.fetch_target_photo",
                        lambda product_id, sol, dest_path: dest_path.write_bytes(b"\xff\xd8\xff fake jpg") or True)

    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    process_dtm_group(dtm_record, poses, scratch_dir, out_dir)

    assert received_pitch == [-15.0]


def test_process_dtm_group_skips_low_entropy_renders(tmp_path, monkeypatch):
    """A rendered condition map below ENTROPY_TH — whether genuine sky or
    real terrain sitting in a no-ortho-coverage gap that also paints as
    sky_value=0 (the real corpus's actual 80%-blank problem) — must be
    skipped: not saved, and its product_id absent from saved_ids. Mocks
    render_ground_view directly to a known-uniform array so this isolates
    the entropy-filter decision itself from the rendering geometry."""
    dtm_record = _make_record("DTM_A", -5.0, -4.0, 137.0, 138.0)
    poses = [_make_pose("P1", lat=-4.5, lon=137.5)]

    fake_fetch_result = {
        "product_id": "DTM_A", "status": "ok",
        "dtm_path": str(tmp_path / "DTM_A.IMG"),
        "ortho_paths": {"obs": str(tmp_path / "obs_ORTHO.JP2")},
    }
    _write_fake_dtm_raster(tmp_path / "DTM_A.IMG")
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")

    photo_fetch_calls = []

    def fake_fetch_target_photo(product_id, sol, dest_path):
        photo_fetch_calls.append(product_id)
        dest_path.write_bytes(b"\xff\xd8\xff fake jpg")
        return True

    monkeypatch.setattr("assemble_paired_corpus.fetch_dtm_and_orthos",
                        lambda record, scratch, dest: fake_fetch_result)
    monkeypatch.setattr("assemble_paired_corpus.load_dtm_arrays",
                        lambda dtm_path, ortho_path: _make_varied_arrays())
    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: (25.0, 25.0))
    monkeypatch.setattr("assemble_paired_corpus.compass_heading_to_render_heading",
                        lambda compass_deg, transform: 0.0)
    monkeypatch.setattr("assemble_paired_corpus.render_ground_view",
                        lambda *args, **kwargs: np.zeros((256, 256), dtype=np.uint8))
    monkeypatch.setattr("assemble_paired_corpus.fetch_target_photo",
                        fake_fetch_target_photo)

    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    result = process_dtm_group(dtm_record, poses, scratch_dir, out_dir)

    assert result["status"] == "ok"
    assert result["pairs_saved"] == 0
    assert result["saved_ids"] == []
    assert not (out_dir / "P1_condition.png").exists()
    assert not (out_dir / "P1_target.jpg").exists()
    # Skipped before the network call for the target photo, not after.
    assert photo_fetch_calls == []


def test_process_dtm_group_cleans_up_raw_files_on_early_failure(
        tmp_path, monkeypatch):
    """fetch_dtm_and_orthos can fail after the DTM .IMG already downloaded
    successfully (e.g. an ortho download fails) — its failure return dict
    carries no dtm_path/ortho_paths keys in that case, so process_dtm_group
    must fall back to the deterministic scratch_dir naming convention to
    find and delete whatever raw files actually landed on disk."""
    dtm_record = _make_record("DTM_A", -5.0, -4.0, 137.0, 138.0,
                              obs_id_a="OBS_A", obs_id_b="OBS_B")
    poses = [_make_pose("P1", lat=-4.5, lon=137.5)]

    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    # Simulate what the real fetch_dtm_and_orthos leaves behind when the
    # DTM and the first ortho downloaded fine but the second ortho failed.
    (scratch_dir / "DTM_A.IMG").write_bytes(b"partial dtm bytes")
    (scratch_dir / "OBS_A_ORTHO.JP2").write_bytes(b"partial ortho bytes")

    monkeypatch.setattr(
        "assemble_paired_corpus.fetch_dtm_and_orthos",
        lambda record, scratch, dest: {
            "product_id": "DTM_A", "status": "ortho_download_failed_for_OBS_B",
        },
    )

    out_dir = tmp_path / "out"
    result = process_dtm_group(dtm_record, poses, scratch_dir, out_dir)

    assert result["status"] == "ortho_download_failed_for_OBS_B"
    assert result["pairs_saved"] == 0
    assert result["saved_ids"] == []
    assert not (scratch_dir / "DTM_A.IMG").exists()
    assert not (scratch_dir / "OBS_A_ORTHO.JP2").exists()


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


def test_process_dtm_group_respects_custom_max_pitch_deg(tmp_path, monkeypatch):
    """A caller-supplied max_pitch_deg overrides the module default --
    needed so the pitch-bucket investigation tool (Task 3) can probe wider
    thresholds than the current MAX_ABS_PITCH_DEG=20.0 without editing the
    module constant."""
    dtm_record = _make_record("DTM_A", -5.0, -4.0, 137.0, 138.0)
    poses = [_make_pose("STEEP", lat=-4.5, lon=137.5, pitch_deg=-40.0)]

    fake_fetch_result = {
        "product_id": "DTM_A", "status": "ok",
        "dtm_path": str(tmp_path / "DTM_A.IMG"),
        "ortho_paths": {"obs": str(tmp_path / "obs_ORTHO.JP2")},
    }
    _write_fake_dtm_raster(tmp_path / "DTM_A.IMG")
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")

    monkeypatch.setattr("assemble_paired_corpus.fetch_dtm_and_orthos",
                        lambda record, scratch, dest: fake_fetch_result)
    monkeypatch.setattr("assemble_paired_corpus.load_dtm_arrays",
                        lambda dtm_path, ortho_path: _make_varied_arrays())
    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: (25.0, 25.0))
    monkeypatch.setattr("assemble_paired_corpus.compass_heading_to_render_heading",
                        lambda compass_deg, transform: 0.0)
    monkeypatch.setattr("assemble_paired_corpus.fetch_target_photo",
                        lambda product_id, sol, dest_path: dest_path.write_bytes(b"\xff\xd8\xff fake jpg") or True)

    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    # Default MAX_ABS_PITCH_DEG (20.0) would skip this -40deg pose; an
    # explicit wider max_pitch_deg should keep it.
    result = process_dtm_group(dtm_record, poses, scratch_dir, out_dir,
                               max_pitch_deg=45.0)

    assert result["saved_ids"] == ["STEEP"]


def test_process_dtm_group_tags_manifest_rows_with_source_mission(
        tmp_path, monkeypatch):
    """source_mission defaults to 'MSL' (every pair produced by this plan
    is Gale/Curiosity) and is threaded into the returned dict for the
    manifest writer (Task 6) to pick up."""
    dtm_record = _make_record("DTM_A", -5.0, -4.0, 137.0, 138.0)
    poses = [_make_pose("P1", lat=-4.5, lon=137.5)]

    fake_fetch_result = {
        "product_id": "DTM_A", "status": "ok",
        "dtm_path": str(tmp_path / "DTM_A.IMG"),
        "ortho_paths": {"obs": str(tmp_path / "obs_ORTHO.JP2")},
    }
    _write_fake_dtm_raster(tmp_path / "DTM_A.IMG")
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")

    monkeypatch.setattr("assemble_paired_corpus.fetch_dtm_and_orthos",
                        lambda record, scratch, dest: fake_fetch_result)
    monkeypatch.setattr("assemble_paired_corpus.load_dtm_arrays",
                        lambda dtm_path, ortho_path: _make_varied_arrays())
    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: (25.0, 25.0))
    monkeypatch.setattr("assemble_paired_corpus.compass_heading_to_render_heading",
                        lambda compass_deg, transform: 0.0)
    monkeypatch.setattr("assemble_paired_corpus.fetch_target_photo",
                        lambda product_id, sol, dest_path: dest_path.write_bytes(b"\xff\xd8\xff fake jpg") or True)

    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    result = process_dtm_group(dtm_record, poses, scratch_dir, out_dir)
    assert result["source_mission"] == "MSL"

    # Recreate the fake DTM since process_dtm_group deletes it in its finally block.
    _write_fake_dtm_raster(tmp_path / "DTM_A.IMG")
    (tmp_path / "obs_ORTHO.JP2").write_bytes(b"fake")
    result2 = process_dtm_group(dtm_record, poses, scratch_dir, out_dir,
                                source_mission="M20")
    assert result2["source_mission"] == "M20"


def test_gather_candidate_poses_dedupes_across_sols(monkeypatch):
    """Pulled out of main() so render_pitch_bucket_samples.py (Task 3) can
    reuse the exact same pose-gathering logic instead of duplicating it."""
    def fake_list_products(sol):
        return [f"P{sol}_1", f"P{sol}_2", f"P{sol}_3"]

    def fake_fetch_and_parse(product_id, sol, localization):
        return _make_pose(product_id, lat=-4.5, lon=137.5)

    monkeypatch.setattr("assemble_paired_corpus.list_navcam_products_for_sol",
                        fake_list_products)
    monkeypatch.setattr("assemble_paired_corpus.fetch_and_parse_pose",
                        fake_fetch_and_parse)

    poses = gather_candidate_poses(sols=[10, 20], per_sol=2, localization={})

    assert len(poses) == 4  # 2 sols x per_sol=2, capped correctly
    assert {p.product_id for p in poses} == {"P10_1", "P10_2", "P20_1", "P20_2"}


def test_gather_candidate_poses_skips_sol_on_listing_error(monkeypatch):
    def fake_list_products(sol):
        if sol == 20:
            raise Exception("network error")
        return [f"P{sol}_1"]

    monkeypatch.setattr("assemble_paired_corpus.list_navcam_products_for_sol",
                        fake_list_products)
    monkeypatch.setattr("assemble_paired_corpus.fetch_and_parse_pose",
                        lambda product_id, sol, localization: _make_pose(product_id, -4.5, 137.5))

    poses = gather_candidate_poses(sols=[10, 20], per_sol=1, localization={})

    assert {p.product_id for p in poses} == {"P10_1"}


def test_render_pose_condition_map_returns_none_when_pixel_out_of_bounds(
        tmp_path, monkeypatch):
    dtm_path = tmp_path / "DTM_A.IMG"
    _write_fake_dtm_raster(dtm_path)
    pose = _make_pose("P1", lat=-4.5, lon=137.5)

    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: None)

    result = render_pose_condition_map(dtm_path, _make_varied_arrays(),
                                       transform=None, pose=pose)
    assert result is None


def test_render_pose_condition_map_renders_when_in_bounds(tmp_path, monkeypatch):
    dtm_path = tmp_path / "DTM_A.IMG"
    _write_fake_dtm_raster(dtm_path)
    pose = _make_pose("P1", lat=-4.5, lon=137.5, pitch_deg=-10.0)

    monkeypatch.setattr("assemble_paired_corpus.latlon_to_dtm_pixel",
                        lambda dtm_path, lat, lon: (25.0, 25.0))
    monkeypatch.setattr("assemble_paired_corpus.compass_heading_to_render_heading",
                        lambda compass_deg, transform: 0.0)

    result = render_pose_condition_map(dtm_path, _make_varied_arrays(),
                                       transform=None, pose=pose)
    assert result is not None
    assert result.shape == (256, 256)


def test_query_region_dtm_coverage_converts_longitude_and_queries(monkeypatch):
    """Uses REGIONS[region_key]'s signed lon bounds, converts to the ODE
    API's 0-360 convention (query_dtm_coverage's own requirement), and
    returns whatever it returns -- this function is a thin, testable
    wrapper so main() can be driven by --region instead of a hardcoded key."""
    captured = {}

    def fake_query(lat_min, lat_max, west, east):
        captured["args"] = (lat_min, lat_max, west, east)
        return ["fake_record"]

    monkeypatch.setattr("assemble_paired_corpus.query_dtm_coverage", fake_query)

    result = query_region_dtm_coverage("gale_crater")

    assert result == ["fake_record"]
    lat_min, lat_max, west, east = captured["args"]
    assert (lat_min, lat_max) == (-8.0, -3.0)
    assert west == pytest.approx(135.0)  # already 0-360, no wraparound needed
    assert east == pytest.approx(140.0)


def test_region_cli_flag_excludes_oxia_planum():
    """Oxia Planum is explicitly out of scope — no real rover target photos
    exist there. This pipeline requires paired condition maps + real photos,
    so oxia_planum must never be a valid --region choice even though it
    exists in the REGIONS dict (which is used elsewhere). Regression test
    to prevent accidental inclusion of oxia_planum in region iterations.

    Exercises main()'s real parser via build_arg_parser() (not a re-typed
    inline copy of its choices=[...]) so this test actually fails if
    main()'s --region argument is ever reverted to
    choices=list(REGIONS.keys()) -- an earlier version of this test built
    its own ArgumentParser and provided zero protection against exactly
    that regression."""
    parser = build_arg_parser()

    # Valid regions should parse successfully
    args_gale = parser.parse_args(["--region", "gale_crater"])
    assert args_gale.region == "gale_crater"

    args_jezero = parser.parse_args(["--region", "jezero_crater"])
    assert args_jezero.region == "jezero_crater"

    # oxia_planum must be rejected
    with pytest.raises(SystemExit):
        parser.parse_args(["--region", "oxia_planum"])


def test_build_arg_parser_default_region_is_gale_crater():
    """Sanity check on build_arg_parser()'s defaults, since main() now
    depends on it entirely for argument definitions."""
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.region == "gale_crater"


def test_write_manifest_includes_source_mission_column(tmp_path):
    """Spec's Output format section requires a source_mission column on
    the manifest (existing Gale rows tagged MSL). process_dtm_group
    already threads source_mission through its return dict (Task 1) --
    this closes the gap where main()'s manifest writer used to hardcode
    a 3-column header and silently drop the key."""
    manifest = [
        {"product_id": "DTM_A", "status": "ok", "pairs_saved": 3,
         "saved_ids": ["P1", "P2", "P3"], "source_mission": "MSL"},
        {"product_id": "DTM_B", "status": "ok", "pairs_saved": 0,
         "saved_ids": [], "source_mission": "M20"},
    ]
    manifest_path = tmp_path / "manifest.csv"

    written = write_manifest(manifest, saved_ids=["P1", "P2", "P3"],
                             manifest_path=manifest_path)

    assert written is True
    rows = list(csv.reader(manifest_path.open()))
    assert rows[0] == ["dtm_product_id", "status", "pairs_saved", "source_mission"]
    assert rows[1] == ["DTM_A", "ok", "3", "MSL"]
    assert rows[2] == ["DTM_B", "ok", "0", "M20"]


def test_write_manifest_defaults_missing_source_mission_to_msl(tmp_path):
    """A manifest row dict missing the source_mission key entirely (e.g.
    hand-constructed, or from an older code path) still writes a valid
    row rather than raising a KeyError."""
    manifest = [{"product_id": "DTM_A", "status": "ok", "pairs_saved": 1}]
    manifest_path = tmp_path / "manifest.csv"

    write_manifest(manifest, saved_ids=["P1"], manifest_path=manifest_path)

    rows = list(csv.reader(manifest_path.open()))
    assert rows[1] == ["DTM_A", "ok", "1", "MSL"]


def test_write_manifest_skips_when_zero_pairs_saved(tmp_path, capsys):
    """--region jezero_crater currently has no working pose-fetch path and
    a bounding box disjoint from the MSL-only pose gatherer's Gale poses --
    running with it produces 0 DTM groups, 0 pairs, and previously silently
    overwrote the git-tracked manifest.csv with a bare-header file. Guard:
    when saved_ids is empty, warn and skip the write entirely rather than
    clobbering a non-empty tracked manifest with an empty one."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("dtm_product_id,status,pairs_saved,source_mission\n"
                             "DTM_OLD,ok,5,MSL\n")

    written = write_manifest(
        [{"product_id": "DTM_A", "status": "ok", "pairs_saved": 0,
          "source_mission": "MSL"}],
        saved_ids=[], manifest_path=manifest_path,
    )

    assert written is False
    # Pre-existing non-empty manifest is untouched, not overwritten.
    assert "DTM_OLD" in manifest_path.read_text()
    assert "WARNING" in capsys.readouterr().out


def test_split_pairs_and_move_clears_stale_files_from_a_previous_run(tmp_path):
    """A re-run against a different total n (e.g. after --region or widened
    sampling changes candidate counts) must not leave a previous run's
    files mixed into train/val/test — split_pairs_and_move used to only
    ever mkdir+move (add), never clear, so a stale product ID from a prior
    run could sit in a different split than this run's shuffle would put
    it, or simply linger as an orphan alongside the new pairs. This is the
    exact hazard Task 8's implementer hit and had to manually work around."""
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "out"

    # Simulate a stale previous run's leftovers already sitting in train/.
    stale_dir = out_dir / "train"
    stale_dir.mkdir(parents=True)
    (stale_dir / "STALE_condition.png").write_bytes(b"old")
    (stale_dir / "STALE_target.jpg").write_bytes(b"old")

    for pid in ["P1", "P2", "P3", "P4"]:
        (staging / f"{pid}_condition.png").write_bytes(b"c")
        (staging / f"{pid}_target.jpg").write_bytes(b"t")

    split_pairs_and_move(["P1", "P2", "P3", "P4"], staging, out_dir, seed=0)

    all_condition_files = {p.name for split in ["train", "val", "test"]
                           for p in (out_dir / split).glob("*_condition.png")}
    assert "STALE_condition.png" not in all_condition_files
    assert len(all_condition_files) == 4
