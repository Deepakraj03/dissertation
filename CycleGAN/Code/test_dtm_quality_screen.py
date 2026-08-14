import numpy as np
import pytest

from dtm_quality_screen import compute_shaded_relief, patch_alignment_score


def test_compute_shaded_relief_flat_terrain_is_uniform():
    heightmap = np.full((20, 20), 3.0, dtype=np.float32)
    shaded = compute_shaded_relief(heightmap, pixel_scale_m=1.0)
    assert shaded.dtype == np.uint8
    assert shaded.std() == pytest.approx(0.0, abs=1e-6)


def test_compute_shaded_relief_varying_slope_is_not_uniform():
    # A linear ramp has *constant* gradient everywhere, so it correctly
    # produces uniform shading (a uniformly tilted plane reflects light
    # uniformly) -- not a useful case for this test. A bump has gradient
    # that varies by location and direction, which must show up as varying
    # shading.
    y, x = np.mgrid[0:20, 0:20].astype(np.float32)
    heightmap = 10.0 * np.exp(-((x - 10) ** 2 + (y - 10) ** 2) / 20.0)
    shaded = compute_shaded_relief(heightmap, pixel_scale_m=1.0)
    assert shaded.std() > 0.0


def test_compute_shaded_relief_handles_nan_without_crashing():
    heightmap = np.full((20, 20), 3.0, dtype=np.float32)
    heightmap[5:10, 5:10] = np.nan
    shaded = compute_shaded_relief(heightmap, pixel_scale_m=1.0)
    assert not np.isnan(shaded).any()


def test_patch_alignment_score_perfect_match_scores_near_one():
    heightmap = np.zeros((32, 32), dtype=np.float32)
    heightmap[:, 16:] = 10.0
    shaded = compute_shaded_relief(heightmap, pixel_scale_m=1.0)

    score = patch_alignment_score(heightmap, shaded, pixel_scale_m=1.0)

    assert score == pytest.approx(1.0, abs=1e-6)


def test_patch_alignment_score_unrelated_ortho_scores_low():
    heightmap = np.zeros((32, 32), dtype=np.float32)
    heightmap[:, 16:] = 10.0  # a hard vertical step

    rng = np.random.default_rng(0)
    unrelated_ortho = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)

    score = patch_alignment_score(heightmap, unrelated_ortho, pixel_scale_m=1.0)

    assert score < 0.4


def test_patch_alignment_score_all_nan_heightmap_returns_zero():
    heightmap = np.full((16, 16), np.nan, dtype=np.float32)
    ortho = np.full((16, 16), 128, dtype=np.uint8)
    assert patch_alignment_score(heightmap, ortho, pixel_scale_m=1.0) == 0.0
