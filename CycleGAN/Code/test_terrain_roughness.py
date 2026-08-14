import numpy as np
import pytest

from terrain_roughness import compute_terrain_roughness


def test_flat_heightmap_has_zero_roughness():
    heightmap = np.full((10, 10), 5.0, dtype=np.float32)
    assert compute_terrain_roughness(heightmap, pixel_scale_m=1.0) == pytest.approx(0.0)


def test_linear_ramp_has_known_closed_form_slope():
    # Each column step rises by 2.0m over a 1.0m pixel -- every column-wise
    # diff is exactly 2.0, every row-wise diff is exactly 0.0 (constant
    # across rows). Mean over both directions combined is not simply 2.0,
    # so assert against the real weighted average instead of assuming it.
    col = np.arange(10, dtype=np.float32) * 2.0
    heightmap = np.tile(col, (10, 1))  # 10 rows, each identical to col
    roughness = compute_terrain_roughness(heightmap, pixel_scale_m=1.0)
    # 9 column-diffs per row * 10 rows, all == 2.0; 10 row-diffs per column
    # * 9 columns, all == 0.0.
    n_col_diffs = 9 * 10
    n_row_diffs = 10 * 9
    expected = (n_col_diffs * 2.0 + n_row_diffs * 0.0) / (n_col_diffs + n_row_diffs)
    assert roughness == pytest.approx(expected)


def test_pixel_scale_normalizes_the_same_raw_relief():
    heightmap = np.zeros((5, 5), dtype=np.float32)
    heightmap[:, 1:] = 10.0  # a single 10m step

    roughness_1m = compute_terrain_roughness(heightmap, pixel_scale_m=1.0)
    roughness_10m = compute_terrain_roughness(heightmap, pixel_scale_m=10.0)

    assert roughness_10m == pytest.approx(roughness_1m / 10.0)


def test_nan_cells_excluded_not_propagated():
    heightmap = np.full((10, 10), 5.0, dtype=np.float32)
    heightmap[3, 3] = np.nan
    heightmap[7, 8] = 100.0  # real relief elsewhere, must still register

    roughness = compute_terrain_roughness(heightmap, pixel_scale_m=1.0)

    assert not np.isnan(roughness)
    assert roughness > 0.0


def test_all_nan_heightmap_returns_zero_not_nan_or_crash():
    heightmap = np.full((5, 5), np.nan, dtype=np.float32)
    assert compute_terrain_roughness(heightmap, pixel_scale_m=1.0) == 0.0


def test_rougher_terrain_scores_higher_than_smoother_terrain():
    rng = np.random.default_rng(0)
    smooth = rng.uniform(0, 1, size=(50, 50)).astype(np.float32)
    rocky = rng.uniform(0, 20, size=(50, 50)).astype(np.float32)

    assert (compute_terrain_roughness(rocky, pixel_scale_m=1.0)
           > compute_terrain_roughness(smooth, pixel_scale_m=1.0))
