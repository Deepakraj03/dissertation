import math

import numpy as np
import pytest

from render_ground_view import render_ground_view, bilinear_sample


def test_bilinear_sample_exact_pixel():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert bilinear_sample(arr, 0, 0) == pytest.approx(1.0)
    assert bilinear_sample(arr, 1, 1) == pytest.approx(4.0)


def test_bilinear_sample_interpolates_midpoint():
    arr = np.array([[0.0, 10.0], [0.0, 10.0]])
    # Halfway between column 0 (value 0) and column 1 (value 10)
    assert bilinear_sample(arr, 0, 0.5) == pytest.approx(5.0)


def test_bilinear_sample_out_of_bounds_is_nan():
    arr = np.zeros((5, 5))
    assert np.isnan(bilinear_sample(arr, -1, 2))
    assert np.isnan(bilinear_sample(arr, 2, 10))


def test_render_ground_view_output_shape_and_dtype():
    heightmap = np.zeros((200, 200), dtype=np.float32)
    albedo = np.full((200, 200), 128, dtype=np.uint8)
    img = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                             camera_row=100, camera_col=100, heading_deg=0,
                             output_size=64)
    assert img.shape == (64, 64)
    assert img.dtype == np.uint8


def test_render_ground_view_flat_terrain_horizon_is_consistent_row():
    # On perfectly flat terrain, every column's horizon (the farthest
    # visible point before the ray exits the heightmap) should land on
    # the same screen row, since camera height and pitch are constant
    # across columns — this is a basic sanity check that the projection
    # math doesn't have a per-column bug.
    heightmap = np.zeros((400, 400), dtype=np.float32)
    albedo = np.full((400, 400), 100, dtype=np.uint8)
    img = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                             camera_row=200, camera_col=200, heading_deg=0,
                             output_size=64, sky_value=0)
    # Every column should have the same count of "ground" (non-sky) rows,
    # since flat terrain at constant camera height/pitch looks identical
    # in every direction.
    ground_counts = (img != 0).sum(axis=0)
    assert ground_counts.max() - ground_counts.min() <= 1  # allow 1px rounding


def test_render_ground_view_foreground_bump_occludes_background():
    # A tall ridge a few meters directly ahead of the camera must hide
    # a second, taller-in-isolation feature placed farther behind it.
    # This is the test that actually exercises the occlusion logic —
    # without it, a renderer that ignores depth entirely (e.g. just
    # paints raw elevation as brightness) would still pass the shape
    # and flat-terrain tests above.
    size = 400
    heightmap = np.zeros((size, size), dtype=np.float32)
    albedo = np.zeros((size, size), dtype=np.uint8)

    camera_row, camera_col = 200, 200
    # Near ridge: 3m tall, 5m ahead (heading 0 = +row direction)
    heightmap[camera_row + 5, camera_col - 2:camera_col + 3] = 3.0
    albedo[camera_row + 5, camera_col - 2:camera_col + 3] = 255  # bright
    # Far feature: 5m tall (would be very prominent if visible), 20m ahead,
    # directly behind the near ridge from the camera's point of view.
    heightmap[camera_row + 20, camera_col - 2:camera_col + 3] = 5.0
    albedo[camera_row + 20, camera_col - 2:camera_col + 3] = 200

    img = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                             camera_row=camera_row, camera_col=camera_col,
                             heading_deg=0, output_size=64, sky_value=0)

    # The center column (looking straight at both features) should show
    # the near ridge's brightness (255), not the far feature's (200) —
    # painter's-algorithm-from-near-to-far with correct occlusion means
    # once the near ridge is drawn, the far feature's rays project to
    # screen rows already claimed and must not overwrite them.
    center_col = 32
    assert 255 in img[:, center_col]
    # The far feature's distinct brightness (200) must not appear at all
    # in this column — it is fully hidden behind the near ridge.
    assert 200 not in img[:, center_col]


def test_render_ground_view_depression_does_not_occlude_anything():
    # A pit (negative height) directly ahead must not hide terrain behind
    # it — it should just look like a dip, never block the view. Guards
    # against an inverted occlusion comparison (a plausible off-by-sign bug
    # given the algorithm's "less than" comparison).
    size = 400
    heightmap = np.zeros((size, size), dtype=np.float32)
    albedo = np.zeros((size, size), dtype=np.uint8)
    camera_row, camera_col = 200, 200
    heightmap[camera_row + 5, camera_col - 2:camera_col + 3] = -3.0
    heightmap[camera_row + 20, camera_col - 2:camera_col + 3] = 1.0
    albedo[camera_row + 20, camera_col - 2:camera_col + 3] = 200

    img = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                             camera_row=camera_row, camera_col=camera_col,
                             heading_deg=0, output_size=64, sky_value=0)
    center_col = 32
    assert 200 in img[:, center_col]


def test_render_ground_view_exits_bounds_shows_sky():
    # Camera near the edge of the heightmap, looking off the edge — rays
    # that leave the array bounds before hitting max_range must fall back
    # to sky_value, not crash or wrap around.
    heightmap = np.zeros((50, 50), dtype=np.float32)
    albedo = np.full((50, 50), 100, dtype=np.uint8)
    img = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                             camera_row=5, camera_col=25, heading_deg=180,
                             output_size=32, sky_value=77, max_range_m=30.0)
    assert (img == 77).any()


def test_render_ground_view_default_pitch_matches_prior_level_behavior():
    # pitch_deg defaults to 0.0 (level camera) so every pre-existing caller
    # (which never passed pitch_deg) keeps rendering exactly as before.
    heightmap = np.zeros((400, 400), dtype=np.float32)
    albedo = np.full((400, 400), 100, dtype=np.uint8)
    img_default = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                                     camera_row=200, camera_col=200,
                                     heading_deg=0, output_size=64, sky_value=0)
    img_explicit_level = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                                            camera_row=200, camera_col=200,
                                            heading_deg=0, output_size=64,
                                            sky_value=0, pitch_deg=0.0)
    assert np.array_equal(img_default, img_explicit_level)


def test_render_ground_view_pitch_down_increases_visible_ground():
    # Real Navcam data is mostly steep down-look (arm-workspace) shots, not
    # horizon shots -- see 2026-08-09 finding. A camera pitched down should
    # reveal more ground (fewer sky pixels) than a level camera on the same
    # flat terrain, since the boresight itself is now aimed at the ground.
    heightmap = np.zeros((400, 400), dtype=np.float32)
    albedo = np.full((400, 400), 100, dtype=np.uint8)
    img_level = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                                   camera_row=200, camera_col=200,
                                   heading_deg=0, output_size=64, sky_value=0,
                                   pitch_deg=0.0)
    img_pitched_down = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                                          camera_row=200, camera_col=200,
                                          heading_deg=0, output_size=64,
                                          sky_value=0, pitch_deg=-30.0)
    assert (img_pitched_down != 0).sum() > (img_level != 0).sum()


def test_render_ground_view_pitch_shifts_screen_row_by_expected_amount():
    # A feature placed exactly at eye height (elevation_angle == 0 relative
    # to a level boresight) must land at the exact screen row the pitch
    # formula predicts, not just "somewhere different" -- precise check
    # that pitch is a boresight offset (elevation_angle - pitch_rad), not
    # applied with the wrong sign or magnitude.
    size = 400
    heightmap = np.zeros((size, size), dtype=np.float32)
    albedo = np.zeros((size, size), dtype=np.uint8)
    camera_row, camera_col = 200, 200
    camera_height_m = 1.2
    dist_m = 10.0  # lands exactly on a 0.5m ray-march step
    feature_row = camera_row + int(dist_m)
    heightmap[feature_row, camera_col] = camera_height_m  # == eye height
    albedo[feature_row, camera_col] = 255

    fov_deg = 45.0
    output_size = 64
    center_row = output_size / 2.0
    center_col = output_size // 2

    img_level = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                                   camera_row=camera_row, camera_col=camera_col,
                                   heading_deg=0, output_size=output_size,
                                   fov_deg=fov_deg, camera_height_m=camera_height_m,
                                   sky_value=0, pitch_deg=0.0)
    # elevation_angle == 0 pre-pitch -> screen_row == center_row exactly.
    assert np.where(img_level[:, center_col] == 255)[0][0] == pytest.approx(
        center_row, abs=1)

    pitch_deg = -11.25  # half of half-FOV (22.5 deg)
    img_pitched = render_ground_view(heightmap, albedo, pixel_scale_m=1.0,
                                     camera_row=camera_row, camera_col=camera_col,
                                     heading_deg=0, output_size=output_size,
                                     fov_deg=fov_deg, camera_height_m=camera_height_m,
                                     sky_value=0, pitch_deg=pitch_deg)
    # elevation_angle_adjusted = 0 - radians(-11.25) = +11.25 deg, which is
    # half the half-FOV -> screen_row = center_row - 0.5*center_row.
    expected_row = center_row - 0.5 * center_row
    assert np.where(img_pitched[:, center_col] == 255)[0][0] == pytest.approx(
        expected_row, abs=1)
