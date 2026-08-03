"""Deterministic height-field-to-ground-view renderer.

Projects a HiRISE-derived DTM (height field) into a simulated ground-level
camera view, using a height-field ray-marching technique (no mesh, no GPU,
no OpenGL/EGL — the A4000 target machine has no admin rights to install
those). This is the geometry stage of the geometry-mediated translation
design: it guarantees correct large-scale shape and viewpoint by
construction, rather than asking an adversarial network to infer both from
a raw nadir crop.

Camera parameters (1.2m height, 45-degree FOV, ~30m range) are grounded in
real Curiosity NAVCAM specifications, verified 2026-08-03.
"""

import math

import numpy as np


def bilinear_sample(arr: np.ndarray, row: float, col: float) -> float:
    """Bilinearly interpolate arr at fractional (row, col). Returns nan if
    the sample point (including its interpolation neighbors) falls outside
    arr's bounds."""
    h, w = arr.shape
    if row < 0 or col < 0 or row > h - 1 or col > w - 1:
        return float("nan")
    r0, c0 = int(math.floor(row)), int(math.floor(col))
    r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)
    fr, fc = row - r0, col - c0
    top = arr[r0, c0] * (1 - fc) + arr[r0, c1] * fc
    bot = arr[r1, c0] * (1 - fc) + arr[r1, c1] * fc
    return float(top * (1 - fr) + bot * fr)


def render_ground_view(heightmap: np.ndarray, albedo: np.ndarray,
                       pixel_scale_m: float, camera_row: float,
                       camera_col: float, heading_deg: float,
                       output_size: int = 256, fov_deg: float = 45.0,
                       camera_height_m: float = 1.2,
                       max_range_m: float = 30.0,
                       sky_value: int = 0) -> np.ndarray:
    """Render a ground-level view of heightmap/albedo from a camera at
    (camera_row, camera_col) in heightmap pixel coordinates, facing
    heading_deg (0 = +row direction, 90 = +col direction, degrees
    clockwise). Returns a (output_size, output_size) uint8 grayscale image.
    """
    ground_h = bilinear_sample(heightmap, camera_row, camera_col)
    if math.isnan(ground_h):
        raise ValueError("camera position is outside the heightmap")
    eye_h = ground_h + camera_height_m

    img = np.full((output_size, output_size), sky_value, dtype=np.uint8)
    vfov_rad = math.radians(fov_deg)  # square FOV: horizontal == vertical
    center_row = output_size / 2.0

    step_m = max(pixel_scale_m * 0.5, 0.1)
    n_steps = max(int(max_range_m / step_m), 1)

    for col in range(output_size):
        az_offset_deg = (col / output_size - 0.5) * fov_deg
        az_rad = math.radians(heading_deg + az_offset_deg)
        dr = math.cos(az_rad)  # direction in heightmap row axis
        dc = math.sin(az_rad)  # direction in heightmap col axis

        nearest_drawn_row = output_size  # nothing drawn yet this column
        for step in range(1, n_steps + 1):
            dist_m = step * step_m
            world_row = camera_row + (dr * dist_m / pixel_scale_m)
            world_col = camera_col + (dc * dist_m / pixel_scale_m)

            terrain_h = bilinear_sample(heightmap, world_row, world_col)
            if math.isnan(terrain_h):
                break  # ray left the heightmap; remainder stays sky_value

            elevation_angle = math.atan2(terrain_h - eye_h, dist_m)
            screen_row = center_row - (elevation_angle / (vfov_rad / 2.0)) * center_row
            screen_row = int(round(screen_row))
            screen_row = max(0, min(output_size - 1, screen_row))

            if screen_row < nearest_drawn_row:
                pixel_val = bilinear_sample(albedo, world_row, world_col)
                if not math.isnan(pixel_val):
                    img[screen_row:nearest_drawn_row, col] = np.uint8(pixel_val)
                nearest_drawn_row = screen_row
                if nearest_drawn_row <= 0:
                    break  # column fully filled top-to-bottom

    return img
