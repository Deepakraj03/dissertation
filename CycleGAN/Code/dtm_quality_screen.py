"""Screen (heightmap, orthoimage) patches for genuine correspondence before
training a single-image DTM estimator on them -- a misaligned or degenerate
DTM/ortho pair would teach the model to associate image content with the
wrong terrain shape. Renders a synthetic hillshade from the heightmap and
compares it against the real orthoimage via SSIM: a real DTM/ortho pair's
shading should structurally resemble the real photo's shading, even though
pixel values differ (synthetic shading vs. real surface reflectance)."""

import math

import numpy as np
from skimage.metrics import structural_similarity


def compute_shaded_relief(heightmap: np.ndarray, pixel_scale_m: float,
                          sun_azimuth_deg: float = 315.0,
                          sun_elevation_deg: float = 45.0) -> np.ndarray:
    """Standard Lambertian hillshade from heightmap's surface normals.
    NaN (nodata) cells are filled with the heightmap's own mean before
    computing gradients, so nodata regions render as locally flat/neutral
    rather than propagating NaN outward through the gradient stencil."""
    if np.all(np.isnan(heightmap)):
        fill_value = 0.0
    else:
        fill_value = float(np.nanmean(heightmap))
    filled = np.where(np.isnan(heightmap), fill_value, heightmap)

    dz_dy, dz_dx = np.gradient(filled, pixel_scale_m)
    normal_x, normal_y, normal_z = -dz_dx, -dz_dy, np.ones_like(filled)
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x, normal_y, normal_z = normal_x / norm, normal_y / norm, normal_z / norm

    az_rad = math.radians(sun_azimuth_deg)
    el_rad = math.radians(sun_elevation_deg)
    sun_x = math.cos(el_rad) * math.sin(az_rad)
    sun_y = math.cos(el_rad) * math.cos(az_rad)
    sun_z = math.sin(el_rad)

    shading = normal_x * sun_x + normal_y * sun_y + normal_z * sun_z
    shading = np.clip(shading, 0.0, 1.0)
    return (shading * 255).astype(np.uint8)


def patch_alignment_score(heightmap: np.ndarray, ortho: np.ndarray,
                          pixel_scale_m: float) -> float:
    """SSIM between heightmap's synthetic hillshade and the real ortho
    patch -- the quality-screening signal used to keep only patches where
    the DTM and ortho genuinely correspond (paper's own screen kept mean
    SSIM > 0.4). Returns 0.0 for an all-nodata heightmap rather than
    computing a meaningless score against a fill-value-flat shading."""
    if np.all(np.isnan(heightmap)):
        return 0.0
    shaded = compute_shaded_relief(heightmap, pixel_scale_m)
    return float(structural_similarity(shaded, ortho.astype(np.uint8), data_range=255))
