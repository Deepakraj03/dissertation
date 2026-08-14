"""Score a DTM heightmap's terrain roughness, to weight global DTM product
selection toward rocky/Gale-analog terrain rather than a uniform sample of
the global catalog (which skews toward smooth terrain like Oxia Planum)."""

import numpy as np


def compute_terrain_roughness(heightmap: np.ndarray, pixel_scale_m: float) -> float:
    """Mean local slope magnitude (rise/run, dimensionless) of heightmap,
    ignoring NaN (nodata) cells. Higher = rockier/more textured terrain;
    lower = smoother. Normalized by pixel_scale_m so scores are comparable
    across DTM products at different native resolutions."""
    dz_dcol = np.diff(heightmap, axis=1)
    dz_drow = np.diff(heightmap, axis=0)
    valid = np.concatenate([
        np.abs(dz_dcol[~np.isnan(dz_dcol)]),
        np.abs(dz_drow[~np.isnan(dz_drow)]),
    ])
    if valid.size == 0:
        return 0.0
    return float(valid.mean()) / pixel_scale_m
