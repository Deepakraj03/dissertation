"""Encode/decode a real height-map patch (float32 meters) to/from the
[-1, 1] range Stable Diffusion's VAE expects, via per-patch min-max
normalization.

Per-patch, not global: absolute Mars elevation varies by kilometers across
the planet (the corpus assembly saves each patch's real, unnormalized
areoid-referenced elevation), while local relief within one 256px patch is
only meters to tens of meters. The network should learn local terrain
shape from local image texture, not memorize which region of Mars a
patch's absolute elevation belongs to -- a global normalization would
conflate those two very different quantities."""

import numpy as np


def encode_height_to_unit_range(height_patch: np.ndarray) -> tuple[np.ndarray, dict]:
    """Returns (encoded, meta) where encoded is height_patch min-max
    normalized to [-1, 1] and meta = {"min": ..., "max": ...} in the
    original units, needed to invert the mapping later. NaN (nodata)
    cells are filled with the patch's own real minimum before encoding --
    they encode to exactly -1.0, distinguishable from real relief."""
    valid = height_patch[~np.isnan(height_patch)]
    if valid.size == 0:
        return np.zeros_like(height_patch, dtype=np.float32), {"min": 0.0, "max": 1.0}
    h_min, h_max = float(valid.min()), float(valid.max())
    if h_max == h_min:
        h_max = h_min + 1e-6
    filled = np.where(np.isnan(height_patch), h_min, height_patch)
    normalized = (filled - h_min) / (h_max - h_min) * 2.0 - 1.0
    return normalized.astype(np.float32), {"min": h_min, "max": h_max}


def decode_height_from_unit_range(encoded: np.ndarray, meta: dict) -> np.ndarray:
    """Inverse of encode_height_to_unit_range."""
    h_min, h_max = meta["min"], meta["max"]
    return ((encoded + 1.0) / 2.0) * (h_max - h_min) + h_min
