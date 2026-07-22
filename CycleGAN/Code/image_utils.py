"""Image conversion helpers shared by the CycleGAN evaluation scripts."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def denormalize_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (1, H, W) tensor in [-1, 1] (CycleGAN generator output)
    to an (H, W) uint8 array in [0, 255]."""
    array = tensor.detach().cpu().numpy()[0]  # (H, W)
    array = (array * 0.5 + 0.5) * 255.0
    array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def to_rgb(array: np.ndarray) -> np.ndarray:
    """Replicate an (H, W) single-channel array to (H, W, 3)."""
    return np.stack([array, array, array], axis=-1)


def save_rgb_png(array: np.ndarray, path: Path) -> None:
    """Save an (H, W, 3) uint8 array as a PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
