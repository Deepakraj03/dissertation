import numpy as np
import torch
from PIL import Image

from image_utils import denormalize_to_uint8, save_rgb_png, to_rgb


def test_denormalize_to_uint8_maps_known_range():
    # -1.0 -> 0, 1.0 -> 255 are unambiguous (no rounding-boundary values).
    tensor = torch.tensor([[[-1.0, 0.0, 1.0]]])  # shape (1, 1, 3)
    result = denormalize_to_uint8(tensor)

    assert result.dtype == np.uint8
    assert result.shape == (1, 3)
    assert result[0, 0] == 0
    assert result[0, 2] == 255


def test_to_rgb_replicates_channel():
    array = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    result = to_rgb(array)

    assert result.shape == (2, 2, 3)
    assert (result[:, :, 0] == array).all()
    assert (result[:, :, 1] == array).all()
    assert (result[:, :, 2] == array).all()


def test_save_rgb_png_round_trips(tmp_path):
    array = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    out_path = tmp_path / "test.png"

    save_rgb_png(array, out_path)

    reloaded = Image.open(out_path)
    assert reloaded.mode == "RGB"
    assert np.array(reloaded).tolist() == array.tolist()
