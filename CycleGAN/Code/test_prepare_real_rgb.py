from pathlib import Path

import numpy as np
from PIL import Image

from prepare_real_rgb import convert_directory_to_rgb


def test_convert_directory_to_rgb_preserves_pixel_values(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    known = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    Image.fromarray(known, mode="L").save(input_dir / "real_0.png")
    Image.fromarray(known, mode="L").save(input_dir / "real_1.png")

    count = convert_directory_to_rgb(input_dir, output_dir)

    assert count == 2
    output = Image.open(output_dir / "real_0.png")
    assert output.mode == "RGB"
    array = np.array(output)
    assert (array[:, :, 0] == known).all()
    assert (array[:, :, 1] == known).all()
    assert (array[:, :, 2] == known).all()


def test_convert_directory_to_rgb_respects_limit(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    known = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    for i in range(4):
        Image.fromarray(known, mode="L").save(input_dir / f"real_{i}.png")

    count = convert_directory_to_rgb(input_dir, output_dir, limit=2)

    assert count == 2
    assert len(list(output_dir.glob("*.png"))) == 2
