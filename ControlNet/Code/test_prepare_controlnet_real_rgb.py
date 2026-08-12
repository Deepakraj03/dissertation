import numpy as np
from PIL import Image

from prepare_controlnet_real_rgb import convert_directory_to_rgb


def test_convert_directory_to_rgb_renames_and_converts(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    known = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    Image.fromarray(known, mode="L").save(input_dir / "P1_target.jpg", quality=100)

    count = convert_directory_to_rgb(input_dir, output_dir)

    assert count == 1
    assert (output_dir / "P1.png").exists()
    assert Image.open(output_dir / "P1.png").mode == "RGB"


def test_convert_directory_to_rgb_only_matches_target_files(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    known = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    Image.fromarray(known, mode="L").save(input_dir / "P1_target.jpg", quality=100)
    Image.fromarray(known, mode="L").save(input_dir / "P1_condition.png")

    count = convert_directory_to_rgb(input_dir, output_dir)

    assert count == 1
    assert list(output_dir.glob("*.png")) == [output_dir / "P1.png"]
