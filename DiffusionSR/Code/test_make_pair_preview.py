from pathlib import Path

from PIL import Image

from make_pair_preview import make_preview


def test_make_preview_creates_side_by_side_composite(tmp_path):
    pair_dir = tmp_path / "ESP_TEST_0001"
    pair_dir.mkdir()
    Image.new("L", (100, 200)).save(pair_dir / "lr.tif")
    Image.new("L", (300, 300)).save(pair_dir / "hr.jpg")

    out_path = make_preview(pair_dir)

    assert out_path == pair_dir / "preview.png"
    assert out_path.exists()
    composite = Image.open(out_path)
    # Height matches hr.jpg (300); width is lr (resized to height=300,
    # preserving its 100:200 aspect -> 150) + hr (300) = 450.
    assert composite.height == 300
    assert composite.width == 450
