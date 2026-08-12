from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from generate_controlnet_translations import generate_all


def _make_fake_pipe():
    """A pipe(...) call returns an object with .images[0] -- a PIL Image,
    matching diffusers' StableDiffusionControlNetPipeline output shape."""
    pipe = MagicMock()
    pipe.device = "cpu"
    fake_output = MagicMock()
    fake_output.images = [Image.fromarray(
        np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")]
    pipe.return_value = fake_output
    return pipe


def test_generate_all_produces_one_output_per_condition_map(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    for pid in ["P1", "P2", "P3"]:
        Image.new("L", (64, 64)).save(input_dir / f"{pid}_condition.png")
        (input_dir / f"{pid}_target.jpg").write_bytes(b"fake jpg")

    pipe = _make_fake_pipe()
    count = generate_all(pipe, input_dir, output_dir)

    assert count == 3
    assert sorted(p.name for p in output_dir.glob("*.png")) == ["P1.png", "P2.png", "P3.png"]


def test_generate_all_only_reads_condition_maps_not_targets(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    Image.new("L", (64, 64)).save(input_dir / "P1_condition.png")
    (input_dir / "P1_target.jpg").write_bytes(b"fake jpg")

    pipe = _make_fake_pipe()
    generate_all(pipe, input_dir, output_dir)

    assert pipe.call_count == 1


def test_generate_all_respects_limit(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    for pid in ["P1", "P2", "P3"]:
        Image.new("L", (64, 64)).save(input_dir / f"{pid}_condition.png")

    pipe = _make_fake_pipe()
    count = generate_all(pipe, input_dir, output_dir, limit=2)

    assert count == 2


def test_generate_all_clears_stale_output_files_first(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (output_dir / "stale.png").write_bytes(b"old")
    Image.new("L", (64, 64)).save(input_dir / "P1_condition.png")

    pipe = _make_fake_pipe()
    generate_all(pipe, input_dir, output_dir)

    output_files = sorted(output_dir.glob("*.png"))
    assert len(output_files) == 1
    assert output_files[0].name == "P1.png"
