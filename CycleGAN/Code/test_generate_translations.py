from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cyclegan import Generator
from generate_translations import generate_all, load_generator


def test_load_generator_restores_g_h2r_weights(tmp_path):
    generator = Generator(in_ch=1, out_ch=1, ngf=64)
    checkpoint_path = tmp_path / "fake_ckpt.pt"
    torch.save({"G_H2R": generator.state_dict()}, checkpoint_path)

    loaded = load_generator(checkpoint_path, torch.device("cpu"))

    assert loaded.training is False  # .eval() was called
    original_param = next(generator.parameters())
    loaded_param = next(loaded.parameters())
    assert torch.equal(original_param, loaded_param)


def test_generate_all_produces_one_output_per_input(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    rng = np.random.default_rng(0)
    for i in range(3):
        arr = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(input_dir / f"patch_{i}.png")

    generator = Generator(in_ch=1, out_ch=1, ngf=64).eval()
    count = generate_all(generator, input_dir, output_dir, torch.device("cpu"))

    assert count == 3
    output_files = sorted(output_dir.glob("*.png"))
    assert len(output_files) == 3
    for f in output_files:
        img = Image.open(f)
        assert img.mode == "RGB"
        assert img.size == (256, 256)


def test_generate_all_respects_limit(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    rng = np.random.default_rng(0)
    for i in range(5):
        arr = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(input_dir / f"patch_{i}.png")

    generator = Generator(in_ch=1, out_ch=1, ngf=64).eval()
    count = generate_all(generator, input_dir, output_dir, torch.device("cpu"), limit=2)

    assert count == 2
    assert len(list(output_dir.glob("*.png"))) == 2


def test_generate_all_clears_stale_output_files_first(tmp_path):
    # Reproduces the real bug: a prior run's leftover generated images
    # silently mixed into a later FID measurement (n_generated was off by
    # exactly the old run's count). output_dir must end up containing only
    # the current call's outputs, even if it previously held more files
    # under different names.
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Simulate leftovers from a previous, larger run.
    for i in range(10):
        (output_dir / f"stale_{i}.png").write_bytes(b"fake old png")

    rng = np.random.default_rng(0)
    for i in range(3):
        arr = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(input_dir / f"patch_{i}.png")

    generator = Generator(in_ch=1, out_ch=1, ngf=64).eval()
    count = generate_all(generator, input_dir, output_dir, torch.device("cpu"))

    assert count == 3
    output_files = sorted(output_dir.glob("*.png"))
    assert len(output_files) == 3
    assert all(f.name.startswith("patch_") for f in output_files)
