import json
from pathlib import Path

import numpy as np
from PIL import Image

from compute_fid import compute_fid_score, frechet_distance, write_results


def test_frechet_distance_is_zero_for_identical_distributions():
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.eye(3)

    result = frechet_distance(mu, sigma, mu, sigma)

    assert abs(result) < 1e-6


def test_compute_fid_score_is_near_zero_for_identical_folders(tmp_path):
    # Reimplemented frechet_distance was verified during planning to
    # produce ~0 for a folder compared against itself using real
    # InceptionV3 features (clean-fid's own version crashes on this
    # environment's scipy — see Global Constraints).
    folder = tmp_path / "images"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(folder / f"img_{i}.png")

    score = compute_fid_score(folder, folder)

    assert score < 1.0


def test_write_results_creates_valid_json(tmp_path):
    out_path = tmp_path / "results" / "fid_results.json"

    write_results(score=42.5, checkpoint="epoch_025.pt",
                  n_generated=4633, n_real=2553, out_path=out_path)

    payload = json.loads(out_path.read_text())
    assert payload["fid"] == 42.5
    assert payload["checkpoint"] == "epoch_025.pt"
    assert payload["n_generated"] == 4633
    assert payload["n_real"] == 2553
    assert "timestamp" in payload
