from pathlib import Path

import numpy as np
from PIL import Image

from train_cyclegan import UnpairedDataset


def _make_pngs(dir_path: Path, n: int) -> None:
    dir_path.mkdir(parents=True)
    for i in range(n):
        arr = np.full((32, 32), i % 256, dtype=np.uint8)
        Image.fromarray(arr).save(dir_path / f"img_{i:04d}.png")


def test_unpaired_dataset_length_is_smaller_domain_size(tmp_path):
    # domain A (geometry corpus) is much smaller than domain B (rover) in
    # the real training setup — an epoch must be defined by the smaller
    # domain so it isn't silently over-repeated relative to a fixed larger
    # epoch length.
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _make_pngs(dir_a, 5)
    _make_pngs(dir_b, 20)
    ds = UnpairedDataset(dir_a, dir_b)
    assert len(ds) == 5


def test_unpaired_dataset_smaller_domain_covered_exactly_once_per_epoch(tmp_path):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _make_pngs(dir_a, 5)
    _make_pngs(dir_b, 20)
    ds = UnpairedDataset(dir_a, dir_b)

    seen_a_values = []
    for idx in range(len(ds)):
        img_a, _ = ds[idx]
        # RandomHorizontalFlip doesn't change the constant fill value used
        # by _make_pngs, so the normalised pixel value identifies which
        # source file was loaded.
        seen_a_values.append(round(img_a.mean().item(), 3))

    # Every one of the 5 domain-A images must appear exactly once — no
    # skips, no repeats, within a single epoch.
    assert len(set(seen_a_values)) == 5


def test_unpaired_dataset_larger_domain_not_restricted_to_a_fixed_prefix(tmp_path):
    # Regression guard: a naive fix that just does `idx % len(files_b)` with
    # an epoch length capped to the smaller domain's size would permanently
    # restrict domain B sampling to only its first len(files_a) sorted
    # files, silently discarding the rest of a much larger real corpus
    # (e.g. only ever seeing rover images 0-1209 out of 20,417). Domain B
    # must be reachable across its full range given enough draws.
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _make_pngs(dir_a, 5)
    _make_pngs(dir_b, 20)
    ds = UnpairedDataset(dir_a, dir_b)

    seen_b_values = set()
    for _ in range(500):  # many draws, well beyond one epoch (len == 5)
        _, img_b = ds[0]
        seen_b_values.add(round(img_b.mean().item(), 3))

    # With 20 possible domain-B files and 500 random draws, every file
    # should be seen at least once, including ones beyond index 5.
    assert len(seen_b_values) == 20
