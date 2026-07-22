from pathlib import Path

from preprocess import split_and_move


def test_split_and_move_respects_ratios(tmp_path):
    staging = tmp_path / "_staging"
    staging.mkdir()
    patches = []
    for i in range(100):
        p = staging / f"patch_{i:03d}.png"
        p.write_bytes(b"fake png data")
        patches.append(p)

    out_dir = tmp_path / "out"
    result = split_and_move(patches, out_dir,
                            splits={"train": 0.80, "val": 0.10, "test": 0.10})

    assert result == {"total_patches": 100, "train": 80, "val": 10, "test": 10}
    assert len(list((out_dir / "train").glob("*.png"))) == 80
    assert len(list((out_dir / "val").glob("*.png"))) == 10
    assert len(list((out_dir / "test").glob("*.png"))) == 10
    # Originals were moved, not copied.
    assert not any(p.exists() for p in patches)


def test_split_and_move_handles_empty_list(tmp_path):
    out_dir = tmp_path / "out"
    result = split_and_move([], out_dir,
                            splits={"train": 0.80, "val": 0.10, "test": 0.10})

    assert result == {"total_patches": 0, "train": 0, "val": 0, "test": 0}
