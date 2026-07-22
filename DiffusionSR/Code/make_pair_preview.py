"""Build a side-by-side LR|HR preview PNG for manual co-registration QA."""

from pathlib import Path

from PIL import Image


def make_preview(pair_dir: Path) -> Path:
    """Create pair_dir/preview.png: the lr crop (resized to hr's height,
    preserving aspect ratio) placed next to the hr crop. Returns the
    output path."""
    lr = Image.open(pair_dir / "lr.tif").convert("L")
    hr = Image.open(pair_dir / "hr.jpg").convert("L")

    scale = hr.height / lr.height
    lr_resized = lr.resize((round(lr.width * scale), hr.height))

    composite = Image.new("L", (lr_resized.width + hr.width, hr.height))
    composite.paste(lr_resized, (0, 0))
    composite.paste(hr, (lr_resized.width, 0))

    out_path = pair_dir / "preview.png"
    composite.save(out_path)
    return out_path
