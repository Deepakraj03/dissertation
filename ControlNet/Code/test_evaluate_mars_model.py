import numpy as np
import pytest
from PIL import Image

from evaluate_mars_model import (
    apply_heat_colormap,
    compute_aggregate_stats,
    compute_kid,
    compute_ssim_psnr,
    find_test_triples,
    make_comparison_grid,
    normalize_histogram_global,
    normalize_mean_std,
    polynomial_kernel,
)


def test_compute_aggregate_stats_matches_known_values():
    images = [np.full((4, 4), 10, dtype=np.uint8), np.full((4, 4), 20, dtype=np.uint8)]
    mean, std = compute_aggregate_stats(images)
    # Equal-sized populations at 10 and 20 -> pooled mean 15, known std.
    assert mean == pytest.approx(15.0)
    assert std == pytest.approx(5.0)


def test_normalize_mean_std_matches_target_population_exactly():
    # A flat image at src_mean, after correction to (ref_mean, ref_std),
    # must land exactly on ref_mean (zero-variance edge case of the affine
    # correction -- easiest case to verify precisely).
    image = np.full((8, 8), 50, dtype=np.uint8)
    corrected = normalize_mean_std(image, src_mean=50.0, src_std=10.0,
                                   ref_mean=120.0, ref_std=30.0)
    assert corrected.mean() == pytest.approx(120.0, abs=0.5)


def test_normalize_mean_std_clips_to_valid_range():
    image = np.full((4, 4), 250, dtype=np.uint8)
    corrected = normalize_mean_std(image, src_mean=10.0, src_std=1.0,
                                   ref_mean=10.0, ref_std=200.0)
    assert corrected.max() <= 255
    assert corrected.min() >= 0


def test_normalize_histogram_global_shifts_mean_toward_reference():
    dark_image = np.full((16, 16), 30, dtype=np.uint8)
    bright_reference = np.full((16, 16), 200, dtype=np.uint8).ravel()
    matched = normalize_histogram_global(dark_image, bright_reference)
    assert matched.mean() > dark_image.mean()
    assert matched.mean() == pytest.approx(200.0, abs=1.0)


def test_compute_ssim_psnr_identical_images_are_perfect():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    result = compute_ssim_psnr(image, image)
    assert result["ssim"] == pytest.approx(1.0)
    assert result["psnr"] > 80  # effectively infinite for identical arrays
    assert result["ssim_map"].shape == image.shape


def test_compute_ssim_psnr_lower_for_distinct_images():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    b = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    result = compute_ssim_psnr(a, b)
    assert result["ssim"] < 0.5


def test_polynomial_kernel_shape_and_known_value():
    x = np.array([[1.0, 0.0]])
    y = np.array([[1.0, 0.0]])
    # gamma defaults to 1/d = 0.5; (0.5*1 + 1)^3 = 1.5^3 = 3.375
    k = polynomial_kernel(x, y, degree=3, coef0=1.0)
    assert k.shape == (1, 1)
    assert k[0, 0] == pytest.approx(3.375)


def test_compute_kid_near_zero_for_two_draws_from_same_distribution():
    # Two INDEPENDENT draws from the same distribution (not the same exact
    # points -- that degenerate case has a real, expected negative bias,
    # since the cross term then includes self-pairs k(x_i, x_i), the
    # kernel's maximum value, inflating it above the within-set terms
    # which exclude the diagonal. This is standard unbiased-MMD behavior,
    # not a bug -- see compute_kid's docstring.)
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8))
    fake = rng.normal(size=(200, 8))
    kid = compute_kid(real, fake)
    assert abs(kid) < 0.05


def test_compute_kid_positive_for_distinct_distributions():
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0.0, size=(20, 8))
    fake = rng.normal(loc=50.0, size=(20, 8))
    kid = compute_kid(real, fake)
    assert kid > 0


def test_find_test_triples_matches_and_skips_incomplete(tmp_path):
    condition_dir = tmp_path / "condition"
    generated_dir = tmp_path / "generated"
    condition_dir.mkdir()
    generated_dir.mkdir()

    for pid in ["P1", "P2"]:
        (condition_dir / f"{pid}_condition.png").write_bytes(b"c")
        (condition_dir / f"{pid}_target.jpg").write_bytes(b"t")
        (generated_dir / f"{pid}.png").write_bytes(b"g")

    # P3 has a generated output but no matching real pair -- must be skipped.
    (generated_dir / "P3.png").write_bytes(b"g")

    triples = find_test_triples(condition_dir, generated_dir)

    assert {t["id"] for t in triples} == {"P1", "P2"}


def test_apply_heat_colormap_endpoints():
    normalized = np.array([[0.0, 1.0]])
    rgb = apply_heat_colormap(normalized)
    assert rgb.shape == (1, 2, 3)
    assert tuple(rgb[0, 0]) == (0, 0, 0)  # 0.0 -> black control point
    assert rgb[0, 1, 0] > 200 and rgb[0, 1, 1] > 200  # 1.0 -> pale yellow


def test_make_comparison_grid_writes_expected_size(tmp_path):
    condition = np.zeros((100, 100), dtype=np.uint8)
    real = np.zeros((100, 100), dtype=np.uint8)
    generated = np.zeros((100, 100), dtype=np.uint8)
    ssim_map = np.ones((100, 100), dtype=np.float64)

    out_path = tmp_path / "grid.png"
    make_comparison_grid(condition, real, generated, ssim_map, out_path, panel_size=64)

    assert out_path.exists()
    img = Image.open(out_path)
    assert img.width == 64 * 4 + 4 * 3
    assert img.height == 64 + 22
