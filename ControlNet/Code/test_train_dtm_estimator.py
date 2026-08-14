import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler
from PIL import Image

from train_dtm_estimator import (
    DtmEstimationDataset,
    build_lora_config,
    compute_x0_estimate,
    height_regression_loss,
    timestep_weight,
)


def test_compute_x0_estimate_recovers_true_x0_when_given_true_noise():
    # Round-trip identity: scheduler.add_noise(x0, noise, t) followed by
    # compute_x0_estimate(x_t, noise, t, scheduler) must recover the
    # original x0 exactly when the "prediction" fed in is the TRUE noise
    # used to corrupt it -- this is the real correctness contract the
    # closed-form DDPM x0 estimate has to satisfy, not an arbitrary
    # formula check.
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, 8, 8)
    noise = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([50, 500])

    noisy = scheduler.add_noise(x0, noise, timesteps)
    recovered = compute_x0_estimate(noisy, noise, timesteps, scheduler)

    torch.testing.assert_close(recovered, x0, atol=1e-4, rtol=1e-4)


def test_timestep_weight_is_one_at_t_zero_and_zero_at_max_t():
    num_train_timesteps = 1000
    timesteps = torch.tensor([0, 999])
    weights = timestep_weight(timesteps, num_train_timesteps)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.0)


def test_timestep_weight_is_monotonically_decreasing():
    timesteps = torch.tensor([0, 250, 500, 750, 999])
    weights = timestep_weight(timesteps, num_train_timesteps=1000)
    assert torch.all(weights[:-1] >= weights[1:])


def test_height_regression_loss_zero_for_identical_inputs():
    pred = torch.randn(2, 1, 16, 16)
    loss = height_regression_loss(pred, pred.clone())
    assert torch.allclose(loss, torch.zeros(2), atol=1e-6)


def test_height_regression_loss_returns_per_sample_not_scalar():
    pred = torch.zeros(3, 1, 8, 8)
    target = torch.ones(3, 1, 8, 8)
    loss = height_regression_loss(pred, target)
    assert loss.shape == (3,)
    assert torch.allclose(loss, torch.ones(3))  # L1 of |0-1| = 1 everywhere


def test_build_lora_config_targets_cross_attention_with_expected_rank():
    config = build_lora_config(rank=16, alpha=16)
    assert config.r == 16
    assert config.lora_alpha == 16
    assert set(config.target_modules) == {"to_q", "to_k", "to_v", "to_out.0"}


def test_dtm_estimation_dataset_loads_real_condition_target_pairs(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    condition = np.random.default_rng(0).integers(0, 255, size=(64, 64), dtype=np.uint8)
    Image.fromarray(condition).save(split_dir / "PROD_p0000_orig_condition.png")

    height = np.random.default_rng(1).uniform(0, 10, size=(64, 64)).astype(np.float32)
    np.save(split_dir / "PROD_p0000_orig_target.npy", height)

    dataset = DtmEstimationDataset(split_dir)

    assert len(dataset) == 1
    item = dataset[0]
    assert item["condition_pixel_values"].shape == (1, 64, 64)
    assert item["target_pixel_values"].shape == (1, 64, 64)
    assert item["condition_pixel_values"].min() >= -1.0
    assert item["condition_pixel_values"].max() <= 1.0
    assert item["target_pixel_values"].min() >= -1.0
    assert item["target_pixel_values"].max() <= 1.0
    assert item["height_min"] == pytest.approx(float(height.min()))
    assert item["height_max"] == pytest.approx(float(height.max()))


def test_dtm_estimation_dataset_only_pairs_matching_condition_and_target(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    # An orphaned condition file with no matching target must not appear
    # in the dataset -- same skip-incomplete-pairs posture as
    # find_test_triples elsewhere in this codebase.
    condition = np.zeros((32, 32), dtype=np.uint8)
    Image.fromarray(condition).save(split_dir / "ORPHAN_condition.png")

    height = np.zeros((32, 32), dtype=np.float32)
    Image.fromarray(condition).save(split_dir / "PAIRED_condition.png")
    np.save(split_dir / "PAIRED_target.npy", height)

    dataset = DtmEstimationDataset(split_dir)

    assert len(dataset) == 1
