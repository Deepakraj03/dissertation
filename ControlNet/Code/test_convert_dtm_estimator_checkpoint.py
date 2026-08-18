from pathlib import Path

from convert_dtm_estimator_checkpoint import accelerate_model_state_path


def test_accelerate_model_state_path_index_zero_is_unsuffixed():
    path = accelerate_model_state_path(Path("/ckpt"), 0)
    assert path == Path("/ckpt/model.safetensors")


def test_accelerate_model_state_path_index_one_is_suffixed():
    path = accelerate_model_state_path(Path("/ckpt"), 1)
    assert path == Path("/ckpt/model_1.safetensors")


def test_accelerate_model_state_path_index_two_is_suffixed():
    path = accelerate_model_state_path(Path("/ckpt"), 2)
    assert path == Path("/ckpt/model_2.safetensors")
