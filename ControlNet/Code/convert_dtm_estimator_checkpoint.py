#!/usr/bin/env python
"""Convert a raw accelerator.save_state() checkpoint from
train_dtm_estimator.py into the from_pretrained-loadable layout
evaluate_dtm_estimator.py expects: a real ControlNetModel directory and a
real PEFT adapter directory for the LoRA-wrapped UNet.

accelerator.prepare(unet, controlnet, optimizer, train_dataloader) in
train_dtm_estimator.py registers unet first, controlnet second, so
accelerate's save_state naming convention (model.safetensors for
registration index 0, model_1.safetensors for index 1, ...) puts the
LoRA-wrapped UNet's full state dict in model.safetensors and the
ControlNet's full state dict in model_1.safetensors -- not documented,
confirmed by reading accelerate/checkpointing.py directly.
"""

import argparse
from pathlib import Path

from diffusers import ControlNetModel, UNet2DConditionModel
from peft import get_peft_model
from safetensors.torch import load_file

from train_dtm_estimator import build_lora_config


def accelerate_model_state_path(checkpoint_dir, index: int) -> Path:
    """accelerate's save_state model file naming: model.safetensors at
    index 0, model_1.safetensors at index 1, model_2.safetensors at index
    2, etc."""
    name = "model.safetensors" if index == 0 else f"model_{index}.safetensors"
    return Path(checkpoint_dir) / name


def convert_controlnet(checkpoint_dir, controlnet_base: str, out_dir):
    controlnet = ControlNetModel.from_pretrained(controlnet_base)
    state_dict = load_file(str(accelerate_model_state_path(checkpoint_dir, 1)))
    controlnet.load_state_dict(state_dict)
    controlnet.save_pretrained(str(out_dir))


def convert_unet_lora(checkpoint_dir, base_model: str, out_dir,
                      lora_rank: int = 16, lora_alpha: int = 16):
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    unet = get_peft_model(unet, build_lora_config(rank=lora_rank, alpha=lora_alpha))
    state_dict = load_file(str(accelerate_model_state_path(checkpoint_dir, 0)))
    unet.load_state_dict(state_dict)
    unet.save_pretrained(str(out_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Raw accelerator.save_state() checkpoint dir"
                             " (e.g. checkpoints/dtm_estimator/final)")
    parser.add_argument("--base-model", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-base", type=str,
                        default="lllyasviel/sd-controlnet-depth")
    parser.add_argument("--out-controlnet-dir", type=str, required=True)
    parser.add_argument("--out-unet-lora-dir", type=str, required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    convert_controlnet(args.checkpoint_dir, args.controlnet_base, args.out_controlnet_dir)
    convert_unet_lora(args.checkpoint_dir, args.base_model, args.out_unet_lora_dir,
                      lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    print(f"Converted controlnet -> {args.out_controlnet_dir}")
    print(f"Converted unet LoRA adapter -> {args.out_unet_lora_dir}")


if __name__ == "__main__":
    main()
