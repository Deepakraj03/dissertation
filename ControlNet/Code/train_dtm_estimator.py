#!/usr/bin/env python
"""Train a ControlNet + LoRA single-image DTM height estimator: condition
on a real HiRISE orthoimage patch, predict the matching height-map patch
(min-max encoded to [-1,1] per height_encoding.py).

Keeps the real diffusion process rather than becoming a plain deterministic
regressor -- single-image height estimation is genuinely ill-posed in
shadowed/texture-poor regions, and diffusion's ability to model a
distribution of plausible outputs is a real fit here (the same reasoning
behind diffusion-based depth estimators like Marigold), not just
infrastructure reuse (2026-08-14 design decision).

L_total = L_diffusion + lambda_height * L_height, where L_height is an L1
loss between a real height-map target and an x0-estimate derived from the
model's noise prediction at each sampled timestep, decoded through the
frozen VAE -- weighted toward low timesteps (the x0-estimate is unreliable
at high noise) via timestep_weight.

Base: stable-diffusion-v1-5/stable-diffusion-v1-5 + lllyasviel/sd-controlnet-depth
(depth-style init matches this task's single-channel height-map target).
LoRA on UNet cross-attention (to_q/to_k/to_v/to_out.0), trained jointly
with the ControlNet weights; base UNet, VAE, and text encoder stay frozen.

NOT YET RUN FOR REAL: the pure functions here (compute_x0_estimate,
timestep_weight, height_regression_loss, DtmEstimationDataset,
build_lora_config) are unit-tested on CPU against real torch/diffusers/peft
(see test_train_dtm_estimator.py). main()'s full training loop requires a
real pretrained-weight download and GPU -- it follows diffusers'
train_controlnet.py's established accelerate structure closely, but has
not been executed end-to-end. Run a short smoke test on real GPU infra
(A4000/Eureka2) before trusting it at scale.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))
from height_encoding import encode_height_to_unit_range

DTM_ESTIMATOR_CAPTION = "a Mars terrain elevation map derived from an orbital image"


class DtmEstimationDataset(Dataset):
    """Loads (orthoimage, height-map) patch pairs written by
    assemble_dtm_estimation_corpus.py — <id>_condition.png (real HiRISE
    orthoimage, uint8 grayscale) + <id>_target.npy (real height, float32
    meters, per-patch min-max encoded to [-1,1] here via
    height_encoding.py)."""

    def __init__(self, split_dir: Path):
        self.split_dir = Path(split_dir)
        self.pairs = []
        for condition_path in sorted(self.split_dir.glob("*_condition.png")):
            base = condition_path.name.removesuffix("_condition.png")
            target_path = self.split_dir / f"{base}_target.npy"
            if target_path.exists():
                self.pairs.append((condition_path, target_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        condition_path, target_path = self.pairs[idx]

        condition = np.array(Image.open(condition_path).convert("L"), dtype=np.float32)
        condition = condition / 127.5 - 1.0

        height_patch = np.load(target_path)
        target, meta = encode_height_to_unit_range(height_patch)

        return {
            "condition_pixel_values": torch.from_numpy(condition).unsqueeze(0),
            "target_pixel_values": torch.from_numpy(target).unsqueeze(0),
            "height_min": meta["min"],
            "height_max": meta["max"],
        }


def collate_fn(examples: list[dict]) -> dict:
    return {
        "condition_pixel_values": torch.stack(
            [e["condition_pixel_values"] for e in examples]).float(),
        "target_pixel_values": torch.stack(
            [e["target_pixel_values"] for e in examples]).float(),
        "height_min": torch.tensor([e["height_min"] for e in examples]),
        "height_max": torch.tensor([e["height_max"] for e in examples]),
    }


def build_lora_config(rank: int = 16, alpha: int = 16) -> LoraConfig:
    """LoRA on the UNet's cross-attention projections, per the 2026-08-14
    design decision — fewer trainable parameters than full fine-tuning,
    lower overfit risk on a corpus this task's real-world coverage limits
    keep modest relative to typical deep-learning training sets."""
    return LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )


def compute_x0_estimate(noisy_latents: torch.Tensor, model_pred: torch.Tensor,
                        timesteps: torch.Tensor, scheduler: DDPMScheduler,
                        ) -> torch.Tensor:
    """Closed-form DDPM x0-estimate from the model's noise prediction at
    the sampled timestep: x0 = (x_t - sqrt(1-alpha_t)*eps) / sqrt(alpha_t).
    Round-trip identity (recovers the true x0 exactly when model_pred is
    the true noise used to corrupt it) is the correctness contract this
    must satisfy — see test_compute_x0_estimate_recovers_true_x0_when_given_true_noise."""
    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_alpha_t = alpha_t.sqrt()
    sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
    return (noisy_latents - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t


def timestep_weight(timesteps: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """Linear ramp: 1.0 at t=0 (cleanest, x0-estimate reliable), 0.0 at the
    highest-noise timestep (x0-estimate unreliable) — scales
    lambda_height's real per-sample contribution, per the 2026-08-14
    design's timestep-weighted x0-estimate approach."""
    return 1.0 - (timesteps.float() / (num_train_timesteps - 1))


def height_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample (not scalar) L1 loss between a decoded x0-estimate and
    the real height-map target, so the caller can apply timestep_weight
    per-sample before reducing to a scalar."""
    return F.l1_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model_name_or_path", type=str,
                        default="lllyasviel/sd-controlnet-depth")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Directory containing train/ (and optionally val/)"
                             " with <id>_condition.png/<id>_target.npy pairs")
    parser.add_argument("--output_dir", type=str, default="dtm_estimator_output")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--lambda_height", type=float, default=0.5,
                        help="Weight on the height-regression loss term")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)

    lora_config = build_lora_config(rank=args.lora_rank, alpha=args.lora_alpha)
    unet = get_peft_model(unet, lora_config)

    # A fixed caption for every sample -- height estimation is conditioned
    # entirely on the depth-style condition map, not text (same convention
    # Track 3's original ControlNet training used).
    fixed_input_ids = tokenizer(
        DTM_ESTIMATOR_CAPTION, padding="max_length",
        max_length=tokenizer.model_max_length, truncation=True,
        return_tensors="pt",
    ).input_ids

    train_dataset = DtmEstimationDataset(Path(args.train_data_dir) / "train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,
    )

    trainable_params = list(controlnet.parameters()) + [
        p for p in unet.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    unet, controlnet, optimizer, train_dataloader = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    num_train_timesteps = noise_scheduler.config.num_train_timesteps
    global_step = 0
    progress_target = args.max_train_steps

    while global_step < progress_target:
        for batch in train_dataloader:
            with accelerator.accumulate(controlnet):
                condition = batch["condition_pixel_values"].repeat(1, 3, 1, 1)
                target = batch["target_pixel_values"].repeat(1, 3, 1, 1)

                target_latents = vae.encode(target).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor

                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                timesteps = torch.randint(
                    0, num_train_timesteps, (bsz,), device=target_latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                encoder_hidden_states = text_encoder(
                    fixed_input_ids.to(accelerator.device).repeat(bsz, 1))[0]

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=condition, return_dict=False,
                )
                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                diffusion_loss = F.mse_loss(model_pred.float(), noise.float())

                x0_estimate_latents = compute_x0_estimate(
                    noisy_latents, model_pred, timesteps, noise_scheduler)
                x0_estimate_pixels = vae.decode(
                    x0_estimate_latents / vae.config.scaling_factor).sample
                x0_estimate_height = x0_estimate_pixels.mean(dim=1, keepdim=True)

                per_sample_height_loss = height_regression_loss(
                    x0_estimate_height, batch["target_pixel_values"])
                weights = timestep_weight(timesteps, num_train_timesteps)
                weighted_height_loss = (per_sample_height_loss * weights).mean()

                loss = diffusion_loss + args.lambda_height * weighted_height_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                if global_step >= progress_target:
                    break

    save_path = Path(args.output_dir) / "final"
    accelerator.save_state(str(save_path))


if __name__ == "__main__":
    main()
