"""
CycleGAN model components.

- Generator:     ResNet-9 (256×256 greyscale input/output)
- Discriminator: PatchGAN-70 (30×30 patch-level predictions)
- ImageBuffer:   Replay buffer to stabilise discriminator training
- weights_init:  Gaussian N(0, 0.02) initialisation (original paper)
"""

import random
import torch
import torch.nn as nn


# ── Building blocks ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with reflection padding and instance normalisation."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    ResNet-9 generator for 256×256 single-channel images.

    Architecture (Zhu et al., CycleGAN, ICCV 2017):
      Encoder  : c7s1-64 → d128 → d256
      Transform: R256 × 9
      Decoder  : u128 → u64 → c7s1-1 → Tanh
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),

            nn.Conv2d(ngf,   ngf*2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),

            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        ]
        res = [ResBlock(ngf * 4) for _ in range(n_blocks)]
        decoder = [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf,   3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*encoder, *res, *decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN discriminator (70×70 receptive field → 30×30 output).

    Each output element judges a 70×70 patch of the input.
    LSGAN loss (MSE) is used rather than BCE.
    """
    def __init__(self, in_ch: int = 1, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1 — no instance norm on first layer
            nn.Conv2d(in_ch, ndf,   4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf,   ndf*2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*8, 1,     4, stride=1, padding=1),   # 30×30 patch map
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ImageBuffer:
    """
    Stores up to `size` previously generated images.
    With probability 0.5, returns a stored image instead of the current one,
    replacing the stored image with the current one.
    Reduces mode collapse and training oscillation.
    """
    def __init__(self, size: int = 50):
        self.size = size
        self.buf: list[torch.Tensor] = []

    def push(self, imgs: torch.Tensor) -> torch.Tensor:
        out = []
        for img in imgs:
            img = img.unsqueeze(0)
            if len(self.buf) < self.size:
                self.buf.append(img)
                out.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.size - 1)
                out.append(self.buf[idx].clone())
                self.buf[idx] = img
            else:
                out.append(img)
        return torch.cat(out, dim=0)


# ── Weight initialisation ─────────────────────────────────────────────────────

def weights_init(m: nn.Module) -> None:
    """Gaussian N(0, 0.02) for Conv; N(1, 0.02) for InstanceNorm weights."""
    name = type(m).__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "Norm" in name and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
