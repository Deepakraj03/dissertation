"""
CycleGAN training script — HiRISE (nadir) ↔ Rover NAVCAM (perspective).

Trains two generators and two discriminators using unpaired patches from:
  Data/processed/hirise/train/   (domain A — nadir orbital)
  Data/processed/rover/train/    (domain B — perspective rover)

Usage:
    python train_cyclegan.py                        # defaults
    python train_cyclegan.py --epochs 200 --batch 4
    python train_cyclegan.py --resume checkpoints/epoch_050.pt
    python train_cyclegan.py --cpu                  # force CPU (slow)

Outputs:
    checkpoints/epoch_{N:03d}.pt     model weights
    samples/epoch_{N:03d}.png        visual comparison grid
    logs/                            TensorBoard event files
    training_log.csv                 loss history
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Allow running from either Code/ or project root
sys.path.insert(0, str(Path(__file__).parent))
from cyclegan import Generator, Discriminator, ImageBuffer, weights_init

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_HIRISE = ROOT / "Data" / "processed" / "hirise"
DATA_ROVER  = ROOT / "Data" / "processed" / "rover"
CKPT_DIR    = ROOT / "checkpoints"
SAMPLE_DIR  = ROOT / "samples"
LOG_DIR     = ROOT / "logs"


# ── Dataset ───────────────────────────────────────────────────────────────────

class UnpairedDataset(Dataset):
    """
    Returns (img_A, img_B) pairs sampled independently from two directories.
    The two domains are not aligned — CycleGAN requires no correspondence.
    Images are loaded as greyscale, normalised to [-1, 1].
    """
    def __init__(self, dir_a: Path, dir_b: Path, size: int = 256):
        self.files_a = sorted(dir_a.glob("*.png"))
        self.files_b = sorted(dir_b.glob("*.png"))
        if not self.files_a:
            raise FileNotFoundError(f"No PNG files in {dir_a}")
        if not self.files_b:
            raise FileNotFoundError(f"No PNG files in {dir_b}")
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(size, transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                   # [0, 1]
            transforms.Normalize([0.5], [0.5]),      # → [-1, 1]
        ])

    def __len__(self) -> int:
        # An epoch is one pass through the SMALLER domain. Defining it by
        # the larger domain instead would silently over-repeat the smaller
        # one every epoch (e.g. the geometry corpus, at ~1/17th the size of
        # the rover corpus, would be shown ~17x per epoch) with no way to
        # tell from the training log alone.
        return min(len(self.files_a), len(self.files_b))

    def __getitem__(self, idx: int):
        # The smaller domain is indexed directly (idx already in range, one
        # full pass per epoch, no repeats/skips). The larger domain is
        # sampled independently at random each call rather than wrapped
        # with idx % len(files) — a naive modulo wrap would permanently
        # restrict it to only its first len(smaller domain) sorted files,
        # discarding the rest of a much larger real corpus for all of
        # training. Random sampling keeps its full range reachable across
        # training instead.
        if len(self.files_a) <= len(self.files_b):
            path_a = self.files_a[idx]
            path_b = self.files_b[random.randint(0, len(self.files_b) - 1)]
        else:
            path_a = self.files_a[random.randint(0, len(self.files_a) - 1)]
            path_b = self.files_b[idx]
        img_a = Image.open(path_a)
        img_b = Image.open(path_b)
        return self.tf(img_a), self.tf(img_b)


# ── Loss helpers ──────────────────────────────────────────────────────────────

class LSGANLoss(nn.Module):
    """Least-squares GAN loss (Mao et al., 2017). More stable than BCE."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def __call__(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.mse(pred, target)


# ── LR scheduler ─────────────────────────────────────────────────────────────

def make_lr_lambda(n_epochs_const: int, n_epochs_decay: int):
    """Constant LR for first n_epochs_const, then linear decay to 0."""
    def lr_lambda(epoch: int) -> float:
        if epoch < n_epochs_const:
            return 1.0
        return max(0.0, 1.0 - (epoch - n_epochs_const) / n_epochs_decay)
    return lr_lambda


# ── Sample grid ───────────────────────────────────────────────────────────────

def save_sample_grid(real_h: torch.Tensor, real_r: torch.Tensor,
                     fake_r: torch.Tensor, fake_h: torch.Tensor,
                     rec_h:  torch.Tensor, rec_r:  torch.Tensor,
                     path: Path) -> None:
    """Save a 2×3 grid: [real_H, fake_R, rec_H] / [real_R, fake_H, rec_R]."""
    try:
        from torchvision.utils import save_image
        imgs = torch.cat([
            real_h[:1], fake_r[:1], rec_h[:1],
            real_r[:1], fake_h[:1], rec_r[:1],
        ], dim=0)
        # Denormalise [-1,1] → [0,1]
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        save_image(imgs, path, nrow=3, padding=4)
    except Exception:
        pass   # non-critical — skip if torchvision save fails


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Directories
    CKPT_DIR.mkdir(exist_ok=True)
    SAMPLE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    # ── Dataset & loader ──────────────────────────────────────────────────────
    domain_a_dir = Path(args.domain_a_dir) if args.domain_a_dir else DATA_HIRISE
    dataset = UnpairedDataset(
        domain_a_dir / "train",
        DATA_ROVER   / "train",
        size=args.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"Dataset: {len(dataset.files_a)} domain-A ({domain_a_dir.name}) + "
          f"{len(dataset.files_b)} Rover patches → "
          f"{len(loader)} batches/epoch")

    # ── Models ────────────────────────────────────────────────────────────────
    G_H2R = Generator(in_ch=1, out_ch=1, ngf=args.ngf).to(device)  # nadir → perspective
    G_R2H = Generator(in_ch=1, out_ch=1, ngf=args.ngf).to(device)  # perspective → nadir
    D_H   = Discriminator(in_ch=1, ndf=args.ndf).to(device)         # judges HiRISE
    D_R   = Discriminator(in_ch=1, ndf=args.ndf).to(device)         # judges Rover

    G_H2R.apply(weights_init)
    G_R2H.apply(weights_init)
    D_H.apply(weights_init)
    D_R.apply(weights_init)

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt_G = torch.optim.Adam(
        list(G_H2R.parameters()) + list(G_R2H.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    opt_D_H = torch.optim.Adam(D_H.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_R = torch.optim.Adam(D_R.parameters(), lr=args.lr, betas=(0.5, 0.999))

    n_const = args.epochs // 2
    n_decay = args.epochs - n_const
    lr_lambda = make_lr_lambda(n_const, n_decay)
    sched_G   = torch.optim.lr_scheduler.LambdaLR(opt_G,   lr_lambda)
    sched_D_H = torch.optim.lr_scheduler.LambdaLR(opt_D_H, lr_lambda)
    sched_D_R = torch.optim.lr_scheduler.LambdaLR(opt_D_R, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G_H2R.load_state_dict(ckpt["G_H2R"])
        G_R2H.load_state_dict(ckpt["G_R2H"])
        D_H.load_state_dict(ckpt["D_H"])
        D_R.load_state_dict(ckpt["D_R"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D_H.load_state_dict(ckpt["opt_D_H"])
        opt_D_R.load_state_dict(ckpt["opt_D_R"])
        start_epoch = ckpt["epoch"] + 1
        # Fast-forward schedulers
        for _ in range(start_epoch):
            sched_G.step(); sched_D_H.step(); sched_D_R.step()
        print(f"Resumed from epoch {start_epoch}")

    # ── Losses ────────────────────────────────────────────────────────────────
    crit_gan   = LSGANLoss()
    crit_cycle = nn.L1Loss()
    crit_idt   = nn.L1Loss()

    buf_H = ImageBuffer(size=50)
    buf_R = ImageBuffer(size=50)

    # ── AMP scaler (CUDA only) ────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(LOG_DIR)
    except ImportError:
        print("  TensorBoard not available — logging to CSV only")

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path = ROOT / "training_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "loss_G", "loss_D_H", "loss_D_R",
                         "loss_cyc_H", "loss_cyc_R", "lr", "epoch_time_s"])

    print(f"\nStarting training: {args.epochs} epochs, "
          f"batch={args.batch}, λ_cyc={args.lambda_cyc}, λ_idt={args.lambda_idt}")
    print(f"Checkpoints → {CKPT_DIR}")
    print(f"Samples     → {SAMPLE_DIR}\n")

    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        G_H2R.train(); G_R2H.train(); D_H.train(); D_R.train()

        sum_G = sum_DH = sum_DR = sum_cyc_H = sum_cyc_R = 0.0

        for i, (real_H, real_R) in enumerate(loader):
            if args.max_steps and i >= args.max_steps:
                break
            real_H = real_H.to(device, non_blocking=True)
            real_R = real_R.to(device, non_blocking=True)

            # ── 1. Generators ─────────────────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake_R  = G_H2R(real_H)        # HiRISE → Rover
                fake_H  = G_R2H(real_R)        # Rover  → HiRISE
                rec_H   = G_R2H(fake_R)        # cycle  back
                rec_R   = G_H2R(fake_H)
                idt_H   = G_R2H(real_H)        # identity
                idt_R   = G_H2R(real_R)

                loss_gan_H2R = crit_gan(D_R(fake_R), True)
                loss_gan_R2H = crit_gan(D_H(fake_H), True)

                loss_cyc_H   = crit_cycle(rec_H, real_H) * args.lambda_cyc
                loss_cyc_R   = crit_cycle(rec_R, real_R) * args.lambda_cyc

                loss_idt_H   = crit_idt(idt_H, real_H) * args.lambda_idt
                loss_idt_R   = crit_idt(idt_R, real_R) * args.lambda_idt

                loss_G = (loss_gan_H2R + loss_gan_R2H
                          + loss_cyc_H + loss_cyc_R
                          + loss_idt_H + loss_idt_R)

            opt_G.zero_grad(set_to_none=True)
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

            # ── 2. Discriminator D_H ──────────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake_H_buf = buf_H.push(fake_H.detach())
                loss_D_H = 0.5 * (
                    crit_gan(D_H(real_H),     True) +
                    crit_gan(D_H(fake_H_buf), False)
                )
            opt_D_H.zero_grad(set_to_none=True)
            scaler.scale(loss_D_H).backward()
            scaler.step(opt_D_H)

            # ── 3. Discriminator D_R ──────────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake_R_buf = buf_R.push(fake_R.detach())
                loss_D_R = 0.5 * (
                    crit_gan(D_R(real_R),     True) +
                    crit_gan(D_R(fake_R_buf), False)
                )
            opt_D_R.zero_grad(set_to_none=True)
            scaler.scale(loss_D_R).backward()
            scaler.step(opt_D_R)

            scaler.update()

            sum_G     += loss_G.item()
            sum_DH    += loss_D_H.item()
            sum_DR    += loss_D_R.item()
            sum_cyc_H += loss_cyc_H.item()
            sum_cyc_R += loss_cyc_R.item()

            global_step += 1
            if (i + 1) % args.log_every == 0:
                print(f"  [{epoch+1:03d}/{args.epochs}] "
                      f"step {i+1}/{len(loader)} | "
                      f"G={loss_G.item():.3f}  "
                      f"D_H={loss_D_H.item():.3f}  "
                      f"D_R={loss_D_R.item():.3f}  "
                      f"cyc={loss_cyc_H.item():.3f}/{loss_cyc_R.item():.3f}")

        # ── End of epoch ──────────────────────────────────────────────────────
        n = len(loader)
        avg = dict(G=sum_G/n, DH=sum_DH/n, DR=sum_DR/n,
                   cyc_H=sum_cyc_H/n, cyc_R=sum_cyc_R/n)
        elapsed = time.time() - t0
        lr_now  = opt_G.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"G={avg['G']:.4f}  D_H={avg['DH']:.4f}  D_R={avg['DR']:.4f}  "
              f"cyc_H={avg['cyc_H']:.4f}  cyc_R={avg['cyc_R']:.4f}  "
              f"lr={lr_now:.2e}  t={elapsed:.0f}s")

        sched_G.step(); sched_D_H.step(); sched_D_R.step()

        # TensorBoard
        if writer:
            writer.add_scalars("Loss/Generator",     {"total": avg["G"]}, epoch)
            writer.add_scalars("Loss/Discriminator", {"D_H": avg["DH"], "D_R": avg["DR"]}, epoch)
            writer.add_scalars("Loss/Cycle",         {"H": avg["cyc_H"], "R": avg["cyc_R"]}, epoch)
            writer.add_scalar("LR", lr_now, epoch)

        # CSV
        csv_writer.writerow([epoch+1, avg["G"], avg["DH"], avg["DR"],
                              avg["cyc_H"], avg["cyc_R"], lr_now, f"{elapsed:.1f}"])
        csv_file.flush()

        # Samples
        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            G_H2R.eval(); G_R2H.eval()
            with torch.no_grad():
                save_sample_grid(
                    real_H, real_R,
                    G_H2R(real_H), G_R2H(real_R),
                    G_R2H(G_H2R(real_H)), G_H2R(G_R2H(real_R)),
                    SAMPLE_DIR / f"epoch_{epoch+1:03d}.png",
                )

        # Checkpoints
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ckpt_path = CKPT_DIR / f"epoch_{epoch+1:03d}.pt"
            torch.save({
                "epoch":   epoch,
                "G_H2R":   G_H2R.state_dict(),
                "G_R2H":   G_R2H.state_dict(),
                "D_H":     D_H.state_dict(),
                "D_R":     D_R.state_dict(),
                "opt_G":   opt_G.state_dict(),
                "opt_D_H": opt_D_H.state_dict(),
                "opt_D_R": opt_D_R.state_dict(),
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path.name}")

    csv_file.close()
    if writer:
        writer.close()
    print(f"\nTraining complete. Log: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train CycleGAN: HiRISE ↔ Rover NAVCAM"
    )
    # Training schedule
    p.add_argument("--epochs",      type=int,   default=200,
                   help="Total epochs (first half constant LR, second half linear decay)")
    p.add_argument("--batch",       type=int,   default=1,
                   help="Batch size (1 is standard for CycleGAN)")
    p.add_argument("--lr",          type=float, default=2e-4,
                   help="Initial learning rate")
    p.add_argument("--lambda-cyc",  type=float, default=10.0,
                   help="Cycle consistency loss weight")
    p.add_argument("--lambda-idt",  type=float, default=5.0,
                   help="Identity loss weight")
    p.add_argument("--img-size",    type=int,   default=256)

    # Architecture
    p.add_argument("--ngf",         type=int,   default=64,
                   help="Generator base filters")
    p.add_argument("--ndf",         type=int,   default=64,
                   help="Discriminator base filters")

    # Logging & checkpointing
    p.add_argument("--log-every",    type=int,   default=200,
                   help="Print interval in steps")
    p.add_argument("--sample-every", type=int,   default=5,
                   help="Save sample images every N epochs")
    p.add_argument("--save-every",   type=int,   default=10,
                   help="Save checkpoint every N epochs")

    # Data
    p.add_argument("--domain-a-dir", type=str,  default=None,
                   help="Override domain-A (nadir) directory "
                        f"(default: {DATA_HIRISE})")

    # System
    p.add_argument("--workers",     type=int,   default=4,
                   help="DataLoader worker processes")
    p.add_argument("--cpu",         action="store_true",
                   help="Force CPU (GPU not used even if available)")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint .pt to resume from")
    p.add_argument("--max-steps",   type=int,   default=None,
                   help="Stop after N steps per epoch (useful for smoke-testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
