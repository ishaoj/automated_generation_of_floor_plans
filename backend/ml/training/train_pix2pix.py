"""
Train Pix2Pix Floor Plan Renderer

Converts semantic room-layout maps into realistic floor plan images.
Input:  color-coded room polygon map rendered from CubiCasa5k SVG data
Target: F1_scaled.png (actual architectural floor plan image)

In the production pipeline the semantic map comes from the GNN-refined
layout; for training we use ground-truth SVG positions directly so the
GNN weights are never loaded or modified here.

Usage:
    python backend/ml/training/train_pix2pix.py \
        --data_dir data/cubicasa5k_extracted/cubicasa5k \
        --output_dir ml/models/saved/pix2pix_v1 \
        --epochs 200 --batch_size 8

Resume:
    python backend/ml/training/train_pix2pix.py \
        --resume ml/models/saved/pix2pix_v1/latest_checkpoint.pt \
        --epochs 200
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.cubicasa_loader import CubiCasaDataset
from ml.models.pix2pix_renderer import Pix2PixModel

# plot_dims is stored in feet; polygon coords are raw SVG pixels at 300 DPI
_PIXEL_TO_FEET = 0.033

_ROOM_COLORS: Dict[str, Tuple[int, int, int]] = {
    "bedroom":     (255, 228, 225),
    "living_room": (224, 240, 255),
    "kitchen":     (255, 228, 181),
    "bathroom":    (230, 230, 250),
    "dining":      (240, 255, 240),
    "storage":     (245, 245, 220),
    "utility":     (245, 245, 220),
    "balcony":     (224, 255, 255),
    "parking":     (220, 220, 220),
    "corridor":    (255, 248, 220),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Pix2PixDataset(Dataset):
    """
    Returns (semantic_map, rendered_image) pairs for pix2pix training.

    semantic_map  — flat color-coded room layout rendered from SVG polygons
    rendered_image — actual F1_scaled.png floor plan photograph
    """

    def __init__(
        self,
        base_dataset: CubiCasaDataset,
        image_size: int = 256,
    ) -> None:
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Keep only samples that have the target PNG
        self.samples: List[Dict] = []
        for s in base_dataset.samples:
            png = Path(s["svg_path"]).parent / "F1_scaled.png"
            if png.exists():
                self.samples.append(s)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "semantic_map":    self.transform(self._make_semantic(sample)),
            "rendered_image":  self.transform(self._load_rendered(sample)),
        }

    def _make_semantic(self, sample: Dict) -> Image.Image:
        plot_w_ft, plot_h_ft = sample["plot_dims"]
        plot_w_px = plot_w_ft / _PIXEL_TO_FEET
        plot_h_px = plot_h_ft / _PIXEL_TO_FEET
        sz = self.image_size
        scale = min(sz / max(plot_w_px, 1), sz / max(plot_h_px, 1))

        img = Image.new("RGB", (sz, sz), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for room in sample["rooms"]:
            color = _ROOM_COLORS.get(room["type"], (255, 255, 255))
            pts = [(x * scale, y * scale) for x, y in room["polygon"]]
            if len(pts) >= 3:
                draw.polygon(pts, fill=color, outline=(0, 0, 0))
        return img

    def _load_rendered(self, sample: Dict) -> Image.Image:
        png = Path(sample["svg_path"]).parent / "F1_scaled.png"
        return Image.open(png).convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Pix2PixTrainer:

    def __init__(
        self,
        model: Pix2PixModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str,
        output_dir: str,
        use_amp: bool = True,
    ) -> None:
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Compile for extra speed (torch ≥2.0, works on MPS too)
        self.model = torch.compile(model.to(device))

        self.train_loader = train_loader
        self.val_loader = val_loader

        lr = config["learning_rate"]
        self.optimizer_g = optim.Adam(
            self.model.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        decay = config.get("lr_decay_epochs", 100)
        gamma = config.get("lr_decay_gamma", 0.5)
        self.scheduler_g = optim.lr_scheduler.StepLR(
            self.optimizer_g, step_size=decay, gamma=gamma
        )
        self.scheduler_d = optim.lr_scheduler.StepLR(
            self.optimizer_d, step_size=decay, gamma=gamma
        )

        # Mixed precision
        self.use_amp = use_amp
        if use_amp and device == "cuda":
            self.scaler_g = torch.cuda.amp.GradScaler()
            self.scaler_d = torch.cuda.amp.GradScaler()
            self._amp_dtype = torch.float16
        else:
            self.scaler_g = self.scaler_d = None
            self._amp_dtype = torch.bfloat16  # MPS; CPU ignores autocast dtype

        self._autocast_ctx = (
            torch.amp.autocast(device_type=device, dtype=self._amp_dtype)
            if use_amp
            else nullcontext()
        )

        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Core loops                                                           #
    # ------------------------------------------------------------------ #

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals = {k: 0.0 for k in
                  ("gen_gan", "gen_l1", "gen_total", "disc_real", "disc_fake", "disc_total")}

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch}/{self.config['epochs']} [Train]",
                    leave=False)
        for batch in pbar:
            cond = batch["semantic_map"].to(self.device, non_blocking=True)
            real = batch["rendered_image"].to(self.device, non_blocking=True)

            # ---- Discriminator ----
            self.optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device, dtype=self._amp_dtype,
                                    enabled=self.use_amp):
                d_losses = self.model.compute_discriminator_loss(cond, real)
            if self.scaler_d:
                self.scaler_d.scale(d_losses["disc_total"]).backward()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                d_losses["disc_total"].backward()
                self.optimizer_d.step()

            # ---- Generator ----
            self.optimizer_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device, dtype=self._amp_dtype,
                                    enabled=self.use_amp):
                g_losses = self.model.compute_generator_loss(cond, real)
            if self.scaler_g:
                self.scaler_g.scale(g_losses["gen_total"]).backward()
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                g_losses["gen_total"].backward()
                self.optimizer_g.step()

            for k in totals:
                if k in g_losses:
                    totals[k] += g_losses[k].item()
                elif k in d_losses:
                    totals[k] += d_losses[k].item()

            pbar.set_postfix(
                G=f"{g_losses['gen_total'].item():.3f}",
                D=f"{d_losses['disc_total'].item():.3f}",
                L1=f"{g_losses['gen_l1'].item():.3f}",
            )
            self.global_step += 1

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        totals = {k: 0.0 for k in
                  ("gen_gan", "gen_l1", "gen_total", "disc_real", "disc_fake", "disc_total")}

        for batch in tqdm(self.val_loader,
                          desc=f"Epoch {epoch} [Val]", leave=False):
            cond = batch["semantic_map"].to(self.device, non_blocking=True)
            real = batch["rendered_image"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device, dtype=self._amp_dtype,
                                    enabled=self.use_amp):
                g = self.model.compute_generator_loss(cond, real)
                d = self.model.compute_discriminator_loss(cond, real)

            for k in totals:
                if k in g:
                    totals[k] += g[k].item()
                elif k in d:
                    totals[k] += d[k].item()

        n = len(self.val_loader)
        return {k: v / n for k, v in totals.items()}

    # ------------------------------------------------------------------ #
    # Checkpoint I/O                                                       #
    # ------------------------------------------------------------------ #

    def _save(self, epoch: int, val_loss: float, is_best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "scheduler_g_state_dict": self.scheduler_g.state_dict(),
            "scheduler_d_state_dict": self.scheduler_d.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(ckpt, self.output_dir / "latest_checkpoint.pt")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pt")
            print(f"  [best] val_G={val_loss:.4f}")
        save_every = self.config.get("save_frequency", 20)
        if epoch % save_every == 0:
            torch.save(ckpt, self.output_dir / f"checkpoint_epoch_{epoch}.pt")
            print(f"  [ckpt] epoch {epoch} saved")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
        self.scheduler_g.load_state_dict(ckpt["scheduler_g_state_dict"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d_state_dict"])
        self.start_epoch = ckpt["epoch"]
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt["best_val_loss"]
        # Restore training history if present
        hist_path = Path(path).parent / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                h = json.load(f)
            self.train_history = h.get("train", [])
            self.val_history = h.get("val", [])
        print(f"Resumed from epoch {self.start_epoch}  "
              f"best_val_loss={self.best_val_loss:.4f}")

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        total = self.config["epochs"]
        val_every = self.config.get("val_every", 5)
        print(f"\nTraining epochs {self.start_epoch + 1} → {total}")
        print(f"Device: {self.device}  AMP: {self.use_amp}  "
              f"Val every: {val_every}  Save every: {self.config.get('save_frequency', 20)}")
        gen_p  = sum(p.numel() for p in self.model.generator.parameters())
        disc_p = sum(p.numel() for p in self.model.discriminator.parameters())
        print(f"Generator: {gen_p:,}  Discriminator: {disc_p:,}\n")

        val_losses: Dict = {}
        for epoch in range(self.start_epoch + 1, total + 1):
            train_losses = self._train_epoch(epoch)
            self.train_history.append({"epoch": epoch, **train_losses})

            # Validate every val_every epochs
            if epoch % val_every == 0 or epoch == total:
                val_losses = self._validate(epoch)
                self.val_history.append({"epoch": epoch, **val_losses})
                is_best = val_losses["gen_total"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses["gen_total"]
                self._save(epoch, val_losses["gen_total"], is_best)
                print(
                    f"Epoch {epoch:3d}/{total} | "
                    f"Train G={train_losses['gen_total']:.4f} "
                    f"D={train_losses['disc_total']:.4f} "
                    f"L1={train_losses['gen_l1']:.4f} | "
                    f"Val G={val_losses['gen_total']:.4f} "
                    f"D={val_losses['disc_total']:.4f} "
                    f"L1={val_losses['gen_l1']:.4f} | "
                    f"lr={self.optimizer_g.param_groups[0]['lr']:.2e}"
                )
            else:
                # Non-val epoch: save latest checkpoint only at save_frequency
                save_every = self.config.get("save_frequency", 20)
                if epoch % save_every == 0:
                    dummy_val = val_losses.get("gen_total", self.best_val_loss)
                    self._save(epoch, dummy_val, is_best=False)
                print(
                    f"Epoch {epoch:3d}/{total} | "
                    f"Train G={train_losses['gen_total']:.4f} "
                    f"D={train_losses['disc_total']:.4f} "
                    f"L1={train_losses['gen_l1']:.4f}"
                )

            self.scheduler_g.step()
            self.scheduler_d.step()

        # Persist history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(
                {"train": self.train_history, "val": self.val_history,
                 "best_val_loss": self.best_val_loss},
                f, indent=2,
            )
        print(f"\nDone. Best val G_loss: {self.best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/cubicasa5k_extracted/cubicasa5k")
    parser.add_argument("--output_dir", default="ml/models/saved/pix2pix_v1")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=0.0002)
    parser.add_argument("--lambda_l1",  type=float, default=100.0)
    parser.add_argument("--image_size", type=int,   default=256)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--ngf",        type=int,   default=32,
                        help="Generator base channels (32=fast, 64=original quality)")
    parser.add_argument("--fp16",       action="store_true",
                        help="Enable mixed-precision (AMP)")
    parser.add_argument("--val_every",  type=int, default=10)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Faster matmul on MPS and CUDA without significant precision loss
    torch.set_float32_matmul_precision("medium")

    config = {
        "data_dir":       args.data_dir,
        "output_dir":     args.output_dir,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  args.lr,
        "lambda_l1":      args.lambda_l1,
        "image_size":     args.image_size,
        "lr_decay_epochs": 100,
        "lr_decay_gamma":  0.5,
        "save_frequency":  args.save_every,
        "val_every":       args.val_every,
    }

    # Datasets
    print("Loading datasets...")
    base_train = CubiCasaDataset(
        data_dir=args.data_dir, split="train", include_images=False,
        filter_non_residential=True, min_rooms=2, max_rooms=10,
    )
    base_val = CubiCasaDataset(
        data_dir=args.data_dir, split="val", include_images=False,
        filter_non_residential=True, min_rooms=2, max_rooms=10,
    )
    train_ds = Pix2PixDataset(base_train, image_size=args.image_size)
    val_ds   = Pix2PixDataset(base_val,   image_size=args.image_size)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = Pix2PixModel(
        in_channels=3, out_channels=3,
        use_multiscale=False, lambda_l1=args.lambda_l1,
        ngf=args.ngf,
    )

    trainer = Pix2PixTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=args.output_dir,
        use_amp=args.fp16,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
    else:
        print("No checkpoint found — starting from epoch 1")

    trainer.train()


if __name__ == "__main__":
    main()
