"""
Visualize the overfit test:
  1. Training loss curve (from the latest CSVLogger run)
  2. Actual vs predicted signal overlay for a handful of samples

Run:
    python inspect_overfit.py
    python inspect_overfit.py --depth 3 --latent_dim 64 --epochs 500
    python inspect_overfit.py --n_samples 6 --out figures/overfit/
"""
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datamodule import ECGDataModule
from lit_model import SignalAutoencoderModule


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # ── data ──
    p.add_argument("--batch_size", type=int, default=8,
                   help="Samples to memorize (keep small)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for dataset split and model init")
    # ── model ──
    p.add_argument("--seq_len", type=int, default=625)
    p.add_argument("--depth", type=int, default=4,
                   help="Number of encoder/decoder blocks")
    p.add_argument("--latent_dim", type=int, default=32,
                   help="Latent dim — keep large here so the model can memorize")
    # ── training ──
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Set >0 to test regularisation effect")
    # ── output ──
    p.add_argument("--n_samples", type=int, default=4, help="Signal panels to plot")
    p.add_argument("--out", type=str, default="figures/overfit",
                   help="Directory to save figures")
    p.add_argument("--no_show", action="store_true", help="Skip plt.show()")
    return p.parse_args()


# ── DATA ─────────────────────────────────────────────────────────────────────

def get_overfit_batch(batch_size: int, random_state: int = 42) -> torch.Tensor:
    """Pull one batch from the train split only (never val/test)."""
    dm = ECGDataModule(batch_size=batch_size, num_workers=0, pin_memory=False,
                       random_state=random_state)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    return batch.float()


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train_on_batch(x: torch.Tensor, epochs: int, lr: float, weight_decay: float,
                   seq_len: int, depth: int, latent_dim: int, seed: int):
    """Re-run the overfit loop and return (model, losses)."""
    torch.manual_seed(seed)
    module = SignalAutoencoderModule(seq_len=seq_len, depth=depth, latent_dim=latent_dim,
                                    lr=lr, weight_decay=weight_decay, use_scheduler=False)
    model = module.model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(x), x)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return model, losses


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_loss_curve(losses: list[float], csv_losses: pd.Series | None,
                    out_dir: Path, show: bool):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="steelblue", linewidth=1.2, label="re-run (bare PyTorch)")
    if csv_losses is not None:
        ax.plot(csv_losses.values, color="coral", linewidth=1.2,
                linestyle="--", label="latest Lightning run (CSVLogger)")
    ax.set_xlabel("Epoch / Step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Overfit test — training loss")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    _save_or_show(fig, out_dir / "loss_curve.png", show)


def plot_reconstructions(model: torch.nn.Module, x: torch.Tensor,
                         n: int, out_dir: Path, show: bool):
    model.eval()
    with torch.no_grad():
        x_hat = model(x[:n])

    x_np = x[:n].cpu().numpy()
    x_hat_np = x_hat.cpu().numpy()
    t = np.linspace(0, 2.5, x_np.shape[1])  # 625 pts @ 250 Hz = 2.5 s

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, x_np[i], color="steelblue", linewidth=0.9, label="actual")
        ax.plot(t, x_hat_np[i], color="coral", linewidth=0.9,
                linestyle="--", label="predicted")
        mse = float(F.mse_loss(x_hat[i], x[i]))
        ax.set_title(f"Sample {i}  —  MSE {mse:.4f}", fontsize=9)
        ax.set_ylabel("Amplitude")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Overfit test — actual vs predicted", fontsize=11)
    fig.tight_layout()
    _save_or_show(fig, out_dir / "reconstructions.png", show)


def plot_error(model: torch.nn.Module, x: torch.Tensor,
               n: int, out_dir: Path, show: bool):
    model.eval()
    with torch.no_grad():
        err = (model(x[:n]) - x[:n]).cpu().numpy()

    t = np.linspace(0, 2.5, err.shape[1])
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, err[i], color="mediumpurple", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Error")
        ax.set_title(f"Sample {i}", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Overfit test — reconstruction error (predicted − actual)", fontsize=11)
    fig.tight_layout()
    _save_or_show(fig, out_dir / "error.png", show)


def _save_or_show(fig: plt.Figure, path: Path, show: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"  saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_csv_losses() -> pd.Series | None:
    files = sorted(glob.glob("logs/overfit_test/**/metrics.csv", recursive=True))
    if not files:
        return None
    df = pd.read_csv(files[-1])[["train/loss_epoch"]].dropna().reset_index(drop=True)
    return df["train/loss_epoch"]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out)
    show = not args.no_show

    print("Loading overfit batch...")
    x = get_overfit_batch(args.batch_size, args.seed)
    print(f"  batch: {x.shape}  mean={x.mean():.4f}  std={x.std():.4f}")

    cfg = f"depth={args.depth}  latent_dim={args.latent_dim}  lr={args.lr}  wd={args.weight_decay}"
    print(f"Re-training for {args.epochs} epochs  [{cfg}]...")
    model, losses = train_on_batch(
        x, args.epochs, args.lr, args.weight_decay,
        args.seq_len, args.depth, args.latent_dim, args.seed,
    )
    print(f"  loss: {losses[0]:.4f} → {losses[-1]:.4f}  (mean baseline ≈ {x.var():.4f})")

    csv_losses = load_csv_losses()

    print("Generating figures...")
    plot_loss_curve(losses, csv_losses, out_dir, show)
    plot_reconstructions(model, x, args.n_samples, out_dir, show)
    plot_error(model, x, args.n_samples, out_dir, show)
    print("Done.")


if __name__ == "__main__":
    main()
