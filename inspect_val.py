"""
Visualize validation-set reconstruction quality for the trained autoencoder.

Ranks every validation sample by per-sample MSE and plots:
  - Best 3 (lowest MSE)
  - Worst 3 (highest MSE)

Each panel shows: original signal, reconstruction, and absolute error.

Run:
    python inspect_val.py
    python inspect_val.py --ckpt checkpoints/signal_autoencoder/epoch=049-val_loss=0.0132.ckpt
    python inspect_val.py --out figures/val_inspection --no_show
"""
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from datamodule import ECGDataModule
from lit_model import SignalAutoencoderModule


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint. Defaults to the best (lowest val_loss) in "
             "checkpoints/signal_autoencoder/.",
    )
    p.add_argument("--out", type=str, default="figures/val_inspection",
                   help="Directory to save figures.")
    p.add_argument("--no_show", action="store_true", help="Skip plt.show().")
    p.add_argument("--device", type=str, default="cpu",
                   help="Inference device: cpu or cuda.")
    return p.parse_args()


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def best_checkpoint(ckpt_dir: str = "checkpoints/signal_autoencoder") -> str:
    """Return the checkpoint path with the smallest val_loss in the filename."""
    ckpts = glob.glob(f"{ckpt_dir}/*.ckpt")
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def val_loss(path: str) -> float:
        name = Path(path).stem
        # filename pattern: epoch=049-val_loss=0.0132
        for part in name.split("-"):
            if part.startswith("val_loss="):
                return float(part.split("=")[1])
        return float("inf")

    return min(ckpts, key=val_loss)


# ── INFERENCE ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: torch.nn.Module, dm: ECGDataModule, device: torch.device):
    """
    Run the model over the entire validation set.

    Returns:
        x_all:     (N, 625) numpy array of original signals
        x_hat_all: (N, 625) numpy array of reconstructions
        mse_all:   (N,)     per-sample MSE
    """
    model.eval()
    xs, x_hats = [], []

    for batch in dm.val_dataloader():
        x = batch.float().to(device)
        x_hat = model(x)
        xs.append(x.cpu())
        x_hats.append(x_hat.cpu())

    x_all = torch.cat(xs, dim=0).numpy()      # (N, 625)
    x_hat_all = torch.cat(x_hats, dim=0).numpy()

    mse_all = ((x_all - x_hat_all) ** 2).mean(axis=1)  # (N,)
    return x_all, x_hat_all, mse_all


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def plot_samples(
    indices: list[int],
    x_all: np.ndarray,
    x_hat_all: np.ndarray,
    mse_all: np.ndarray,
    title_prefix: str,
    out_dir: Path,
    no_show: bool,
):
    """
    For each index, plot a 2-row panel:
      row 0 – original and reconstruction overlaid
      row 1 – absolute error
    All samples are combined into one figure (columns = samples).
    """
    n = len(indices)
    timesteps = np.arange(x_all.shape[1])

    fig, axes = plt.subplots(
        2, n,
        figsize=(5 * n, 5),
        sharex=True,
    )
    # Ensure axes is always 2-D
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, idx in enumerate(indices):
        x = x_all[idx]
        x_hat = x_hat_all[idx]
        err = np.abs(x - x_hat)
        mse = mse_all[idx]

        # Row 0: overlay original and reconstruction
        ax0 = axes[0, col]
        ax0.plot(timesteps, x,     color="steelblue", linewidth=0.8, label="Original")
        ax0.plot(timesteps, x_hat, color="tomato",    linewidth=0.8, label="Reconstruction", alpha=0.85)
        ax0.set_title(f"Sample #{idx}\nMSE={mse:.4f}", fontsize=9)
        if col == 0:
            ax0.set_ylabel("Signal", fontsize=9)
        ax0.legend(fontsize=7, loc="upper right")
        ax0.tick_params(labelsize=7)

        # Row 1: absolute error
        ax1 = axes[1, col]
        ax1.fill_between(timesteps, 0, err, color="darkorange", alpha=0.7, linewidth=0)
        ax1.plot(timesteps, err, color="darkorange", linewidth=0.6)
        if col == 0:
            ax1.set_ylabel("Absolute Error", fontsize=9)
        ax1.set_xlabel("Timestep", fontsize=8)
        ax1.tick_params(labelsize=7)

    fig.suptitle(title_prefix, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{title_prefix.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved → {fname}")

    if not no_show:
        plt.show()
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── Load checkpoint ──
    ckpt_path = args.ckpt or best_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")
    module = SignalAutoencoderModule.load_from_checkpoint(ckpt_path, map_location=device)
    model = module.model.to(device)

    # ── Prepare validation data ──
    dm = ECGDataModule(num_workers=0, pin_memory=False)
    dm.setup()

    # ── Inference ──
    print("Running inference on validation set …")
    x_all, x_hat_all, mse_all = run_inference(model, dm, device)
    print(f"Validation samples: {len(mse_all)}, "
          f"mean MSE: {mse_all.mean():.4f}, "
          f"min: {mse_all.min():.4f}, max: {mse_all.max():.4f}")

    # ── Rank ──
    ranked = np.argsort(mse_all)
    best_idx = ranked[:3].tolist()
    worst_idx = ranked[-3:][::-1].tolist()

    out_dir = Path(args.out)

    # ── Plot ──
    plot_samples(best_idx,  x_all, x_hat_all, mse_all,
                 "Best 3 (lowest MSE)",  out_dir, args.no_show)
    plot_samples(worst_idx, x_all, x_hat_all, mse_all,
                 "Worst 3 (highest MSE)", out_dir, args.no_show)


if __name__ == "__main__":
    main()
