import functools
import torch
import torch.nn.functional as F
import lightning as L

from models import Autoencoder


def _fft_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MAE on the magnitude spectrum of the real FFT along the time axis."""
    # cuFFT requires float32 for non-power-of-two sizes; cast temporarily.
    mag_hat = torch.abs(torch.fft.rfft(x_hat.float(), dim=-1))
    mag = torch.abs(torch.fft.rfft(x.float(), dim=-1))
    return F.l1_loss(mag_hat, mag)


def _mae_fft_loss(x_hat: torch.Tensor, x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Weighted sum of time-domain MAE and frequency-domain magnitude MAE.

    loss = (1 - alpha) * mae + alpha * fft_mae
    alpha=0 -> pure MAE, alpha=1 -> pure FFT, alpha=0.5 -> equal weight.
    """
    return (1 - alpha) * F.l1_loss(x_hat, x) + alpha * _fft_loss(x_hat, x)


# Default STFT resolutions: (n_fft, hop_length) pairs spanning coarse to fine.
_MRSTFT_RESOLUTIONS = [
    (32,  8),
    (64,  16),
    (128, 32),
    (256, 64),
]


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Compute STFT magnitude for a batch shaped (B, T) or (B, 1, T)."""
    x = x.float()
    if x.dim() == 3:
        x = x.squeeze(1)
    window = torch.hann_window(n_fft, device=x.device)
    return torch.abs(
        torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    )  # (B, F, frames)


def _mrstft_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    resolutions: list = _MRSTFT_RESOLUTIONS,
) -> torch.Tensor:
    """Multi-resolution STFT loss: mean of per-resolution magnitude MAEs."""
    total = torch.tensor(0.0, device=x.device)
    for n_fft, hop in resolutions:
        total = total + F.l1_loss(_stft_mag(x_hat, n_fft, hop), _stft_mag(x, n_fft, hop))
    return total / len(resolutions)


def _mae_mrstft_loss(x_hat: torch.Tensor, x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Weighted sum of time-domain MAE and multi-resolution STFT loss.

    loss = (1 - alpha) * mae + alpha * mrstft
    """
    return (1 - alpha) * F.l1_loss(x_hat, x) + alpha * _mrstft_loss(x_hat, x)


class SignalAutoencoderModule(L.LightningModule):
    """Stage 1: 1D CNN autoencoder trained on raw 625-timestep ECG signal windows."""

    def __init__(
        self,
        seq_len: int = 625,
        depth: int = 4,
        latent_dim: int = 32,
        base_kernels: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        use_scheduler: bool = True,
        loss: str = "mse",
        fft_alpha: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Autoencoder(seq_len=seq_len, depth=depth, latent_dim=latent_dim, base_kernels=base_kernels)
        _supported = {
            "mse": F.mse_loss,
            "mae": F.l1_loss,
            "fft": _fft_loss,
            "mae+fft": functools.partial(_mae_fft_loss, alpha=fft_alpha),
            "mrstft": _mrstft_loss,
            "mae+mrstft": functools.partial(_mae_mrstft_loss, alpha=fft_alpha),
        }
        if loss not in _supported:
            raise ValueError(f"loss must be one of {list(_supported)}, got '{loss}'")
        self._loss_fn = _supported[loss]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch.float()
        loss = self._loss_fn(self(x), x)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.float()
        x_hat = self(x)
        self.log("val/loss", self._loss_fn(x_hat, x), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mae", F.l1_loss(x_hat, x), on_epoch=True, sync_dist=True)
        self.log("val/mse", F.mse_loss(x_hat, x), on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x = batch.float()
        x_hat = self(x)
        self.log("test/loss", self._loss_fn(x_hat, x), sync_dist=True)
        self.log("test/mae", F.l1_loss(x_hat, x), sync_dist=True)
        self.log("test/mse", F.mse_loss(x_hat, x), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.hparams.use_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
