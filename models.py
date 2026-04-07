import math
from torch import nn
import torch.nn.functional as F


def _pool_flags(padded_seq_len, depth, latent_dim):
    """Return per-layer booleans (True=apply pool) and the resulting compressed seq len.

    Pool is skipped at a layer when halving would bring seq_len below latent_dim,
    ensuring the spatial dimension is always >= latent_dim.
    """
    flags = []
    seq = padded_seq_len
    for _ in range(depth):
        if seq // 2 >= latent_dim:
            flags.append(True)
            seq //= 2
        else:
            flags.append(False)
    return flags, seq


class Encoder(nn.Module):
    def __init__(self, seq_len, depth, latent_dim, base_kernels: int = 16): # 625, 4, 64
        super().__init__()
        self.seq_len = seq_len
        self.depth = depth
        padded_seq_len = math.ceil(seq_len / (2 ** depth)) * (2 ** depth)

        self.pool_flags, compressed_seq_len = _pool_flags(padded_seq_len, depth, latent_dim)

        # double channels and halve sequence length at each layer
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(depth):
            in_channels = 1 if i == 0 else base_kernels * (2 ** (i - 1))
            out_channels = base_kernels * (2 ** i) if i < depth - 1 else latent_dim
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.1),
                )
            )
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # collapse the channel feature maps into a single vector maintaining the sequence order
        self.pool_conv = nn.Conv1d(latent_dim, 1, kernel_size=1, stride=1, padding=0)
        # map compressed spatial dim to latent dimension
        self.latent_linear = nn.Linear(compressed_seq_len, latent_dim)

    def forward(self, x):
        pad = (2 ** self.depth - self.seq_len % (2 ** self.depth)) % (2 ** self.depth)
        x = F.pad(x, (0, pad))  # pad to next 2^depth multiple
        x = x.unsqueeze(1)  # (B, 1, padded_seq_len)
        for conv, pool, do_pool in zip(self.conv_layers, self.pool_layers, self.pool_flags):
            x = conv(x)
            if do_pool:
                x = pool(x)
            # print(f"After layer (pool={do_pool}): {x.shape}")
        x = self.pool_conv(x)
        # print(f"After pool_conv: {x.shape}")
        x = x.view(x.size(0), -1)
        x = self.latent_linear(x)
        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, depth, latent_dim, base_kernels: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.depth = depth
        padded_seq_len = math.ceil(seq_len / (2 ** depth)) * (2 ** depth)

        pool_flags, compressed_seq_len = _pool_flags(padded_seq_len, depth, latent_dim)
        # decoder upsamples in the reverse order of encoder's pools
        self.upsample_flags = list(reversed(pool_flags))

        # mirror of encoder's pool_conv + latent_linear
        self.latent_linear = nn.Linear(latent_dim, compressed_seq_len)
        self.unpool_conv = nn.Conv1d(1, latent_dim, kernel_size=1, stride=1, padding=0)

        # halve channels and double sequence length at each layer
        self.conv_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(depth):
            in_channels = latent_dim if i == 0 else base_kernels * (2 ** (depth - i - 1))
            out_channels = base_kernels * (2 ** (depth - i - 2)) if i < depth - 1 else 1
            self.conv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(out_channels) if i < depth - 1 else nn.Identity(),
                )
            )
            self.upsample_layers.append(nn.Upsample(scale_factor=2))

    def forward(self, x):
        x = self.latent_linear(x)       # (B, compressed_seq_len)
        x = x.unsqueeze(1)              # (B, 1, compressed_seq_len)
        x = self.unpool_conv(x)         # (B, latent_dim, compressed_seq_len)
        # print(f"After unpool_conv: {x.shape}")
        for i, (conv, upsample, do_upsample) in enumerate(
            zip(self.conv_layers, self.upsample_layers, self.upsample_flags)
        ):
            if do_upsample:
                x = upsample(x)
            x = conv(x)
            # No activation on the final layer — the output must be able to
            # represent negative ECG signal amplitudes.
            if i < self.depth - 1:
                x = F.leaky_relu(x, 0.1)
            # print(f"After layer (upsample={do_upsample}): {x.shape}")

        return x[..., :self.seq_len].squeeze(1)  # crop padding introduced in encoder

class Autoencoder(nn.Module):
    def __init__(self, seq_len, depth, latent_dim, base_kernels: int = 16):
        super().__init__()
        self.encoder = Encoder(seq_len, depth, latent_dim, base_kernels)
        self.decoder = Decoder(seq_len, depth, latent_dim, base_kernels)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

if __name__ == "__main__":
    import torch
    seq_len = 625
    depth = 8
    latent_dim = 32
    encoder = Encoder(seq_len, depth, latent_dim)
    decoder = Decoder(seq_len, depth, latent_dim)

    x = torch.randn(1, seq_len) # (B, seq_len)
    z = encoder(x)
    x_recon = decoder(z)
    print(f"Input shape: {x.shape}, Latent shape: {z.shape}, Reconstructed shape: {x_recon.shape}")

    autoencoder = Autoencoder(seq_len, depth, latent_dim)
    x_recon_auto = autoencoder(x)
    print(f"Autoencoder reconstructed shape: {x_recon_auto.shape}")