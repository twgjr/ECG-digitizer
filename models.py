import math
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, seq_len, depth, latent_dim): # 625, 4, 64
        super().__init__()
        self.seq_len = seq_len
        self.depth = depth
        padded_seq_len = math.ceil(seq_len / (2 ** depth)) * (2 ** depth)
        # double channels and halve sequence length at each layer
        self.conv_list = nn.ModuleList()
        for i in range(depth):
            in_channels = (2 ** i)
            out_channels = (2 ** (i+1))

            if i == depth - 1:
                out_channels = latent_dim
            self.conv_list.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.ReLU(),
                )
            )
        # collapse the channel feature maps into a single vector maintaining the sequence order
        self.pool_conv = nn.Conv1d(latent_dim, 1, kernel_size=1, stride=1, padding=0)
        # regardless of the out size of pool_conv, we want to map it to the latent dimension
        self.latent_linear = nn.Linear(padded_seq_len // (2 ** depth), latent_dim)

    def forward(self, x):
        pad = (2 ** self.depth - self.seq_len % (2 ** self.depth)) % (2 ** self.depth)
        x = F.pad(x, (0, pad))  # pad to next 2^depth multiple
        x = x.unsqueeze(1)  # (B, 1, padded_seq_len)
        for conv in self.conv_list:
            x = conv(x)
        x = self.pool_conv(x)
        x = x.view(x.size(0), -1)
        x = self.latent_linear(x)
        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, depth, latent_dim):
        super().__init__()
        self.seq_len = seq_len
        padded_seq_len = math.ceil(seq_len / (2 ** depth)) * (2 ** depth)
        # mirror of encoder's pool_conv + latent_linear
        self.latent_linear = nn.Linear(latent_dim, padded_seq_len // (2 ** depth))
        self.unpool_conv = nn.Conv1d(1, latent_dim, kernel_size=1, stride=1, padding=0)
        # halve channels and double sequence length at each layer
        self.conv_list = nn.ModuleList()
        for i in range(depth):
            in_channels = (latent_dim if i == 0 else (2 ** (depth - i)))
            out_channels = (2 ** (depth - i - 1))
            self.conv_list.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.Upsample(scale_factor=2),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        x = self.latent_linear(x)       # (B, seq_len // 2**depth)
        x = x.unsqueeze(1)              # (B, 1, seq_len // 2**depth)
        x = self.unpool_conv(x)         # (B, latent_dim, seq_len // 2**depth)
        for conv in self.conv_list:
            x = conv(x)
        return x[..., :self.seq_len].squeeze(1)  # crop padding introduced in encoder

class Autoencoder(nn.Module):
    def __init__(self, seq_len, depth, latent_dim):
        super().__init__()
        self.encoder = Encoder(seq_len, depth, latent_dim)
        self.decoder = Decoder(seq_len, depth, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

if __name__ == "__main__":
    import torch
    seq_len = 625
    depth = 4
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