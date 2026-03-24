# 1D CNN Autoencoder — Design Plan

## Overview

Two models, trained in two stages.

**Phase 1 — Pretrain the signal autoencoder (`AE625`) on raw digital ECG data.**  
Digital ECG recordings are everywhere and don't need any labels. Each recording gives us up to 15 channels, and each channel is a 2.5-second, 625-point signal — exactly one column of a standard ECG printout. We train `AE625` to compress and reconstruct these 625-point segments. This means every channel of every recording is a valid training sample, no paired images needed. The autoencoder squeezes each segment down to a 20-number latent that captures the shape and timing of the waveform.

**Phase 2 — Train the image encoder (`ECGImageEncoder`) to produce the same latents from image crops.**  
When we have a real scanned ECG, a YOLO detector finds each lead region on the page — 11 standard lead cells plus the Lead II rhythm strip (split into 4 equal windows) — giving us 15 image patches, one for each 2.5-second channel. We run these through `ECGImageEncoder`, a small 2D CNN that maps each image crop to the same 20-number latent space that `AE625` uses. We train it so its output matches what the frozen signal encoder would have produced for the real waveform. This is called a few different things depending on who you ask:

- **Knowledge distillation** — signal encoder is the teacher, image encoder is the student
- **Latent space alignment** / **embedding alignment** — nudging two different inputs to land in the same place in latent space
- **Cross-modal distillation** — transferring what one modality (signal) knows to another (image)
- **Representation matching** — just minimising the distance between the two embeddings

Since the `AE625` decoder is already trained and frozen, once the image encoder gets the latent right the decoder can immediately reconstruct a clean signal — we don't need to supervise the output signal directly.

---

**Input:** 625-point window (2500-pt / 10 s / 250 Hz signal, split into 4 quarters)  
**Latent:** sequence `[B, 20]` — temporal order preserved

---

## Architecture: `AE625`

**DownBlock:** `Conv1d(k=5, s=2)` → `Conv1d(k=3, s=1)` → ReLU  
**UpBlock:** `ConvTranspose1d(k=4, s=2)` → `Conv1d(k=3, s=1)` → ReLU

| Encoder | Length | | Decoder | Length |
|---------|--------|-|---------|--------|
| Input   | 625    | | Latent  | 20     |
| Block 1 | 312    | | Block 1 | 39     |
| Block 2 | 156    | | Block 2 | 78     |
| Block 3 | 78     | | Block 3 | 156    |
| Block 4 | 39     | | Block 4 | 312    |
| Block 5 | 19 → pool → 20 | | Block 5 | 625 |

Latent: `[B, 256, 20]` → `Conv1d(256→1)` → `[B, 20]`

```python
class AE625(nn.Module):
    def __init__(self, latent_len=20):
        super().__init__()
        self.enc1 = DownBlock(1, 16);   self.enc2 = DownBlock(16, 32)
        self.enc3 = DownBlock(32, 64);  self.enc4 = DownBlock(64, 128)
        self.enc5 = DownBlock(128, 256)
        self.to_fixed_len = nn.AdaptiveAvgPool1d(latent_len)
        self.channel_pool = nn.Conv1d(256, 1, 1)
        self.expand = nn.Conv1d(1, 256, 1)
        self.dec1 = UpBlock(256, 128);  self.dec2 = UpBlock(128, 64)
        self.dec3 = UpBlock(64, 32);    self.dec4 = UpBlock(32, 16)
        self.dec5 = UpBlock(16, 1)
        self.final = nn.Conv1d(1, 1, 3, padding=1)

    def encode(self, x):
        x = self.enc5(self.enc4(self.enc3(self.enc2(self.enc1(x)))))
        return self.channel_pool(self.to_fixed_len(x)).squeeze(1)

    def decode(self, z):
        x = self.expand(z.unsqueeze(1))
        x = self.dec5(self.dec4(self.dec3(self.dec2(self.dec1(x)))))
        return self.final(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
```

---

## Lead II (10-second rhythm strip)

Split 2500-pt strip into 4 × 625-pt segments, process each with `AE625`:

```python
def encode_lead_ii(model, signal, device="cpu"):
    x = torch.tensor(signal.reshape(4, 625), dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        recon, z = model(x)   # (4, 1, 625), (4, 20)
    return recon.squeeze(1).cpu().numpy(), z.cpu().numpy()
```

Reconstruct full strip: `recon.reshape(2500)`  
Latents: stack `(4, 20)` to preserve temporal order.

---

## Image-to-Signal Pipeline

**15 channels:** 11 standard lead crops + 4 Lead II strip segments (grid Lead II copy skipped).

### Stage 1 — YOLO
Detects 12 bboxes: 11 `lead_standard` + 1 `lead_ii_strip` → split ÷4.  
All 15 crops normalized to `(1, 64, 512)`.

### Stage 2 — `ECGImageEncoder` (2D CNN)

```python
class ECGImageEncoder(nn.Module):
    def __init__(self, latent_len=20):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,   16,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16,  32,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.pool    = nn.AdaptiveAvgPool2d((1, latent_len))
        self.ch_pool = nn.Conv1d(128, 1, 1)

    def forward(self, x):
        # x: (15, 1, H, W) → (15, 20)
        f = self.pool(self.cnn(x)).squeeze(2)   # (15, 128, 20)
        return self.ch_pool(f).squeeze(1)        # (15, 20)
```

### Stage 3 — Frozen `AE625` decoder

Decodes `(15, 20)` latents → `(15, 1, 625)` signals.  
Lead II: `signals[11:15].squeeze(1).reshape(2500)` → full 10-second strip.

### Training

| Phase | Trains | Frozen |
|-------|--------|--------|
| 1 — 1D pretraining | `AE625` | — |
| 2 — 2D distillation | `ECGImageEncoder` | `AE625` |
| 3 — Fine-tune (optional) | `ECGImageEncoder` + `AE625.decode` | `AE625.encode` |

Loss (phase 2): `MSE(ECGImageEncoder(crops), AE625.encode(signals).detach())`
