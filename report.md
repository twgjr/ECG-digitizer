# Intro
* Started with Fall 2024 project based on the PhysioNet Challenge 2024. Since then it has finsihed, published and is in late submissions on Kaggle.
* The dataset is much more readily available that during the challenge where I had to generate synthetic ECG images. Now there is over 80GB of ECG images and signal data available for download from Kaggle.
* I have taken it as a challenge to turn this into a full production ready pipeline and efficient models.

# Plan
The approach I am taking is to use a multi-stage training process which is typically called 
I plan to train the model in three stages:
1. Train a signal autocoder to learn a good representation of the ECG signals without the images or labels. 1D to 1D CNN.
2. Freeze the decoder and train an image encoder to learn to map the ECG images to the same latent space as the signal autocoder. 2D to 1D CNN.
3. Unfreeze the decoder and train the whole thing end to end with the labels. 2D to 1D CNN with classification head.

# Data Management

## Restore data
If you tinker with the raw data and need to restore it (e.g. to retest the EDA notebook), use:
```bash
dvc pull data.dvc
```
Note: `dvc pull` alone will fail because `models/signal_autoencoder.ckpt` has no hash recorded (the training pipeline has not been run yet and there is no `dvc.lock`). Use `dvc pull data.dvc` to pull only the data.

## Save changes to data
After modifying dataset files, re-hash, push, and commit:
```bash
dvc add data        # re-hashes data/, updates data.dvc
dvc push            # uploads changed files to Azure
git add data.dvc && git commit -m "describe your changes"
```

## Verify remote is up to date
```bash
dvc status --cloud
```
If the output says `Data and pipelines are up to date.` the remote is in sync. Any files listed need to be pushed.

# EDA Summary

## Dataset overview
- 977 ECG samples in `data/train/`, each sample stored as a directory containing a signal CSV and one or more PNG images of the ECG printout.
- `data/train.csv` index has three columns: `ecg_id`, `fs` (sampling rate in Hz), `sig_len` (total samples for Lead II). No nulls in the index.

## Signal structure
- Each CSV has 12 leads. **Lead II** contains the full 10-second signal. All other leads contain only a **2.5-second segment**, displayed at different intervals across the 10s window — the remainder of those leads is `NaN`.
- This is consistent with standard 12-lead ECG printout layout and is expected, not a data quality issue.
- Treating the 2.5s segments as independent signals gives **977 × 15 = 14,655** effective training windows for the signal autoencoder (12 leads minus Lead II = 11 short segments, plus Lead II itself split into four 2.5s chunks, per sample; but practically the notebook computed 977 × 15 = ~14,655 usable windows).

## Data quality issues found and fixed
- **162 samples had off-by-one rounding errors** for the 1025 Hz sampling rate: some of their leads had 2563 samples instead of the expected 2562 (i.e., `2.5 × 1025 = 2562.5` rounds inconsistently).
- Fix applied in-place: the extra trailing sample in any affected lead was set to `NaN`, making all signals consistent.
- Re-validation confirmed **zero errors** after the fix. The corrected CSVs were saved back to disk and pushed to DVC remote.

## Sample rate and signal length analysis
- **6 distinct sampling rates** in the range [250, 1025] Hz.
- **6 distinct signal lengths** in the range [2500, 10250] samples.
- All recordings are exactly **10 seconds** in duration — signal length varies only as a function of sampling rate.

## Conclusion
The dataset is consistent, complete, and clean. Confident to proceed to preprocessing.

# Preprocessing Plan

The key challenge is handling the 6 different sampling rates before feeding signals into a single autoencoder. Four options were considered:

| Option | Approach | Trade-off |
|--------|----------|-----------|
| 1 (baseline) | **Downsample all to 250 Hz** | Simplest; loses information for higher-rate signals |
| 2 | **Upsample all to 1025 Hz** | Risks teaching the model to smooth interpolated points |
| 3 | **Separate model per sample rate** | Poor generalization; drastically fewer samples per model |
| 4 | **Chunked multi-vector encoding** (single adaptive model) | Most flexible; borrows from dense retrieval; most complex |

**Starting with Option 1 (downsample to 250 Hz) as the baseline.** This gives a single uniform input size (`2500` samples for 10s Lead II, `625` samples for 2.5s segments) and maximises simplicity for the first training run. Higher-quality preprocessing strategies can be evaluated once there is a working baseline to compare against.

# Signal Autoencoder Architecture Design

## Overview

The signal autoencoder is a 1D CNN bottleneck autoencoder. The encoder compresses a raw ECG signal into a fixed-size latent vector; the decoder reconstructs the signal from that vector. The latent vector is the intended interface point for the image encoder in Stage 2 of the multi-stage training plan.

## Architecture

**Encoder:**
- A stack of `depth` convolutional blocks, each doubling the number of channels and halving the sequence length via `MaxPool1d(stride=2)`.
- Channels grow as powers of 2 (1 → 2 → 4 → 8 → `latent_dim` at the final block).
- A `Conv1d(latent_dim, 1, kernel_size=1)` collapses the channel dimension back to 1, preserving sequence order.
- A `Linear` layer maps the flattened sequence to a fixed `latent_dim`-dimensional vector.

**Decoder:**
- Mirrors the encoder in reverse.
- A `Linear` layer expands the latent vector back to the compressed sequence length.
- A `Conv1d(1, latent_dim, kernel_size=1)` restores the channel dimension.
- A stack of `depth` transposed convolutional blocks, each halving channels and doubling sequence length via `Upsample(scale_factor=2)`.

## Key Design Decision: Handling Arbitrary Sequence Lengths

### The Problem

With `depth=4`, the encoder applies 4 consecutive `stride=2` pooling operations, so the input must be divisible by $2^4 = 16$. The baseline sequence length of 2500 is not divisible by 16:

$$2500 \div 16 = 156.25 \quad \Rightarrow \quad 156 \times 16 = 2496 \neq 2500$$

The encoder produces 156 compressed timesteps, the decoder reconstructs $156 \times 16 = 2496$ samples, leaving a 4-sample shortfall.

### Options Considered

**Non-learned interpolation (`F.interpolate`)** — originally used as a quick fix. Stretches 2496 → 2500 using fixed linear interpolation. Not learned, distorts high-frequency content, and was rejected.

**Design data to be a multiple of $2^{depth}$** — resampling to 2496 instead of 2500 works, but tightly couples preprocessing to the architectural hyperparameter `depth`. Changing depth later requires re-running the preprocessing pipeline.

**Linear layer at the output** — learned, but a large dense linear over thousands of samples partially defeats the purpose of using a CNN. Rejected.

**Pad input, crop output (chosen approach)** — the encoder zero-pads the input to the next multiple of $2^{depth}$ before the forward pass. The decoder reconstructs the padded length, then crops back to the original length. The network never sees the padded positions in the loss function. No interpolation, nothing non-learned, and no coupling between preprocessing and model depth.

```
Input (2500) → pad → (2512) → Encoder → Bottleneck (157) → Decoder → (2512) → crop → Output (2500)
```

### Implementation

`padded_seq_len` is computed once at `__init__` time and used to size the linear layers:

```python
padded_seq_len = math.ceil(seq_len / (2 ** depth)) * (2 ** depth)
```

At forward time, padding is computed dynamically from `self.seq_len` and `self.depth`:

```python
# Encoder.forward
pad = (2 ** self.depth - self.seq_len % (2 ** self.depth)) % (2 ** self.depth)
x = F.pad(x, (0, pad))

# Decoder.forward (after reconstruction)
return x[..., :self.seq_len]
```

### Why This Is Better Than U-Net Style Skip Connections

U-Net also solves this mismatch (via center-crop at each skip connection), but U-Net does not produce a single compact latent vector — the skip connections bypass the bottleneck, making the latent space non-compact. Since Stage 2 of this project requires a single latent vector as the bridge between the signal and image encoders, a pure bottleneck autoencoder without skip connections is the correct choice.

## Multi-Sample-Rate Compatibility

Because `seq_len` and `depth` are constructor parameters, separate models for different sample rates are straightforward:

```python
model_250hz  = Autoencoder(seq_len=2500, depth=4, latent_dim=64)  # 250 Hz, 10s
model_500hz  = Autoencoder(seq_len=5000, depth=4, latent_dim=64)  # 500 Hz, 10s
model_1000hz = Autoencoder(seq_len=10000, depth=4, latent_dim=64) # 1000 Hz, 10s
```

Each model handles its own geometry internally. The pad/crop logic in `forward` is computed at runtime from the stored `seq_len` and `depth`, so no preprocessing changes are needed when changing depth — the model adapts automatically.

**Note:** Models trained at different sample rates have different `latent_linear` sizes (since `padded_seq_len // 2^depth` differs) and their latent spaces are not directly comparable. Cross-rate latent compatibility would require an additional alignment step.

## Future Work: Rate-Conditioned Single Model

Rather than training separate models per sample rate, the autoencoder could be conditioned on sample rate as a context signal — one model, one latent space, across all rates. This is the same pattern used in diffusion models (timestep conditioning) and conditional generation (FiLM). Several approaches in increasing order of expressiveness:

**Scalar concatenation** — append the sample rate (normalized) to the latent vector before any downstream head. Simple but only influences the fully-connected layers, not the conv stack.

**Feature-wise Linear Modulation (FiLM)** — at each conv block, predict a scale $\gamma$ and shift $\beta$ from the sample rate embedding and apply:
$$y = \gamma(r) \cdot \text{conv}(x) + \beta(r)$$
This lets the sample rate modulate every layer's activations, including the conv stack. The model can learn that a QRS complex spans ~100 samples at 1000 Hz and ~25 samples at 250 Hz, and adjust its effective receptive field accordingly.

**Sinusoidal / learned rate embedding** — embed the sample rate into a vector (like a positional encoding) and add it to feature maps at each scale. Most expressive; closest to how diffusion models handle the timestep.

**Temporal scale normalization (physics-informed)** — instead of telling the model the sample rate, normalize time by segmenting on heartbeat cycles (requires upstream R-peak detection). The model always sees the same temporal structure regardless of rate — sidesteps the problem entirely.

This is particularly relevant for Stage 2: if the image encoder must map to the same latent space as the signal encoder, a rate-conditioned single latent space is far cleaner than maintaining one latent space per rate.