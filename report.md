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

# Signal Autoencoder Training — April 5, 2026

## Overview

A full day of experimentation on the signal autoencoder covering architecture hyperparameter tuning and a systematic exploration of loss functions. The best checkpoint to date is `checkpoints/run_002/epoch=059-val_loss=0.0425.ckpt`, trained with `mae+mrstft` loss. All experiments used `seq_len=625`, `batch_size=256`, `lr_patience=5`, `lr_factor=0.5`, `weight_decay=1e-5`, and the ReduceLROnPlateau scheduler unless noted otherwise.

---

## Phase 1: Architecture Hyperparameter Sweep (MSE loss)

All runs in this phase used MSE as the training and validation loss. Loss values are directly comparable across runs.

| Version | depth | latent_dim | base_kernels | lr | Best val/loss (MSE) | Notes |
|---------|-------|-----------|--------------|-----|---------------------|-------|
| v0 | 8 | 32 | — | 1e-3 | 0.00582 | Initial baseline, no `base_kernels` param |
| v3–v5 | 8 | 32 | 16 | 1e-3 | 0.00630 | Introduced `base_kernels`; slightly worse than v0 |
| v6 | 8 | 32 | 32 | 1e-3 | 0.00531 | Wider channels help |
| v7 | 8 | 32 | 32 | 1e-2 | 0.01102 | LR 10× too high with this latent size; diverged |
| v8 | 8 | **64** | 32 | 1e-3 | **0.00276** | Doubling latent_dim gave the biggest single gain |
| v13 | **9** | 64 | 32 | 1e-3 | 0.00552 | Extra depth hurt slightly; more compression than signal complexity warrants |
| v9–v12 | 9–10 | 64 | 32 | 1e-3 | — | Aborted (small batch sizes 4–16 caused instability / very slow) |

**Key findings:**
- `latent_dim=64` was the most impactful change (0.00582 → 0.00276).
- `base_kernels=32` was better than 16.
- `depth=8` beat `depth=9` — adding more pooling stages compresses the 625-sample signal into just 3 timesteps at the bottleneck, which appears to be over-aggressive.
- `lr=1e-3` with the scheduler consistently outperformed `lr=1e-2` for MSE training.

---

## Phase 2: Loss Function Exploration

After settling on `depth=8, latent_dim=64, base_kernels=32` as the architecture, loss functions were explored systematically. MSE is a well-understood baseline but penalises large errors quadratically, which tends to produce over-smoothed reconstructions — it discourages the model from reconstructing sharp QRS spikes if it risks a large penalty.

### Infrastructure added

A `loss` string parameter was added to `SignalAutoencoderModule`, mapping to one of several loss functions. An `fft_alpha` parameter controls the blend weight for combination losses. Both are exposed in the YAML config. The validation step always logs `val/mae` and `val/mse` alongside `val/loss` regardless of which loss is being optimised, making runs cross-comparable on a common metric.

### MAE (v15)

| loss | fft_alpha | Best val/loss | val/mse (approx) |
|------|-----------|--------------|-----------------|
| mae | — | 0.0236 | — |

The core motivation for switching to MAE was peak fidelity. MSE penalises errors quadratically — a 2× larger error contributes 4× the loss. In practice this makes the model risk-averse: it prefers a moderately wrong prediction everywhere over a sharp spike that might be slightly off, because the spike contributes disproportionately to the gradient. For ECG signals this is a problem precisely where accuracy matters most — the QRS complex is a fast, high-amplitude event and the MSE-trained model consistently under-estimated its amplitude, producing blurry peaks.

MAE (`F.l1_loss`) treats all errors linearly, so a large error at a single timestep is penalised in proportion to its size rather than its square. The model is no longer incentivised to smooth out peaks to reduce variance in the loss. Visually, reconstructions showed noticeably sharper QRS complexes compared to any MSE run. The loss value is not comparable to MSE runs — they are on different absolute scales.

### Pure FFT magnitude loss (v16–v17)

Training on the magnitude spectrum of the full-signal FFT (`torch.fft.rfft`) alone.

The reasoning for adding a frequency-domain term is that waveform morphology — the characteristic shape of P, QRS, and T waves — is encoded in the relative magnitudes and relationships of frequency components. A model trained only on time-domain MAE can reconstruct individual sample values accurately but still produce a signal that looks morphologically wrong if the spectral balance is off (e.g., too much high-frequency noise, or missing the smooth baseline). The FFT loss penalises mismatches in the frequency content of the whole signal, encouraging the model to reproduce the correct spectral envelope and thereby improve the shape of individual waves.

- **v16** crashed immediately with `RuntimeError: cuFFT only supports dimensions whose sizes are powers of two when computing in half precision`. `16-mixed` precision caused `rfft` to run in float16 on a 625-sample sequence. **Fix:** cast inputs to `float32` before calling `rfft`.
- **v17** (after fix): converged to `val/loss=0.765`. The FFT loss operates on a different scale (magnitudes of frequency bins, not signal amplitudes), so this number represents a different quantity. Trained on the spectrum alone, the model learned the correct frequency distribution but lost all temporal phase information — there is no constraint on *when* events occur, only on *what frequencies* are present. This is insufficient on its own for ECG reconstruction.

### MAE + FFT (v18–v21)

Combined loss: `(1 - alpha) * MAE + alpha * FFT_MAE`, giving the model simultaneous time-domain and frequency-domain supervision.

The reasoning for combining rather than choosing one: pure FFT loss improved waveform shape but removed the per-sample temporal constraint. Pure MAE improved peak amplitude but had no explicit spectral regularisation. Adding the MAE term back re-introduces point-wise time-domain accuracy — the model must reconstruct each sample correctly *and* get the spectral balance right. For periodic signals like ECG, where the waveform repeats with a consistent profile, this combination is particularly effective: the FFT term enforces the correct overall shape and spectral envelope, while MAE pins individual points to the right amplitudes and preserves the correct timing of events.

- **v18** (`alpha=0.5`): Crashed mid-training with `val/loss=nan`. Root cause: no window function was passed to `torch.stft`, causing a **rectangular window** to be applied. The rectangular window introduces severe spectral leakage — energy from each frequency bin bleeds into all others, producing large spurious magnitudes that inflated gradient norms and drove the loss to NaN. **Fix:** switched to Hann window (`torch.hann_window`), which tapers each frame smoothly to zero at its edges.
- **v19** (`alpha=0.5`, after fix): Stable training, `val/loss=0.345`.
- **v20** (`alpha=0.25`): `val/loss=0.0907`. The lighter frequency weighting gave the time-domain MAE more influence and produced significantly better results.
- **v21** (`alpha=0.5`): `val/loss=0.447`. More frequency weight consistently hurt.

`alpha=0.25` was clearly the sweet spot for `mae+fft`. The sensitivity to `alpha` confirms the intuition: the MAE term is doing the heavy lifting for temporal accuracy, and the FFT term is acting as a shape regulariser — useful, but should not dominate.

### Multi-Resolution STFT loss (v22–v23) — current best

The single-resolution FFT captures global frequency content but is blind to how frequency composition changes over time. An ECG signal is explicitly non-stationary: the QRS complex is a brief, high-frequency transient; the P and T waves are slower, smoother events; and the baseline is near-DC. A global FFT treats all of these as a single flat spectrum, so if the QRS is poorly reconstructed the model gets penalised in the high-frequency bins but cannot tell *where* in time the problem occurred. This limits its ability to drive improvements in localised morphology.

The Short-Time Fourier Transform (STFT) divides the signal into overlapping frames and computes the FFT of each frame, producing a time-frequency representation. The MRSTFT loss runs this at multiple `(n_fft, hop)` resolutions and averages the per-resolution magnitude MAEs. Small windows (fine resolution) see local transients like QRS spikes; large windows (coarse resolution) see global spectral structure like the P and T wave envelopes. Training against both simultaneously biases the model toward reconstructing the signal correctly at every temporal scale, addressing aperiodic and morphologically variable waveforms that the global FFT could not capture well.

Mixing MRSTFT with MAE plays the same role as in `mae+fft`: the spectral terms improve wave shape across scales, while the MAE term anchors the reconstruction to correct point-wise amplitudes in the time domain, preventing the model from drifting to spectrally plausible but temporally displaced waveforms.

Resolutions used: `(32, 8)`, `(64, 16)`, `(128, 32)`, `(256, 64)` — spanning fine-grained temporal detail to coarse spectral structure.

| Version | loss | fft_alpha | lr | Best val/loss |
|---------|------|-----------|-----|--------------|
| v22 | mae+mrstft | 0.5 | 1e-2 | 0.0779 |
| v23 | mae+mrstft | 0.5 | 1e-2 | **0.0425** ← current best |

`lr=1e-2` worked well here, likely because the MRSTFT loss landscape is smoother and better conditioned than MSE, allowing a larger learning rate. The same `lr=1e-2` had failed badly in the pure MSE phase (v7: 0.0110 vs 0.00276 for v8 at 1e-3).

Like the FFT loss, MRSTFT magnitude values are not on the same scale as MSE, so the `val/loss` numbers are not directly comparable to the MSE-phase results. Visual inspection of reconstructions via `inspect_val.py` confirmed that the MRSTFT-trained model produces the sharpest QRS complexes and best-preserved waveform morphology of all runs to date.

---

## Current Best Configuration

```yaml
model:
  seq_len: 625
  depth: 8
  latent_dim: 64
  base_kernels: 32
  lr: 1.0e-2
  weight_decay: 1.0e-5
  lr_patience: 5
  lr_factor: 0.5
  loss: mae+mrstft
  fft_alpha: 0.5
```

Checkpoint: `checkpoints/run_002/epoch=059-val_loss=0.0425.ckpt`

---

## Next Steps

- Explore `fft_alpha` values below 0.5 for `mae+mrstft` (as `alpha=0.25` was found to help for `mae+fft`).
- Consider adding a weighted combination of MRSTFT scales (currently all equal weight) to emphasise coarser or finer temporal resolution.
- Evaluate whether `latent_dim=128` improves reconstruction given that the bottleneck at `depth=8` produces only 3 timesteps for a 625-sample input.
- Begin Stage 2: freeze the trained decoder and train an image encoder to map ECG images to the same latent space.