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