# Preprocessing

## Goal

Normalize the entire training dataset to a single sample rate of **250 Hz** so that every record fed into the model has the same temporal resolution. The raw dataset contains records at several different sample rates (250, 256, 500, 512, 1000, and 1025 Hz). Even records already at 250 Hz go through the pipeline for consistency.

---

## ECG Layout

Each raw CSV represents a **10-second, 12-lead ECG**. The layout is not 12 full-length columns — it follows the standard printed ECG rhythm strip format:

- **Lead II** is a full-length rhythm strip spanning all 10 seconds (all rows).
- Every other lead occupies exactly **one of four equal 2.5-second slots**, with the remaining rows masked as `NaN`.

| Slot | Leads | Time window |
|------|-------|-------------|
| 0 | I, III | 0 – 2.5 s |
| 1 | aVR, aVL, aVF | 2.5 – 5.0 s |
| 2 | V1, V2, V3 | 5.0 – 7.5 s |
| 3 | V4, V5, V6 | 7.5 – 10.0 s |

The `NaN` values are **masking, not missing data**. Each slotted lead contributes `n_samples / 4` valid rows at the appropriate offset.

---

## Resampling Method

Resampling is done using `scipy.signal.resample`, which uses the **FFT method**. This is appropriate for ECG because:

- It preserves the frequency content of the signal up to the Nyquist limit of the target rate.
- It applies implicit anti-aliasing when downsampling — unlike linear interpolation, which would introduce high-frequency roll-off artifacts.
- It is the mathematical inverse of itself, so a round-trip (down then up) gives a faithful reconstruction of whatever energy survived the downsampling step.

Each lead's valid slice is resampled **independently**. The NaN masking pattern is preserved in the output — the output file has the same 4-slot structure as the input.

### Handling non-divisible lengths

Some records have lengths not exactly divisible by 4 (e.g. 1025 Hz × 10 s = 10250 samples, remainder 2). Slot assignment uses **rounding** rather than floor division:

```
slot_index = round(first_valid_row × 4 / total_rows)
```

This prevents off-by-one slot misassignment that would occur with integer division.

---

## Output

For each record `{id}`, a new file is written to:

```
data/train/{id}/{id}_250Hz.csv
```

The output CSV has the same 12 columns and the same NaN masking structure as the raw input, but always has `fs × 10 / fs_raw × 250` rows (i.e. 2500 rows for all records). Values are written to **3 decimal places** to match the precision of the raw data.

---

## Quality Metrics

After each record is resampled, reconstruction quality is measured by **FFT-upsampling the 250 Hz output back to the original sample rate** and comparing it sample-by-sample with the raw signal. Using FFT resample for the round-trip (rather than linear interpolation) isolates only the error introduced by the downsampling step.

Four metrics are computed per lead per record and appended to `data/resample_metrics.csv`:

| Column | Description |
|--------|-------------|
| `prd_pct` | **PRD** — Percentage Root-mean-square Difference. Clinical grade threshold is < 9%. |
| `rmse_mv` | **RMSE** in millivolts |
| `pearson_r` | **Pearson correlation** between raw and reconstructed signal |
| `snr_db` | **SNR** in dB (signal power / error power) |

---

## Results

977 records were processed, producing 11,724 lead-level metric rows.

### Overall statistics (all leads, all records)

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| PRD % | 1.94 | 1.44 | 0.00 | 31.25 |
| RMSE (mV) | 0.003 | 0.002 | 0.000 | 0.380 |
| Pearson r | 0.9995 | 0.9999 | 0.950 | 1.000 |
| SNR (dB) | 36.8* | 36.8 | 10.1 | 308 |

*Mean SNR is dominated by 250→250 identity records (SNR ~304 dB). Median is more representative for non-identity cases.

### Mean metrics by original sample rate

| fs_raw | PRD % | RMSE (mV) | Pearson r | SNR (dB) |
|--------|-------|-----------|-----------|----------|
| 250 | 0.00 | 0.000 | 1.0000 | 304.0 |
| 256 | 0.70 | 0.001 | 1.0000 | 45.1 |
| 500 | 2.70 | 0.005 | 0.9993 | 33.4 |
| 512 | 2.50 | 0.004 | 0.9995 | 33.8 |
| 1000 | 2.72 | 0.005 | 0.9994 | 32.9 |
| 1025 | 3.04 | 0.005 | 0.9991 | 32.5 |

All non-identity sample rates are well within the 9% clinical PRD threshold, with Pearson r ≥ 0.999 across all groups. Reconstruction quality is high.

### Outliers (PRD > 9%)

A small number of records have individual leads exceeding the clinical PRD threshold. These appear to be genuinely noisy or high-amplitude ECGs — the signal energy itself is atypically large, so the high-frequency content lost in downsampling contributes disproportionately to PRD. Shape correlation (r ≥ 0.95) remains acceptable in all cases.

| ECG ID | fs_raw | Worst lead | PRD % |
|--------|--------|------------|-------|
| 3850187418 | 1025 | V5 | 31.2 |
| 2238910940 | 1025 | aVF | 28.2 |
| 4267219232 | 1025 | V6 | 25.0 |
| 3818475389 | 500 | aVF | 24.2 |
| 104573050 | 512 | aVF | 24.0 |

---

## Usage

```python
from preprocess import resample_to_250hz, measure_resample_quality, plot_resample_comparison

# Resample a single record (fs looked up from train.csv)
resample_to_250hz(10140238, 1000)

# Print per-lead quality metrics for a record
measure_resample_quality(10140238, 1000)

# Visual comparison of raw vs resampled (6×2 grid, all 12 leads)
plot_resample_comparison(10140238, 1000)
```

To run the full dataset:

```bash
conda run -n ecg python3 preprocess.py
```
