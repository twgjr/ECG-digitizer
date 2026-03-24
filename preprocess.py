"""
The purpose of this module is to store helper functions for transoforming the
base dataset into a format compatible with the upcoming model training.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import resample

ECG_DIR = os.path.join(os.getcwd(), 'data', 'ecg')
TRAIN_DIR = os.path.join(os.getcwd(), 'data', 'train')
TRAIN_CSV = os.path.join(os.getcwd(), 'data', 'train.csv')
METRICS_CSV = os.path.join(os.getcwd(), 'data', 'resample_metrics.csv')
TARGET_FS = 250

def open_sample(ecg_id: int):
    signal_csv_path = os.path.join(ECG_DIR, str(ecg_id), f'{ecg_id}.csv')
    sample_df = pd.read_csv(signal_csv_path)
    return sample_df

def save_sample(ecg_id: int, sample_df: pd.DataFrame):
    print(ecg_id)
    signal_csv_path = os.path.join(ECG_DIR, str(ecg_id), f'{ecg_id}.csv')
    sample_df.to_csv(signal_csv_path, index=False)
    return sample_df

def _lead_slices(df, n):
    """
    Return {col: (start, end)} for the valid row range of each lead.
    Slot index is determined by rounding (first_valid * 4 / n) so records
    whose length is not exactly divisible by 4 are handled correctly.
    """
    slices = {}
    for col in df.columns:
        if df[col].isna().all():
            continue
        if col == 'II':
            slices[col] = (0, n)
        else:
            first_valid = df[col].first_valid_index()
            last_valid  = df[col].last_valid_index()
            slices[col] = (first_valid, last_valid + 1)
    return slices


def _slot_index(first_valid, n):
    """Return which of the four 2.5s slots a lead belongs to, using rounding."""
    return round(first_valid * 4 / n)


def _compute_metrics(y_raw, y_res):
    """Round-trip FFT upsample y_res back to len(y_raw) and return (prd, rmse, r, snr)."""
    from scipy.stats import pearsonr
    y_reconstructed = resample(y_res, len(y_raw))
    error           = y_raw - y_reconstructed
    prd  = np.sqrt(np.sum(error ** 2) / np.sum(y_raw ** 2)) * 100
    rmse = np.sqrt(np.mean(error ** 2))
    r, _ = pearsonr(y_raw, y_reconstructed)
    noise_power = np.mean(error ** 2)
    snr  = 10 * np.log10(np.mean(y_raw ** 2) / noise_power) if noise_power > 0 else float('inf')
    return prd, rmse, r, snr


def _append_metrics(rows):
    """Append a list of metric dicts to METRICS_CSV, creating the file if needed."""
    new_df  = pd.DataFrame(rows)
    write_header = not os.path.exists(METRICS_CSV)
    new_df.to_csv(METRICS_CSV, mode='a', index=False, header=write_header, float_format='%.4f')


def resample_to_250hz(ecg_id: int, fs: int):
    """
    Read a raw signal CSV, resample every lead to TARGET_FS Hz using
    scipy's FFT-based resample, and write a new CSV named {ecg_id}_250Hz.csv.

    Layout: the 10-second recording is divided into four equal time slots.
    Lead II is a full-length rhythm strip spanning all four slots.
    Every other lead occupies exactly one slot; the remaining rows are NaN
    (masking, not missing data). Each lead's valid slice is resampled
    independently so the NaN mask is preserved in the output.
    """
    in_path  = os.path.join(TRAIN_DIR, str(ecg_id), f'{ecg_id}.csv')
    out_path = os.path.join(TRAIN_DIR, str(ecg_id), f'{ecg_id}_250Hz.csv')

    df = pd.read_csv(in_path)
    n_in          = len(df)
    n_out         = int(round(n_in * TARGET_FS / fs))
    slot_size_out = n_out // 4

    out_df       = pd.DataFrame(np.nan, index=range(n_out), columns=df.columns)
    in_slices    = _lead_slices(df, n_in)
    metric_rows  = []

    for col, (src_start, src_end) in in_slices.items():
        y_in      = df[col].iloc[src_start:src_end].values.astype(float)
        resampled = resample(y_in, slot_size_out if col != 'II' else n_out)

        if col == 'II':
            out_df[col] = resampled
            dst_start, dst_end = 0, n_out
        else:
            slot_idx  = _slot_index(src_start, n_in)
            dst_start = slot_idx * slot_size_out
            dst_end   = dst_start + slot_size_out
            out_df.iloc[dst_start:dst_end, out_df.columns.get_loc(col)] = resampled

        prd, rmse, r, snr = _compute_metrics(y_in, resampled)
        metric_rows.append({'id': ecg_id, 'fs_raw': fs, 'fs_resampled': TARGET_FS, 'lead': col,
                            'prd_pct': prd, 'rmse_mv': rmse, 'pearson_r': r, 'snr_db': snr})

    out_df.to_csv(out_path, index=False, float_format='%.3f')
    _append_metrics(metric_rows)
    print(f'{ecg_id}: {fs} Hz -> {TARGET_FS} Hz  ({n_in} -> {n_out} samples)  -> {out_path}')


def measure_resample_quality(ecg_id: int, fs: int):
    """
    Measure reconstruction quality of the 250Hz resampled signal by FFT-upsampling
    it back to the original fs and comparing with the raw signal per lead.

    Metrics per lead:
      PRD  — Percentage Root-mean-square Difference (lower is better, <9% is clinical grade)
      RMSE — Root Mean Square Error in mV
      r    — Pearson correlation coefficient (closer to 1 is better)
      SNR  — Signal-to-Noise Ratio in dB (higher is better)
    """
    raw_path = os.path.join(TRAIN_DIR, str(ecg_id), f'{ecg_id}.csv')
    res_path = os.path.join(TRAIN_DIR, str(ecg_id), f'{ecg_id}_250Hz.csv')

    raw = pd.read_csv(raw_path)
    res = pd.read_csv(res_path)

    n_raw = len(raw)
    n_res = len(res)

    print(f"\nECG {ecg_id}  |  {fs} Hz -> {TARGET_FS} Hz -> {fs} Hz (round-trip)")
    print(f"{'Lead':<6} {'PRD%':>8} {'RMSE(mV)':>10} {'r':>8} {'SNR(dB)':>9}")
    print("-" * 45)

    in_slices  = _lead_slices(raw, n_raw)
    out_slices = _lead_slices(res, n_res)

    for col in in_slices:
        src_start, src_end = in_slices[col]
        dst_start, dst_end = out_slices[col]
        y_raw = raw[col].iloc[src_start:src_end].values.astype(float)
        y_res = res[col].iloc[dst_start:dst_end].values.astype(float)
        prd, rmse, r, snr = _compute_metrics(y_raw, y_res)
        print(f"{col:<6} {prd:>8.2f} {rmse:>10.4f} {r:>8.4f} {snr:>9.2f}")


def plot_resample_comparison(ecg_id: int, fs: int):
    """
    Plot all 12 leads comparing the raw signal and the 250Hz resampled signal.
    Each lead is shown over its correct time window within the 10-second recording.
    """
    import matplotlib.pyplot as plt

    raw_path = os.path.join(TRAIN_DIR, str(ecg_id), f'{ecg_id}.csv')
    res_path = os.path.join(TRAIN_DIR, str(ecg_id), f'{ecg_id}_250Hz.csv')

    raw = pd.read_csv(raw_path)
    res = pd.read_csv(res_path)

    n_raw  = len(raw)
    n_res  = len(res)
    slot_size_raw = n_raw // 4
    slot_size_res = n_res // 4

    leads = raw.columns.tolist()
    fig, axes = plt.subplots(6, 2, figsize=(16, 18))
    fig.suptitle(f'ECG {ecg_id}  —  raw ({fs} Hz) vs resampled ({TARGET_FS} Hz)', fontsize=13)

    for idx, col in enumerate(leads):
        ax = axes[idx // 2, idx % 2]

        if col == 'II':
            t_raw = np.linspace(0, 10, n_raw)
            t_res = np.linspace(0, 10, n_res)
            y_raw = raw[col].values
            y_res = res[col].values
        else:
            first_valid = raw[col].first_valid_index()
            if first_valid is None:
                ax.set_title(col)
                ax.axis('off')
                continue
            slot_idx  = first_valid // slot_size_raw
            t_start   = slot_idx * 2.5
            t_end     = t_start + 2.5

            src_start = slot_idx * slot_size_raw
            src_end   = src_start + slot_size_raw
            dst_start = slot_idx * slot_size_res
            dst_end   = dst_start + slot_size_res

            t_raw = np.linspace(t_start, t_end, slot_size_raw)
            t_res = np.linspace(t_start, t_end, slot_size_res)
            y_raw = raw[col].iloc[src_start:src_end].values
            y_res = res[col].iloc[dst_start:dst_end].values

        ax.plot(t_raw, y_raw, label=f'raw ({fs} Hz)', alpha=0.6, linewidth=0.8)
        ax.plot(t_res, y_res, label=f'{TARGET_FS} Hz', alpha=0.8, linewidth=0.8, linestyle='--')
        ax.set_title(col)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('mV')
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    meta = pd.read_csv(TRAIN_CSV)
    for _, row in meta.iterrows():
        resample_to_250hz(int(row['id']), int(row['fs']))