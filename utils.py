# utils.py

import numpy as np
import numbers

from scipy.signal import savgol_filter, decimate
from scipy.ndimage import percentile_filter
from sklearn.linear_model import LinearRegression


def load_data(paths):
    """
    Load all project data (all .npy) into a dict.
    paths: dict of {name: filepath}
    """
    data = {}
    for key, p in paths.items():
        data[key] = np.load(p, allow_pickle=True)
    return data

def isolate_sparse_epochs(data):
    """
    Return boolean mask over timepoints belonging to sparse-noise epochs.
    Works if stim_epoch_table is a (n_epochs,3) object array with one string column
    and two numeric columns (start_frame, end_frame).
    """
    t = data['t']
    mask = np.zeros_like(t, dtype=bool)
    epochs = data['stim_epoch_table']  # shape (n_epochs, 3)

    # detect which column is text
    first = epochs[0]
    name_col = next(i for i,x in enumerate(first) if isinstance(x, str))
    num_cols = [i for i,x in enumerate(first) if isinstance(x, numbers.Number)]
    if len(num_cols) != 2:
        raise ValueError(f"Expected 2 numeric cols, got {num_cols}")
    start_col, end_col = num_cols

    names  = epochs[:, name_col].astype(str)
    is_sp  = np.char.find(names, 'sparse') >= 0

    for row in epochs[is_sp]:
        start = int(row[start_col])
        end   = int(row[end_col])
        mask[start:end] = True

    return mask

def compute_qc_metrics(dff):
    """
    Compute per-cell ΔF/F variance and a simple SNR metric.
    Returns (variance array, snr array) of length n_cells.
    """
    var = np.var(dff, axis=1)
    p98 = np.percentile(dff, 98, axis=1)
    p02 = np.percentile(dff,  2, axis=1)
    noise_std = np.std(dff, axis=1)
    snr = (p98 - p02) / (noise_std + 1e-8)
    return var, snr

def isolate_sparse_epochs(data, target='locally_sparse_noise', offset=0):
    """
    Return a boolean mask marking only epochs named exactly `target`.
    Optionally skip the first `offset` frames of each block.
    """
    t      = data['t']
    mask   = np.zeros_like(t, dtype=bool)
    epochs = data['stim_epoch_table']  # plain (n_epochs,3) array

    # detect which column is the name vs. start/end
    first   = epochs[0]
    name_i  = next(i for i,x in enumerate(first) if isinstance(x, str))
    num_i   = [i for i,x in enumerate(first) if isinstance(x, numbers.Number)]
    start_i, end_i = num_i

    names = epochs[:, name_i].astype(str)
    sel   = names == target

    for row in epochs[sel]:
        s = int(row[start_i]) + offset
        e = int(row[end_i])
        mask[s:e] = True

    return mask


def estimate_neuropil_proxy(dff):
    """
    Simple proxy neuropil for each cell: mean of all other cells.
    Inputs:
      dff: (n_cells, n_time) ΔF/F traces
    Returns:
      neuropil: (n_cells, n_time)
    """
    total = np.sum(dff, axis=0, keepdims=True)
    neuropil = (total - dff) / (dff.shape[0] - 1)
    return neuropil

def regress_neuropil_robust(dff, neuropil, clamp=(0.5, 0.9)):
    """
    Perform per-cell robust regression of dff_i ~ ρ_i * neuropil_i + intercept,
    then subtract ρ_i * neuropil_i to obtain cleaned traces.

    Inputs:
      dff:      (n_cells, n_time) raw or sparse ΔF/F
      neuropil: (n_cells, n_time) neuropil proxy traces
      clamp:    (min, max) ρ value bounds

    Returns:
      cleaned:  (n_cells, n_time) neuropil-regressed traces
      coefs:    (n_cells,)    array of ρ_i values
    """
    from sklearn.linear_model import HuberRegressor

    n_cells = dff.shape[0]
    cleaned = np.zeros_like(dff)
    coefs   = np.zeros(n_cells)

    for i in range(n_cells):
        y = dff[i]
        X = neuropil[i][:, None]
        model = HuberRegressor().fit(X, y)
        rho = model.coef_[0]
        rho_clamped = np.clip(rho, clamp[0], clamp[1])
        cleaned[i] = y - rho_clamped * neuropil[i]
        coefs[i]   = rho_clamped

    return cleaned, coefs

def sliding_baseline(arr, t, window_sec=60, pct=10, mode="nearest"):
    """
    Compute and subtract a running-percentile baseline from each row of `arr`.

    Parameters
    ----------
    arr : array_like, shape (n_cells, n_time)
        The ΔF/F traces to baseline-correct.
    t : array_like, shape (n_time,)
        Time vector in seconds.
    window_sec : float
        Length of the sliding window in seconds.
    pct : float
        Percentile to use for baseline (e.g. 10).
    mode : str
        How to handle boundaries in scipy.ndimage.percentile_filter.

    Returns
    -------
    baseline : ndarray, shape (n_cells, n_time)
        The running baseline.
    corrected : ndarray, shape (n_cells, n_time)
        `arr - baseline`.
    win_frames : int
        Number of frames in the sliding window.
    """    

    dt = np.median(np.diff(t))
    win_frames = int(np.round(window_sec / dt))
    if win_frames % 2 == 0:
        win_frames += 1

    baseline = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        baseline[i] = percentile_filter(arr[i], percentile=pct, size=win_frames, mode=mode)

    corrected = arr - baseline
    return baseline, corrected, win_frames

def smooth_dff_savgol(dff: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """
    Apply a Savitzky–Golay filter to each cell’s ΔF/F trace.

    Parameters
    ----------
    dff : array, shape (n_cells, n_time)
        The neuropil‐regressed, drift‐corrected ΔF/F traces.
    window : int
        Length of filter window (must be odd).
    polyorder : int
        Order of the polynomial fit (must be < window).

    Returns
    -------
    dff_smooth : array, shape (n_cells, n_time)
        The smoothed traces.
    """
    # enforce odd window
    if window % 2 == 0:
        raise ValueError("window must be odd")
    # apply filter along time axis=1
    return savgol_filter(dff, window_length=window, polyorder=polyorder, axis=1)

def downsample_traces(dff, t, target_fs):
    """
    Down-sample ΔF/F traces (and their timebase) to a lower sampling rate.

    Parameters
    ----------
    dff : array_like, shape (n_cells, n_time)
        High‑rate ΔF/F traces.
    t : array_like, shape (n_time,)
        Time vector corresponding to the columns of dff (in seconds).
    target_fs : float
        Desired sampling rate in Hz (e.g., 10).

    Returns
    -------
    dff_ds : ndarray, shape (n_cells, n_time_ds)
        Down‑sampled traces.
    t_ds : ndarray, shape (n_time_ds,)
        Down‑sampled time vector.
    factor : int
        Integer decimation factor used.
    actual_fs : float
        Achieved sampling rate (≈ target_fs).
    """
    # estimate native rate
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    # integer factor to get as close as possible to target_fs
    factor = int(round(fs / float(target_fs)))
    if factor < 1:
        raise ValueError(f"target_fs ({target_fs} Hz) >= original fs ({fs:.1f} Hz)")
    # decimate (zero‑phase FIR)
    dff_ds = decimate(dff, q=factor, axis=1, ftype="fir", zero_phase=True)
    # down‑sample the time vector
    t_ds = t[::factor]
    actual_fs = fs / factor
    return dff_ds, t_ds, factor, actual_fs

