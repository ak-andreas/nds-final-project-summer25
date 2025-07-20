# utils.py

import numpy as np
import pandas as pd
import numbers

from scipy.signal import savgol_filter, decimate
from scipy.ndimage import percentile_filter
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import pearsonr




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


# utils.py
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from oasis.functions import deconvolve

def estimate_noise_sigma(dff: np.ndarray, dt: float = None) -> np.ndarray:
    """
    Estimate per-cell noise level from high-frequency fluctuations.
    Uses median absolute deviation of the first difference.

    Parameters
    ----------
    dff : ndarray, shape (n_cells, n_time)
        ΔF/F traces.
    dt : float, optional
        If provided, divides sigma by sqrt(dt) to convert to per‑second units.

    Returns
    -------
    noise_sigma : ndarray, shape (n_cells,)
        Estimated noise standard deviation for each cell (per frame).
    """
    diffs = np.diff(dff, axis=1)
    mad = np.median(np.abs(diffs), axis=1)
    sigma = mad / (0.6745 * np.sqrt(2))
    if dt is not None:
        sigma = sigma / np.sqrt(dt)
    return sigma

def deconvolve_oasis(
    dff: np.ndarray,
    fs: float,
    penalty: float = 1.0,
    g_init: float = None,
    optimize_g: bool = True
) -> dict:
    """
    Run OASIS L1 deconvolution on each cell's ΔF/F trace.
    """
    n_cells, n_t = dff.shape
    c = np.zeros((n_cells, n_t), float)
    s = np.zeros((n_cells, n_t), float)
    b = np.zeros(n_cells, float)
    g = np.zeros(n_cells, float)

    for i in range(n_cells):
        trace = np.asarray(dff[i], dtype=float)
        kwargs = {'penalty': penalty, 'optimize_g': optimize_g, 'b_nonneg': True}
        if g_init is not None:
            kwargs['g'] = np.array([g_init], dtype=float)
        out = deconvolve(trace, **kwargs, sn=None)
        # unpack first four outputs
        ci, si, bi, gi = out[0], out[1], out[2], out[3]
        c[i], s[i], b[i], g[i] = ci, si, bi, gi

    return {'c': c, 's': s, 'b': b, 'g': g}

def reconvolve_calcium_kernel(s: np.ndarray, g: float, b: np.ndarray) -> np.ndarray:
    """
    Reconstruct a calcium trace from an inferred spike train under an AR(1) model.
    """
    T = len(s)
    c = np.zeros(T, dtype=float)
    c[0] = s[0]
    for t in range(1, T):
        c[t] = g * c[t-1] + s[t]
    return c + b

def sweep_oasis_params(
    dff_clean: np.ndarray,
    fs: float,
    noise_sigma: np.ndarray,
    params: Sequence[Tuple[float, float]],
    cells: Optional[Sequence[int]] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Random (or grid) search over OASIS parameters.
    Returns a DataFrame with columns ['cell','g','lam','r2','rate_hz'].
    """
    n_cells, T = dff_clean.shape
    if cells is None:
        cells = np.arange(n_cells)

    def _evaluate_combo(g: float, lam: float, cell: int) -> dict:
        y = dff_clean[cell]
        o = deconvolve_oasis(
            dff=y[None, :],
            fs=fs,
            penalty=lam,
            g_init=g,
            optimize_g=False
        )
        s_hat = o['s'][0]
        b_hat = o['b'][0]
        g_hat = o['g'][0]
        y_rec  = reconvolve_calcium_kernel(s_hat, g_hat, b_hat)
        r2     = r2_score(y, y_rec)
        rate_hz = s_hat.sum() * fs / T
        return dict(cell=cell, g=g_hat, lam=lam, r2=r2, rate_hz=rate_hz)

    jobs = [(g, lam, cell) for (g, lam) in params for cell in cells]
    records = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_combo)(g, lam, cell) for g, lam, cell in jobs
    )
    return pd.DataFrame.from_records(records)


def compute_fit_metrics(
    y_true: np.ndarray,
    s_hat:  np.ndarray,
    g_hat:  np.ndarray,
    b_hat:  np.ndarray,
    fs:     float
):
    """
    Given clean traces y_true, and OASIS outputs (s_hat,g_hat,b_hat),
    reconstruct c(t), then compute per-cell R², MSE, event_count, rate_hz.
    """
    n_cells, n_t = y_true.shape
    dt = 1.0 / fs

    # reconstruct each trace
    y_rec = np.zeros_like(y_true)
    for i in range(n_cells):
        y_rec[i] = reconvolve_calcium_kernel(s_hat[i], g_hat[i], b_hat[i])

    # residuals
    ss_res = np.sum((y_true - y_rec)**2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True))**2, axis=1)

    r2          = 1 - ss_res / ss_tot
    mse         = ss_res / n_t
    event_count = s_hat.sum(axis=1).astype(int)
    rate_hz     = event_count / (n_t * dt)

    return r2, mse, event_count, rate_hz




def map_sparse_to_stim(
    stim_table: np.ndarray,
    mask_sparse: np.ndarray,
    modal_duration: int = 7
) -> np.ndarray:
    """
    Map each sparse‑noise sample index to its corresponding stimulus index,
    filtering to only stimuli shown for exactly `modal_duration` full frames.
    """
    tbl = stim_table.astype(int)
    durations = tbl[:, 2] - tbl[:, 1]
    valid = np.where(durations == modal_duration)[0]

    sparse_idxs = np.flatnonzero(mask_sparse)
    stim_id = np.full(sparse_idxs.size, -1, dtype=int)

    for i in valid:
        s_full, e_full = tbl[i, 1], tbl[i, 2]
        mask = (sparse_idxs >= s_full) & (sparse_idxs < e_full)
        stim_id[mask] = i

    return stim_id


def compute_sta(
    stim: np.ndarray,
    stim_id: np.ndarray,
    spikes: np.ndarray,
    pre_frames: int,
    baseline: float = 127.0
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the spike-triggered average (STA) and receptive-field map,
    properly handling integer event counts per frame.
    """
    # center stimuli
    stim0 = stim.astype(float) - baseline
    H, W  = stim0.shape[1:]
    
    # accumulator
    sta_accum = np.zeros((pre_frames, H, W), dtype=float)
    count     = 0
    
    for t_idx, s in enumerate(spikes):
        n_ev = int(round(s))           # integer # of events this frame
        if n_ev <= 0 or t_idx < pre_frames:
            continue
        
        window = stim_id[t_idx - pre_frames : t_idx]
        valid  = window >= 0
        if not np.any(valid):
            continue
        
        imgs = stim0[window[valid]]    # (n_valid, H, W)
        # add this window n_ev times
        sta_accum += imgs.sum(axis=0) * n_ev
        count     += imgs.shape[0] * n_ev
    
    if count > 0:
        sta = sta_accum / count
    else:
        sta = sta_accum
    
    rf_map = sta.sum(axis=0)
    return sta, rf_map, count


# 2D Gaussian fitting 
import scipy.optimize as opt

def fit_2d_gaussian(
    rf_map: np.ndarray
) -> dict:
    """
    Fit a 2D Gaussian to the receptive-field map.

    Parameters
    ----------
    rf_map : ndarray, shape (H, W)
        The 2D RF map to fit.

    Returns
    -------
    params : dict
        Gaussian fit parameters: {'amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset'}
    """
    H, W = rf_map.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    total = rf_map.sum()
    if total != 0:
        x0_com = (X * rf_map).sum() / total
        y0_com = (Y * rf_map).sum() / total
    else:
        x0_com, y0_com = (W-1)/2, (H-1)/2

    def gauss(xy, amp, x0, y0, sx, sy, th, off):
        x, y = xy
        c, s = np.cos(th), np.sin(th)
        a =  (c**2)/(2*sx**2) + (s**2)/(2*sy**2)
        b = -(s*c)/(2*sx**2) + (s*c)/(2*sy**2)
        d =  (s**2)/(2*sx**2) + (c**2)/(2*sy**2)
        return off + amp * np.exp(- (a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + d*(y-y0)**2))

    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = rf_map.ravel()

    amp0   = float(rf_map.max() - rf_map.min())
    sigma0 = float(max(1.0, min(W, H)/6.0))
    p0     = [amp0, x0_com, y0_com, sigma0, sigma0, 0.0, float(rf_map.min())]

    lower = [-np.inf, 0,    0,    1,   1,   -np.pi/4, -np.inf]
    upper = [ np.inf, W-1,  H-1,  W/2, H/2, np.pi/4,  np.inf]

    try:
        popt, _ = curve_fit(gauss, xdata, ydata, p0=p0,
                            bounds=(lower, upper), maxfev=30000)
        amp, x0, y0, sx, sy, th, off = popt
    except:
        amp, x0, y0, sx, sy, th, off = amp0, x0_com, y0_com, sigma0, sigma0, 0.0, float(rf_map.min())

    # Clamp to valid pixel indices
    x0 = float(np.clip(x0, 0, W-1))
    y0 = float(np.clip(y0, 0, H-1))

    return {
        'amplitude': amp,
        'x0':        x0,
        'y0':        y0,
        'sigma_x':   sx,
        'sigma_y':   sy,
        'theta':     th,
        'offset':    off,
    }

def eval_gaussian(params: dict, shape: tuple[int,int]) -> np.ndarray:
    """
    Reconstruct the fitted 2D Gaussian on an (H,W) grid.
    """
    H, W = shape
    x = np.arange(W); y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    amp = params['amplitude']
    x0  = params['x0']
    y0  = params['y0']
    sx  = params['sigma_x']
    sy  = params['sigma_y']
    th  = params['theta']
    off = params['offset']

    c, s = np.cos(th), np.sin(th)
    a =  (c**2)/(2*sx**2) + (s**2)/(2*sy**2)
    b = -(s*c)/(2*sx**2) + (s*c)/(2*sy**2)
    d =  (s**2)/(2*sx**2) + (c**2)/(2*sy**2)

    # Single exp() term (no accidental square)
    return off + amp * np.exp(- (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + d*(Y-y0)**2))


def cross_validate_rf(
    spikes: np.ndarray,
    stim: np.ndarray,
    stim_id: np.ndarray,
    pre_frames: int,
    baseline: float = 127.0,
    fit_func = fit_2d_gaussian
) -> dict:
    """
    Perform odd/even cross-validation for one cell's spike train.

    Returns a dict with keys:
      'sta_odd', 'sta_even',
      'rf_odd', 'rf_even',
      'params_odd', 'params_even',
      'corr'
    """
    # Identify spike times
    spike_times = np.flatnonzero(spikes > 0)
    odd_times   = spike_times[::2]
    even_times  = spike_times[1::2]

    # Build binary trains
    n = spikes.size
    ev_odd  = np.zeros(n, dtype=spikes.dtype)
    ev_even = np.zeros(n, dtype=spikes.dtype)
    ev_odd[odd_times]   = 1
    ev_even[even_times] = 1

    # Compute STAs for odd and even halves
    sta_o, rf_o, _ = compute_sta(stim, stim_id, ev_odd,  pre_frames, baseline)
    sta_e, rf_e, _ = compute_sta(stim, stim_id, ev_even, pre_frames, baseline)

    # Fit Gaussians with error handling
    try:
        params_o = fit_func(rf_o)
    except Exception:
        params_o = None

    try:
        params_e = fit_func(rf_e)
    except Exception:
        params_e = None

    # Compute split‑half correlation if both maps exist
    if params_o is None or params_e is None:
        corr = np.nan
    else:
        mask = ~np.isnan(rf_o) & ~np.isnan(rf_e)
        if mask.sum() < 2:
            corr = np.nan
        else:
            corr, _ = pearsonr(rf_o[mask].ravel(), rf_e[mask].ravel())

    return {
        'sta_odd':   sta_o,
        'sta_even':  sta_e,
        'rf_odd':    rf_o,
        'rf_even':   rf_e,
        'params_odd':  params_o,
        'params_even': params_e,
        'corr':      corr,
    }


def summarize_cv_results(
    cv_results: dict,
    threshold: float = 0.9
) -> dict:
    """
    Summarize cross-validation correlation results across cells.

    Returns a dict with:
      'mean_corr', 'std_corr', 'nan_count', 'below_threshold', 'fraction_below', 'threshold'
    """
    corrs = np.array([r['corr'] for r in cv_results.values()], dtype=float)
    nan_count = np.isnan(corrs).sum()
    valid = np.isfinite(corrs)
    mean_corr = np.nanmean(corrs)
    std_corr  = np.nanstd(corrs)
    below      = np.sum((corrs < threshold) & valid)
    fraction   = below / np.sum(valid) if np.sum(valid) > 0 else np.nan
    return {
        'mean_corr': mean_corr,
        'std_corr': std_corr,
        'nan_count': nan_count,
        'below_threshold': int(below),
        'fraction_below': fraction,
        'threshold': threshold,
    }


def filter_cells_by_cv(
    cv_results: dict,
    min_corr: float = 0.9
) -> list[int]:
    """
    Return list of cell indices whose cross-validation corr >= min_corr.
    """
    good = []
    for cell, r in cv_results.items():
        corr = r.get('corr', np.nan)
        if np.isfinite(corr) and corr >= min_corr:
            good.append(cell)
    return good

def compute_ripley_k(
    points: np.ndarray,
    radii: np.ndarray,
    area: float,
    mode: str = 'none'
) -> np.ndarray:
    """
    Compute Ripley's K-function for a set of 2D points.

    Parameters
    ----------
    points : ndarray of shape (N,2)
        Coordinates of RF centers.
    radii : 1D array
        Radii at which to evaluate K(r).
    area : float
        Total area of observation window (e.g., H*W if rectangular).
    mode : str
        Edge correction: 'none' or 'isotropic'.

    Returns
    -------
    K : ndarray
        Ripley's K(r) values for each radius.
    """
    N = len(points)
    K = np.zeros_like(radii, dtype=float)
    dists = np.sqrt(((points[:,None,:] - points[None,:,:])**2).sum(2))

    for i, r in enumerate(radii):
        counts = (dists <= r).sum(1) - 1
        K[i] = area * counts.sum() / (N * (N-1))
    return K


def simulate_poisson_k(
    N: int,
    area_shape: tuple[int,int],
    radii: np.ndarray,
    n_sims: int = 200,
    mode: str = 'none'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate Poisson point patterns and compute mean & 95% CI of Ripley’s K.

    Parameters
    ----------
    N : int
        Number of points.
    area_shape : (H, W)
        Dimensions of the rectangular window in same units as `radii`.
    radii : ndarray
        Radii at which to evaluate K.
    n_sims : int
        Number of Monte Carlo simulations.
    mode : str
        Edge‐correction mode ('none' or 'isotropic').

    Returns
    -------
    mean : ndarray
    lower : ndarray
    upper : ndarray
    """
    H, W = area_shape
    area = H * W
    sims = np.zeros((n_sims, len(radii)))

    for i in range(n_sims):
        # Uniform CSR in the stimulus window
        pts = np.column_stack([
            np.random.uniform(0, W, N),
            np.random.uniform(0, H, N)
        ])
        sims[i] = compute_ripley_k(pts, radii, area, mode=mode)

    lower = np.percentile(sims, 2.5, axis=0)
    upper = np.percentile(sims, 97.5, axis=0)
    mean  = sims.mean(axis=0)
    return mean, lower, upper

def compute_nn_distances(
    points: np.ndarray
) -> np.ndarray:
    """
    Compute nearest‐neighbor distances for a set of 2D points.

    Parameters
    ----------
    points : ndarray of shape (N,2)
        Coordinates of RF centers.

    Returns
    -------
    dists : ndarray of shape (N,)
        Distance from each point to its nearest neighbor.
    """
    N = points.shape[0]
    diffs = points[:,None,:] - points[None,:,:]
    d2 = (diffs**2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min(axis=1))


def simulate_nn_null(
    N: int,
    area_shape: tuple[int,int],
    n_sims: int = 200
) -> np.ndarray:
    """
    Simulate nearest‐neighbor distances under CSR.

    Returns
    -------
    sims : ndarray of shape (n_sims, N)
        Each row is the nearest‐neighbor distances for one simulation.
    """
    H, W = area_shape
    sims = np.zeros((n_sims, N))
    for i in range(n_sims):
        pts = np.column_stack((
            np.random.uniform(0, W, N),
            np.random.uniform(0, H, N)
        ))
        sims[i] = compute_nn_distances(pts)
    return sims

def compute_k_inhomogeneous(points, radii, area, lambda_pts):
    """
    Inhomogeneous Ripley's K as in Baddeley et al. (2000):
      K_inhom(r) = (1/area) * sum_{i≠j} [ I(d_ij ≤ r) / (λ_i λ_j) ]
    """
    N = len(points)
    dists = np.sqrt(((points[:,None,:] - points[None,:,:])**2).sum(-1))
    K_inh = np.zeros_like(radii)
    
    for idx, r in enumerate(radii):
        mask = (dists <= r)
        # zero out self‐pairs
        np.fill_diagonal(mask, False)
        weights = 1.0 / (lambda_pts[:,None] * lambda_pts[None,:])
        K_inh[idx] = (weights[mask].sum()) / area
    return K_inh

def simulate_inhom_poisson_k(N, area_shape, radii, kde, n_sims=200):
    H, W = area_shape
    area = H * W
    sims = np.zeros((n_sims, len(radii)))
    
    # Draw N points from intensity ∝ kde(x,y)
    xi, yi = np.linspace(0, W, 200), np.linspace(0, H, 200)
    XX, YY = np.meshgrid(xi, yi)
    pdf    = kde(np.vstack([XX.ravel(), YY.ravel()]))
    pdf   /= pdf.sum()
    cdf    = np.cumsum(pdf)
    
    for i in range(n_sims):
        # sample pixel‐indices according to intensity
        us = np.random.rand(N)
        idxs = np.searchsorted(cdf, us)
        xs = XX.ravel()[idxs]
        ys = YY.ravel()[idxs]
        pts_sim = np.column_stack([xs, ys])
        sims[i] = compute_k_inhomogeneous(pts_sim, radii, area, kde(pts_sim.T))
    
    return sims

def compute_stc(
    stim: np.ndarray,
    stim_id: np.ndarray,
    spikes: np.ndarray,
    pre_frames: int,
    baseline: float = 127.0,
    n_components: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spike-triggered covariance (STC) with spatiotemporal windows.

    Returns (eigvals, eigvecs) for up to n_components of C_spike - C_all,
    where each vector is length pre_frames*H*W.
    If there are too few spikes, returns (empty array, empty array).
    """
    import numpy as np

    # center stimuli
    stim0 = stim.astype(float) - baseline
    H, W = stim0.shape[1:]
    D = pre_frames * H * W

    # collect spike-triggered windows
    X = []
    for t in range(pre_frames, len(spikes)):
        count = int(round(spikes[t]))
        if count <= 0:
            continue
        idxs = stim_id[t-pre_frames:t]
        if np.any(idxs < 0):
            continue
        window = stim0[idxs]            # (pre_frames, H, W)
        vec = window.reshape(D)         # flatten
        X.extend([vec] * count)         # replicate if multiple events

    # if not enough events, bail out
    if len(X) < 2:
        return np.array([]), np.empty((D, 0))

    X = np.vstack(X)                    # (n_events, D)
    X0 = X - X.mean(axis=0, keepdims=True)

    # spike-triggered covariance
    C_spike = np.cov(X0, rowvar=False)

    # build baseline covariance from all windows
    Y = []
    for t in range(pre_frames, len(stim_id)):
        idxs = stim_id[t-pre_frames:t]
        if np.any(idxs < 0):
            continue
        window = stim0[idxs]
        Y.append(window.reshape(D))
    if len(Y) < 2:
        return np.array([]), np.empty((D, 0))

    Y = np.vstack(Y)
    Y0 = Y - Y.mean(axis=0, keepdims=True)
    C_all = np.cov(Y0, rowvar=False)

    # difference covariance
    C_diff = C_spike - C_all

    # eigen‐decompose, pick top components
    eigvals, eigvecs = np.linalg.eigh(C_diff)
    order = np.argsort(eigvals)[::-1]
    take = min(n_components, eigvecs.shape[1])
    idx = order[:take]
    return eigvals[idx], eigvecs[:, idx]


