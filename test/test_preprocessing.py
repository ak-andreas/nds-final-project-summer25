# test_utils.py
# This file contains tests for the data processing and analysis functions in utils.py
# To run these tests, navigate to your project directory in the terminal and run the command: pytest

import pytest
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.linalg import svd

# --- Import the functions to be tested from your utils file ---
# Note: Ensure your utils.py file is in the same directory or accessible via PYTHONPATH
from utils import (
    calculate_sampling_rate,
    simple_deconvolution,
    bin_spikes_to_frames,
    prepare_stimulus_matrix,
    negloglike_lnp,
    deriv_negloglike_lnp,
    fit_spatiotemporal_rf,
    extract_all_spatial_kernels,
    extract_all_temporal_kernels,
)

# --- Test Fixtures: Reusable Test Data ---
# Using pytest fixtures makes it easy to generate and reuse synthetic data for tests.

@pytest.fixture
def synthetic_data():
    """
    Generates a set of synthetic data mimicking the project's data structure.
    This provides a known ground truth to test our functions against.
    """
    # Simulation parameters
    num_neurons = 5
    num_time_points = 2000
    fs = 100  # Sampling rate in Hz
    t_time = np.linspace(0, num_time_points / fs, num_time_points)
    tau = 0.5  # Calcium decay time constant in seconds
    gamma = 1 - (1 / (tau * fs))

    # Create a known, ground-truth spike train
    true_spikes = np.zeros((num_neurons, num_time_points))
    spike_indices = [200, 500, 1000, 1500]
    for i in range(num_neurons):
        true_spikes[i, np.array(spike_indices) + i * 50] = 1.0

    # Generate a synthetic calcium trace based on the AR(1) model
    # c(t) = gamma * c(t-1) + s(t)
    true_calcium = np.zeros_like(true_spikes)
    for t in range(1, num_time_points):
        true_calcium[:, t] = gamma * true_calcium[:, t - 1] + true_spikes[:, t]
    
    # Stimulus data
    stim_height, stim_width = 16, 28
    num_pixels = stim_height * stim_width
    num_frames = 500
    
    # Create a known spatio-temporal receptive field (STRF)
    # A simple Gaussian blob that peaks at lag 2
    true_spatial_rf = np.exp(-((np.arange(num_pixels) - num_pixels // 2)**2) / (2 * 10**2))
    true_temporal_kernel = np.array([0, 0.5, 1, 0.5, 0]) # Peaks at lag 2
    true_strf = np.outer(true_spatial_rf, true_temporal_kernel) # Shape (pixels, lags)

    return {
        "t_time": t_time,
        "fs": fs,
        "tau": tau,
        "true_spikes": true_spikes,
        "true_calcium": true_calcium,
        "stim_dims": (stim_height, stim_width),
        "num_pixels": num_pixels,
        "num_frames": num_frames,
        "true_strf": true_strf,
    }

# --- Test Functions ---

def test_calculate_sampling_rate():
    """
    Tests the calculate_sampling_rate function.
    Ensures that the sampling frequency (fs) is correctly computed from a time vector.
    """
    t_vec = np.array([0, 0.1, 0.2, 0.3]) # 10 Hz
    fs = calculate_sampling_rate(t_vec)
    assert np.isclose(fs, 10.0)

def test_simple_deconvolution(synthetic_data):
    """
    Tests the simple_deconvolution function.

    **Formalism Link:** This tests the inversion of the AR(1) model used to approximate
    calcium dynamics. We generate a known calcium trace `c(t)` from a true spike train `s(t)`
    using `c(t) = gamma * c(t-1) + s(t)`. The test then ensures that the deconvolution
    function can recover the original `s(t)` from `c(t)`.
    """
    calcium_trace = synthetic_data["true_calcium"][0, :]
    true_spikes = synthetic_data["true_spikes"][0, :]
    tau = synthetic_data["tau"]
    fs = synthetic_data["fs"]

    recovered_spikes = simple_deconvolution(calcium_trace, tau, fs)
    
    # Check if the recovered spikes are highly correlated with the true spikes
    correlation = np.corrcoef(recovered_spikes, true_spikes)[0, 1]
    assert correlation > 0.95

def test_bin_spikes_to_frames():
    """
    Tests the bin_spikes_to_frames function.

    **Formalism Link:** This function prepares the neural response data for the LNP model
    by converting a continuous spike train into discrete spike counts `c(t)` for each
    stimulus frame time bin.
    """
    inferred_spikes = np.array([[0, 1, 0.5, 0, 2, 0, 1.5]])
    t_filtered = np.array([0, 1, 2, 3, 4, 5, 6])
    stim_table = pd.DataFrame({'start_s': [0, 2, 4], 'end_s': [2, 4, 6]})

    binned = bin_spikes_to_frames(inferred_spikes, t_filtered, stim_table)
    
    expected_bins = np.array([[1.5, 2.5, 1.5]]) # [0+1], [0.5+0], [2+0] -> incorrect logic. Correct is [0+1], [0.5+0], [2+0+1.5]
    # Corrected logic:
    # Bin 1 (0 to 2): spikes at t=1 -> sum = 1.0
    # Bin 2 (2 to 4): spikes at t=2 -> sum = 0.5
    # Bin 3 (4 to 6): spikes at t=4, t=6 -> sum = 2.0 + 1.5 = 3.5
    expected_bins_corrected = np.array([[1.0, 0.5, 3.5]])
    
    assert np.allclose(binned, expected_bins_corrected)

def test_prepare_stimulus_matrix(synthetic_data):
    """
    Tests the prepare_stimulus_matrix function.

    **Formalism Link:** This function prepares the stimulus data `s(t)` for the LNP model
    by reshaping the 3D movie (frames, height, width) into a 2D design matrix
    (pixels, frames).
    """
    h, w = synthetic_data["stim_dims"]
    n_frames = synthetic_data["num_frames"]
    stim_movie = np.arange(n_frames * h * w).reshape(n_frames, h, w)
    
    flattened_stim, _, _ = prepare_stimulus_matrix(stim_movie)
    
    assert flattened_stim.shape == (h * w, n_frames)
    # Test if the first pixel of the second frame is correct
    assert flattened_stim[0, 1] == stim_movie[1, 0, 0]

def test_lnp_gradient(synthetic_data):
    """
    Tests that the analytical gradient (deriv_negloglike_lnp) matches the numerical
    gradient of the negative log-likelihood function (negloglike_lnp).

    **Formalism Link:** This is a crucial test for the Maximum Likelihood Estimation (MLE)
    procedure. It verifies that our derived gradient is correct, ensuring that the
    optimization algorithm (`scipy.optimize.minimize`) will move in the right direction
    to find the optimal receptive field `w`.
    """
    num_pixels = synthetic_data["num_pixels"]
    
    # Create simple stimulus and spike count data
    s_test = np.random.randn(num_pixels, 100)
    c_test = np.random.randint(0, 5, 100)
    
    # Pick a random point in the parameter space to check the gradient
    w_test = np.random.randn(num_pixels)
    
    # Use scipy's check_grad function
    error = check_grad(negloglike_lnp, deriv_negloglike_lnp, w_test, c_test, s_test)
    
    # The error should be very small if the gradient is correct
    assert error < 1e-5

def test_fit_and_extract_rf_components(synthetic_data):
    """
    Tests the end-to-end process of fitting an STRF and extracting its components via SVD.

    **Formalism Link:** This test simulates the entire analysis pipeline on a small scale.
    1. It generates synthetic data from a known "ground-truth" receptive field `w_true`.
    2. It then uses the MLE fitting functions (`fit_spatiotemporal_rf`) to estimate `w_est`.
    3. Finally, it uses SVD (`extract_..._kernels`) to decompose `w_est`.
    The test asserts that the estimated components are very similar to the ground-truth components.
    """
    # --- 1. Generate synthetic data from a known STRF ---
    true_strf = synthetic_data["true_strf"]
    num_pixels = synthetic_data["num_pixels"]
    num_frames = synthetic_data["num_frames"]
    lags = [0, 1, 2, 3, 4]
    
    stim = np.random.randn(num_pixels, num_frames)
    
    # Filter stimulus with the true STRF to get the generator signal
    generator = np.zeros(num_frames)
    for t in range(len(lags), num_frames):
        stim_history = stim[:, t - np.array(lags)]
        generator[t] = np.sum(true_strf * stim_history)
        
    # Apply nonlinearity and generate Poisson spikes
    firing_rate = np.exp(generator)
    spikes = np.random.poisson(firing_rate)

    # --- 2. Fit the STRF from the synthetic data ---
    estimated_strf = fit_spatiotemporal_rf(spikes, stim, lags)

    # --- 3. Extract components using SVD ---
    h, w = synthetic_data["stim_dims"]
    estimated_spatial_rfs = extract_all_spatial_kernels([estimated_strf], h, w)
    estimated_temporal_kernels = extract_all_temporal_kernels([estimated_strf])
    
    estimated_spatial = estimated_spatial_rfs[0]
    estimated_temporal = estimated_temporal_kernels[0]

    # --- 4. Compare estimated components to ground truth ---
    # We need to account for potential sign flips from SVD
    true_spatial_rf = synthetic_data["true_strf"][:, 2] # The peak of the temporal kernel
    true_temporal_kernel = synthetic_data["true_strf"][0, :]

    # Normalize for comparison
    true_spatial_rf /= np.linalg.norm(true_spatial_rf)
    estimated_spatial /= np.linalg.norm(estimated_spatial.ravel())
    true_temporal_kernel /= np.linalg.norm(true_temporal_kernel)
    estimated_temporal /= np.linalg.norm(estimated_temporal)
    
    # Check correlation, allowing for a sign flip
    spatial_corr = np.corrcoef(true_spatial_rf, estimated_spatial.ravel())[0, 1]
    temporal_corr = np.corrcoef(true_temporal_kernel, estimated_temporal)[0, 1]

    assert abs(spatial_corr) > 0.9
    assert abs(temporal_corr) > 0.9
