import numpy as np
import pandas as pd
import numbers
import numpy as np
import os
import datetime
import json
import logging

from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

from scipy.linalg import svd
from scipy.ndimage import center_of_mass

import os
import glob

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# load data
def load_data(path="."):
    def array2df(d, key, cols):
        d[key] = pd.DataFrame(d[key], columns=cols)

    data = np.load(path + "/dff_data_rf.npz", allow_pickle=True)
    data = dict(data)
    array2df(data, "stim_table", ["frame", "start", "end"])
    array2df(data, "stim_epoch_table", ["start", "end", "stimulus"])

    return data


def find_latest_file(directory, file_pattern="*.npz"):
    """
    Finds the most recently created file in a directory matching a pattern.

    Args:
        directory (str): The path to the directory to search.
        file_pattern (str): The pattern of files to look for (e.g., '*.npz').

    Returns:
        str: The full path to the most recent file.
    
    Raises:
        FileNotFoundError: If no files matching the pattern are found.
    """
    search_path = os.path.join(directory, file_pattern)
    list_of_files = glob.glob(search_path)
    
    if not list_of_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in '{directory}'")
    
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_npy_metadata(file_path):
    """
    Extracts metadata from a .npy file.

    Args:
        file_path (str): The path to the .npy file.

    Returns:
        dict: A dictionary containing metadata, or None if an error occurs.
    """
    metadata = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "extension": os.path.splitext(file_path)[1],
    }

    try:
        # File system metadata
        file_stats = os.stat(file_path)
        metadata["file_size_bytes"] = file_stats.st_size
        metadata["creation_time"] = datetime.datetime.fromtimestamp(
            file_stats.st_ctime
        ).isoformat()
        metadata["modification_time"] = datetime.datetime.fromtimestamp(
            file_stats.st_mtime
        ).isoformat()

        # NumPy array metadata
        # For simplicity, we load the array to inspect its properties.
        # This is robust across versions but can be memory-intensive for huge files.
        with open(file_path, "rb") as f:
            data = np.load(
                f, allow_pickle=True)  

        # Handle the case where the loaded data is a dictionary-like object from .npz
        if isinstance(data, np.lib.npyio.NpzFile) or isinstance(data, dict):
             # This block is for handling .npz files if they are passed accidentally
             # For this script, we assume .npy files, so we focus on the array properties
             # If it's a dict-like object from np.load, we can't get a single shape/dtype
             metadata["array_type"] = str(type(data))
             metadata["array_dtype"] = "N/A (dict-like from .npz)"
             metadata["array_shape"] = "N/A (dict-like from .npz)"
        elif hasattr(data, 'shape'): # Check if it's an array-like object
            metadata["array_type"] = str(type(data))
            metadata["array_dtype"] = str(data.dtype)
            metadata["array_shape"] = data.shape
            metadata["array_ndim"] = data.ndim
            metadata["array_itemsize_bytes"] = data.itemsize
            metadata["array_nbytes"] = data.nbytes  # Total bytes consumed by array data

            # Check for structured array fields
            if data.dtype.fields is not None:
                metadata["is_structured_array"] = True
                metadata["structured_array_fields"] = list(data.dtype.fields.keys())
            else:
                metadata["is_structured_array"] = False
                metadata["structured_array_fields"] = []
        else:
             # Fallback for other unexpected types
            metadata["array_type"] = str(type(data))


        return metadata

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def get_npy_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npy")]

def extract_and_save_metadata(npy_dir="./dff_data_rf", output_filename="metadata_summary.json", output_directory="./output"):
    """
    Extracts metadata from a list of .npy files and saves it to a JSON file.

    Args:
        npy_files (list): List of paths to .npy files.
        output_filename (str): Path to the output JSON file.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    npy_files = get_npy_files(npy_dir)
    if not npy_files:
        logging.info("No .npy files found in the specified directory.")
        return

    output_path = os.path.join(output_directory, output_filename)
    
    logging.info(f"Output will be saved to: {output_path}")
    
    all_metadata = []

    logging.info("--- Extracting Metadata ---")
    for npy_file in npy_files:
        # Check if the file exists before attempting to process
        if not os.path.exists(npy_file):
            logging.info(f"Warning: File not found, skipping: {npy_file}")
            continue
            
        meta = U.get_npy_metadata(npy_file)
        if meta:
            all_metadata.append(meta)
            # The detailed printout to the console has been removed as requested.

    # --- Save Metadata to File ---
    if all_metadata:
        logging.info(f"--- Saving collected metadata to {output_filename} ---")
        try:
            with open(output_path, "w") as f_out:
                json.dump(all_metadata, f_out, indent=4)
            logging.info("Successfully saved metadata.")
        except Exception as e:
            logging.error(f"Error saving metadata to file: {e}")
    else:
        logging.info("\nNo metadata was collected, so no file was saved.")

    logging.info("\n--- Checking for dummy files to clean up ---")
    dummy_files_to_check = [
        "dff_data.npy",
        "max_projection.npy",
        "toi_maskss.npy", # Note: original had a typo, checking for that specifically
        "running_speed.npy",
        "stim.npy", # Note: original had 'stim', which is a directory in the data
    ]
    for f in dummy_files_to_check:
        if os.path.exists(f):
            try:
                os.remove(f)
                logging.info(f"Removed dummy file: {f}")
            except OSError as e:
                logging.error(f"Error removing file {f}: {e}")
    logging.info("\n--- Metadata extraction and saving completed ---")
    return output_path



# --- Configuration for Logging ---
# It's good practice to set up a logger for your library
# This allows the user of the library to control the logging level and output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_max_projection(data, save_path=None):
    """
    Visualizes the maximum activity projection image from the data dictionary.

    Args:
        data (dict): The dictionary containing the dataset, expected to have a 'max_projection' key.
        save_path (str, optional): If provided, saves the figure to this path instead of showing it. Defaults to None.
    """
    logging.info("Attempting to visualize maximum activity projection.")
    
    # Safely get the max_projection image from the data dictionary
    max_projection = data.get("max_projection")
    if max_projection is None:
        logging.warning("'max_projection' not found in the provided data dictionary. Skipping plot.")
        return

    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(max_projection, cmap="gray")
        plt.title("Maximum Activity Projection")
        plt.colorbar(label="Pixel Intensity")
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.info(f"Maximum projection plot saved to {save_path}")
            plt.close() # Close the figure to prevent it from displaying in a notebook
        else:
            plt.show()
            
    except Exception as e:
        logging.error(f"An error occurred while plotting max projection: {e}")
        plt.close()


def plot_stim_frames(data, indices_to_display=None, num_frames=5, save_path=None):
    """
    Visualizes a selection of stimulus frames from the data dictionary.

    Args:
        data (dict): The dictionary containing the dataset, expected to have a 'stim' key.
        indices_to_display (list, optional): A list of specific frame indices to display. 
                                            If None, random frames will be chosen. Defaults to None.
        num_frames (int, optional): The number of random frames to display if indices_to_display is None. Defaults to 5.
        save_path (str, optional): If provided, saves the figure to this path instead of showing it. Defaults to None.
    """
    logging.info("Attempting to visualize stimulus frames.")
    
    # Safely get the stimulus frames from the data dictionary
    stim_frames = data.get("stim")
    if stim_frames is None:
        logging.warning("'stim' (stimulus frames) not found in the provided data dictionary. Skipping plot.")
        return

    logging.info(f"Total stimulus frames available: {stim_frames.shape[0]}")
    logging.info(f"Individual stimulus frame shape: {stim_frames.shape[1]}x{stim_frames.shape[2]}")

    # Determine which frames to display
    if indices_to_display is None:
        # Choose random unique indices if none are provided
        if stim_frames.shape[0] < num_frames:
            logging.warning(f"Requested {num_frames} frames, but only {stim_frames.shape[0]} are available. Displaying all.")
            indices_to_display = np.arange(stim_frames.shape[0])
        else:
            indices_to_display = np.random.choice(stim_frames.shape[0], num_frames, replace=False)
            indices_to_display.sort() # Sort for a more ordered display
    
    num_to_display = len(indices_to_display)

    try:
        plt.figure(figsize=(num_to_display * 2.5, 3))
        for i, idx in enumerate(indices_to_display):
            if idx < stim_frames.shape[0]:
                plt.subplot(1, num_to_display, i + 1)
                plt.imshow(stim_frames[idx], cmap="gray")
                plt.title(f"Stim Frame {idx}")
                plt.axis("off")
            else:
                logging.warning(f"Skipping index {idx} as it is out of bounds for stim_frames array.")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logging.info(f"Stimulus frames plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        logging.error(f"An error occurred while plotting stimulus frames: {e}")
        plt.close()



def plot_preprocessing_comparison(raw_trace, processed_trace, time_vector, neuron_id, save_path=None):
    """
    Plots a single neuron's activity trace before and after preprocessing on separate subplots.

    Args:
        raw_trace (np.ndarray): The 1D array of the raw dF/F signal.
        processed_trace (np.ndarray): The 1D array of the signal after preprocessing (e.g., filtering).
        time_vector (np.ndarray): The 1D array of time points corresponding to the traces.
        neuron_id (int or str): The identifier for the neuron being plotted.
        save_path (str, optional): If provided, saves the figure to this path. Defaults to None.
    """
    logging.info(f"Plotting preprocessing comparison for neuron {neuron_id}.")
    
    try:
        # Create a figure with two subplots, one for each trace, sharing the x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        fig.suptitle(f"Preprocessing Comparison for Neuron {neuron_id}", fontsize=16)

        # Plot raw trace on the first subplot
        ax1.plot(time_vector, raw_trace, label='Raw dF/F', color='gray', alpha=0.9)
        ax1.set_title("Raw Signal")
        ax1.set_ylabel("dF/F")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot processed trace on the second subplot
        ax2.plot(time_vector, processed_trace, label='Processed dF/F', color='cornflowerblue', linewidth=1.5)
        ax2.set_title("Processed Signal")
        ax2.set_ylabel("dF/F")
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.info(f"Preprocessing comparison plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        logging.error(f"An error occurred while plotting preprocessing comparison: {e}")
        plt.close()


def calculate_sampling_rate(time_vector):
    """Calculates the sampling rate from a time vector."""
    if len(time_vector) < 2:
        raise ValueError("Time vector must have at least two elements to calculate sampling rate.")
    # Calculate the average difference between consecutive time points
    fs = 1.0 / np.mean(np.diff(time_vector))
    return fs

def simple_deconvolution(c, tau, fs):
    """
    Performs simple deconvolution on a calcium trace based on a first-order
    autoregressive model: c_t = gamma * c_{t-1} + s_t.

    This function estimates the spikes 's_t' by inverting the model.

    Args:
        c (np.ndarray): A 1D array representing the calcium trace (dF/F).
        tau (float): The decay time constant of the calcium indicator in seconds.
                     For GCaMP6f, a value around 0.5-0.7 is reasonable.
        fs (float): The sampling frequency of the recording in Hz.

    Returns:
        np.ndarray: A 1D array representing the estimated spike train.
    """
    # 1. Calculate the autoregression coefficient 'gamma'
    gamma = 1 - (1 / (tau * fs))
    
    # 2. Estimate spikes by inverting the AR(1) model
    # s_t = c_t - gamma * c_{t-1}
    # We can do this efficiently for the whole array.
    # c[1:] corresponds to c_t for t > 0
    # c[:-1] corresponds to c_{t-1} for t > 0
    s = c[1:] - gamma * c[:-1]
    
    # 3. Enforce non-negativity constraint (spikes cannot be negative)
    s[s < 0] = 0
    
    # The resulting spike train is one element shorter than the calcium trace.
    # We can pad it with a zero at the beginning to match the original length.
    s = np.insert(s, 0, 0)
    
    return s

def plot_inference_comparison(dff_trace, inferred_spikes, time_vector, neuron_id, save_path=None):
    """
    Plots the dF/F trace and the inferred spikes for a single neuron.

    Args:
        dff_trace (np.ndarray): The preprocessed dF/F signal.
        inferred_spikes (np.ndarray): The inferred spike train from deconvolution.
        time_vector (np.ndarray): The corresponding time points.
        neuron_id (int or str): The identifier for the neuron.
        save_path (str, optional): Path to save the figure. Defaults to None.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1]})
    
    fig.suptitle(f"Spike Inference for Neuron {neuron_id}", fontsize=16)

    # Plot dF/F trace
    ax1.plot(time_vector, dff_trace, color='cornflowerblue', label='dF/F Trace')
    ax1.set_title("Preprocessed Calcium Signal")
    ax1.set_ylabel("dF/F")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Plot inferred spikes
    ax2.plot(time_vector, inferred_spikes, color='coral', label='Inferred Spikes')
    ax2.set_title("Inferred Neural Activity")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Activity (a.u.)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
        
        
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.special import gammaln

def negloglike_lnp(w, c, s, dt=1.0, R=1.0):
    """Implements the negative log-likelihood of the LNP model."""
    w = w.ravel()
    lin = w @ s
    rates = np.exp(lin) * dt * R
    
    logL = np.dot(c, lin) - rates.sum() - gammaln(c + 1).sum()
    return -logL

def deriv_negloglike_lnp(w, c, s, dt=1.0, R=1.0):
    """Implements the gradient of the negative log-likelihood."""
    w = w.ravel()
    lin = w @ s
    rates = np.exp(lin) * dt * R
    
    grad = s @ (rates - c)
    return grad

def bin_spikes_to_frames_from_active_stim_per_timestep(inferred_spikes, active_stim_per_timestep):
    """
    Bins inferred spikes using a timestep-to-frame_id mapping array,
    preserving the temporal order of stimulus presentations.

    Args:
        inferred_spikes (np.ndarray): Shape (num_neurons, num_timesteps).
        active_stim_per_timestep (np.ndarray): Shape (num_timesteps,). Maps each
                                               timestep to a frame ID (-1 for inactive).

    Returns:
        tuple: A tuple containing:
            - binned_spikes (np.ndarray): The binned spikes, shape (num_neurons, num_presentations).
            - presented_frame_ids (np.ndarray): The frame IDs for each presentation, in order.
    """
    print("Binning spikes using timestep-to-frame mapping array...")
    num_neurons, num_timesteps = inferred_spikes.shape
    
    # --- Step 1: Identify the start of each stimulus presentation ---
    # A presentation starts when the active_stim ID changes from the previous
    # timestep, and the new ID is not -1 (inactive).
    
    # Find all points where the stimulus ID changes
    is_different_from_prev = np.concatenate(([True], np.diff(active_stim_per_timestep) != 0))
    
    # Filter for only the active stimulus timesteps
    is_stim_active = active_stim_per_timestep != -1
    
    # The start of a presentation is where a stimulus is active AND it's a new frame
    presentation_start_indices = np.where(is_stim_active & is_different_from_prev)[0]
    
    if len(presentation_start_indices) == 0:
        print("Warning: No active stimulus frames found in the mapping array.")
        return np.array([[]] * num_neurons), np.array([])

    # --- Step 2: Get the corresponding frame IDs and bin the spikes ---
    
    # The frame ID for each presentation is the value at its start index
    presented_frame_ids = active_stim_per_timestep[presentation_start_indices]
    
    # The end index of a presentation is the start index of the next one
    presentation_end_indices = np.concatenate((presentation_start_indices[1:], [num_timesteps]))
    
    num_presentations = len(presentation_start_indices)
    binned_spikes = np.zeros((num_neurons, num_presentations))
    
    # Sum the spikes within each presentation window
    for i in range(num_presentations):
        start = presentation_start_indices[i]
        end = presentation_end_indices[i]
        
        binned_spikes[:, i] = np.sum(inferred_spikes[:, start:end], axis=1)
        
    print(f"Binning complete. Binned spikes shape: {binned_spikes.shape}")
    return binned_spikes, presented_frame_ids



def prepare_stimulus_matrix_from_ids(presented_frame_ids, full_stim_movie):
    """
    Creates a flattened stimulus matrix from a list of frame IDs, preserving
    the temporal order of presentation.

    Args:
        presented_frame_ids (np.ndarray): A 1D array containing the sequence of
                                          frame IDs that were presented.
        full_stim_movie (np.ndarray): The complete stimulus movie, shape
                                      (total_unique_frames, height, width).

    Returns:
        tuple: A tuple containing:
            - flattened_stim (np.ndarray): The design matrix, shape (num_pixels, num_presentations).
            - stim_h (int): The height of the stimulus frames.
            - stim_w (int): The width of the stimulus frames.
    """
    print("Preparing stimulus matrix from frame IDs...")

    # --- Step 1: Select the presented frames in the correct order ---
    # This is a simple but powerful NumPy indexing operation.
    presented_frames = full_stim_movie[presented_frame_ids]
    
    # --- Step 2: Get dimensions ---
    num_presentations, stim_h, stim_w = presented_frames.shape
    num_pixels = stim_h * stim_w
    
    # --- Step 3: Flatten and transpose the matrix ---
    # We reshape to (num_presentations, num_pixels) and then transpose to get
    # the final shape (num_pixels, num_presentations) for the model.
    flattened_stim = presented_frames.reshape(num_presentations, num_pixels).T
    
    print(f"Stimulus matrix preparation complete. Shape: {flattened_stim.shape}")
    return flattened_stim, stim_h, stim_w






def bin_spikes_to_frames(inferred_spikes, t_filtered, stim_table_filtered_df):
    """Bins continuous inferred spike data into counts per stimulus frame."""
    print("Step 1: Binning inferred spikes...")
    num_neurons = inferred_spikes.shape[0]
    num_stim_frames = len(stim_table_filtered_df)
    
    bin_edges = np.concatenate([stim_table_filtered_df['start_s'].values, 
                                [stim_table_filtered_df['end_s'].values[-1]]])
    
    binned_spikes = np.zeros((num_neurons, num_stim_frames))
    for i in range(num_neurons):
        counts, _ = np.histogram(t_filtered, bins=bin_edges, weights=inferred_spikes[i, :])
        binned_spikes[i, :] = counts
        
    print(f"Binned spikes shape: {binned_spikes.shape}")
    return binned_spikes

def prepare_stimulus_matrix(stim_filtered):
    """Flattens the stimulus frames into a 2D matrix."""
    print("\nStep 2: Preparing stimulus matrix...")
    num_stim_frames, stim_height, stim_width = stim_filtered.shape
    num_pixels = stim_height * stim_width
    
    flattened_stim = stim_filtered.reshape(num_stim_frames, num_pixels).T
    
    print(f"Flattened stimulus shape: {flattened_stim.shape}")
    return flattened_stim, stim_height, stim_width

def fit_spatiotemporal_rf(neuron_spikes, flattened_stim, delta):
    """Fits a spatio-temporal receptive field for a single neuron."""
    num_pixels, _ = flattened_stim.shape
    num_lags = len(delta)
    w_hat_neuron = np.zeros((num_pixels, num_lags))

    for j, lag in enumerate(delta):
        if lag > 0:
            S_lag = flattened_stim[:, :-lag]
            C_lag = neuron_spikes[lag:]
        else:
            S_lag = flattened_stim
            C_lag = neuron_spikes
            
        w0 = np.zeros(num_pixels)
        res = opt.minimize(
            fun=lambda w: negloglike_lnp(w, C_lag, S_lag),
            x0=w0,
            jac=lambda w: deriv_negloglike_lnp(w, C_lag, S_lag),
            method="L-BFGS-B",
            options={'disp': False}
        )
        w_hat_neuron[:, j] = res.x
        
    return w_hat_neuron

def fit_all_neurons_rfs(binned_spikes, flattened_stim, delta, selected_neurons=None):
    """Fits receptive fields for all neurons."""
    print("\nStep 3 & 4: Fitting spatio-temporal receptive fields for all neurons...")
    num_neurons = binned_spikes.shape[0]
    all_neuron_rfs = []
    if selected_neurons is None:
        selected_neurons = range(num_neurons)
    for i in selected_neurons:
        print(f"  Fitting Neuron {i+1}/{num_neurons}...")
        neuron_spikes = binned_spikes[i, :]
        w_hat = fit_spatiotemporal_rf(neuron_spikes, flattened_stim, delta)
        all_neuron_rfs.append(w_hat)
        
    print("Fitting complete.")
    return all_neuron_rfs

def extract_spatial_rfs_svd(all_neuron_rfs, stim_height, stim_width):
    """Separates spatial and temporal components using SVD."""
    print("\nStep 5: Separating spatial and temporal components using SVD...")
    all_spatial_rfs = []

    for w_hat in all_neuron_rfs:
        w_centered = w_hat - w_hat.mean(axis=1, keepdims=True)
        U, _, _ = np.linalg.svd(w_centered, full_matrices=False)
        spatial_component = U[:, 0].reshape(stim_height, stim_width)
        all_spatial_rfs.append(spatial_component)
        
    print(f"SVD analysis complete. Extracted {len(all_spatial_rfs)} spatial receptive fields.")
    return all_spatial_rfs



def visualize_neuron_strf_details(neuron_id, 
                                    spatiotemporal_rf, 
                                    delta, 
                                    stim_dims):
    """
    Creates a detailed visualization for a single neuron, showing the full
    spatio-temporal receptive field (STRF) across all lags and its SVD decomposition.

    Args:
        neuron_id (int or str): The identifier for the neuron.
        spatiotemporal_rf (np.ndarray): The (num_pixels, num_lags) STRF.
        delta (list or np.ndarray): The time lags used in the analysis.
        stim_dims (tuple): The (height, width) of the stimulus.
    """
    print(f"\nGenerating detailed STRF visualization for Neuron {neuron_id}...")
    
    num_lags = len(delta)
    stim_h, stim_w = stim_dims

    # --- Create the figure with two rows of subplots ---
    fig = plt.figure(figsize=(num_lags * 2.5, 6))
    gs = fig.add_gridspec(2, num_lags)
    fig.suptitle(f"Detailed Analysis for Neuron {neuron_id}", fontsize=16)

    # --- Top Row: Full Spatio-Temporal Receptive Field (STRF) ---
    # Find a common color scale for all lags in the top row
    vlim_strf = np.max(np.abs(spatiotemporal_rf))
    
    for i, lag in enumerate(delta):
        ax = fig.add_subplot(gs[0, i])
        rf_at_lag = spatiotemporal_rf[:, i].reshape(stim_h, stim_w)
        ax.imshow(rf_at_lag, cmap='bwr', vmin=-vlim_strf, vmax=vlim_strf)
        ax.set_title(f"STRF @ Lag {lag}")
        ax.axis('off')

    # --- Bottom Row: SVD Decomposition ---
    
    # Perform SVD to get the separated components
    w_centered = spatiotemporal_rf - spatiotemporal_rf.mean(axis=1, keepdims=True)
    U, _, Vt = svd(w_centered, full_matrices=False)
    
    spatial_rf = U[:, 0].reshape(stim_h, stim_w)
    temporal_rf = Vt[0, :]
    
    # Sign correction for consistent plotting
    if np.abs(np.min(spatial_rf)) > np.abs(np.max(spatial_rf)):
        spatial_rf *= -1
        temporal_rf *= -1

    # Plot Separated Spatial RF (takes up first half of bottom row)
    ax_spatial = fig.add_subplot(gs[1, 0:num_lags//2])
    vlim_spatial = np.max(np.abs(spatial_rf))
    ax_spatial.imshow(spatial_rf, cmap='bwr', vmin=-vlim_spatial, vmax=vlim_spatial)
    ax_spatial.set_title("Separated Spatial RF")
    ax_spatial.axis('off')

    # Plot Separated Temporal Kernel (takes up second half of bottom row)
    ax_temporal = fig.add_subplot(gs[1, num_lags//2:])
    ax_temporal.plot(delta, temporal_rf, 'o-')
    ax_temporal.axhline(0, color='grey', linestyle='--')
    ax_temporal.set_title("Separated Temporal Kernel")
    ax_temporal.set_xlabel("Lag (frames)")
    ax_temporal.set_ylabel("Weight")
    ax_temporal.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    
def visualize_neuron_rf(neuron_id, spatiotemporal_rf, spatial_rf, delta):
    """Visualizes the analysis results for a single neuron."""
    print(f"\nVisualizing results for an example neuron (Neuron {neuron_id})...")
    
    # Perform SVD to get the temporal component for plotting
    w_centered = spatiotemporal_rf - spatiotemporal_rf.mean(axis=1, keepdims=True)
    _, _, Vt_ex = np.linalg.svd(w_centered, full_matrices=False)
    temporal_rf = Vt_ex[0, :]
    
    # Correct sign for consistent visualization
    if np.abs(np.min(spatial_rf)) > np.abs(np.max(spatial_rf)):
        spatial_rf *= -1
        temporal_rf *= -1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Example Analysis for Neuron {neuron_id}", fontsize=16)

    stim_height, stim_width = spatial_rf.shape
    
    # 1. Spatio-temporal RF at max energy lag
    lag_max_energy = np.argmax(np.sum(spatiotemporal_rf**2, axis=0))
    vlim = np.max(np.abs(spatiotemporal_rf[:, lag_max_energy]))
    axes[0].imshow(spatiotemporal_rf[:, lag_max_energy].reshape(stim_height, stim_width), 
                   cmap='bwr', vmin=-vlim, vmax=vlim)
    axes[0].set_title(f"Spatio-temporal RF (Lag {lag_max_energy})")
    axes[0].axis('off')

    # 2. Separated spatial component
    vlim_spatial = np.max(np.abs(spatial_rf))
    axes[1].imshow(spatial_rf, cmap='bwr', vmin=-vlim_spatial, vmax=vlim_spatial)
    axes[1].set_title("Separated Spatial RF")
    axes[1].axis('off')

    # 3. Separated temporal component
    axes[2].plot(delta, temporal_rf, 'o-')
    axes[2].axhline(0, color='grey', linestyle='--')
    axes[2].set_title("Separated Temporal Kernel")
    axes[2].set_xlabel("Lag (frames)")
    axes[2].set_ylabel("Weight")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
def extract_all_temporal_kernels(all_rfs_spatiotemporal):
    """
    Extracts the primary temporal kernel for every neuron using SVD.

    This function also corrects for the sign ambiguity of SVD by ensuring
    the main lobe of the corresponding spatial RF is positive.

    Args:
        all_rfs_spatiotemporal (list of np.ndarray): 
            A list where each element is a (num_pixels, num_lags) spatio-temporal RF.

    Returns:
        np.ndarray: A (num_neurons, num_lags) array of all temporal kernels.
    """
    print("Extracting temporal kernels for all neurons...")
    all_temporal_kernels = []

    for w_hat in all_rfs_spatiotemporal:
        # Center the data
        w_centered = w_hat - w_hat.mean(axis=1, keepdims=True)
        
        # Perform SVD
        U, _, Vt = svd(w_centered, full_matrices=False)
        
        # Get the primary spatial and temporal components
        spatial_component = U[:, 0]
        temporal_component = Vt[0, :]
        
        # --- Sign Correction ---
        # Check if the main spatial lobe (peak absolute value) is negative.
        # If so, flip both components to maintain consistency.
        if np.abs(np.min(spatial_component)) > np.abs(np.max(spatial_component)):
            temporal_component *= -1
            
        all_temporal_kernels.append(temporal_component)
        
    print("Extraction complete.")
    return np.array(all_temporal_kernels)



def extract_all_spatial_kernels(all_rfs_spatiotemporal, stim_height, stim_width):
    """
    Extracts the primary spatial kernel for every neuron using SVD.
    Also corrects for sign ambiguity.

    Args:
        all_rfs_spatiotemporal (list of np.ndarray): 
            List of (num_pixels, num_lags) spatio-temporal RFs.
        stim_height (int): The height of the stimulus images.
        stim_width (int): The width of the stimulus images.

    Returns:
        list of np.ndarray: A list of 2D spatial kernels.
    """
    print("Extracting spatial kernels for all neurons...")
    all_spatial_kernels = []

    for w_hat in all_rfs_spatiotemporal:
        w_centered = w_hat - w_hat.mean(axis=1, keepdims=True)
        U, _, _ = svd(w_centered, full_matrices=False)
        spatial_component = U[:, 0].reshape(stim_height, stim_width)

        # Sign Correction: Ensure the main lobe is positive (excitatory)
        if np.abs(np.min(spatial_component)) > np.abs(np.max(spatial_component)):
            spatial_component *= -1

        all_spatial_kernels.append(spatial_component)

    print("Extraction complete.")
    return all_spatial_kernels

def visualize_all_temporal_kernels(all_temporal_kernels, delta):
    """
    Visualizes all temporal kernels together as a heatmap.

    Args:
        all_temporal_kernels (np.ndarray): 
            A (num_neurons, num_lags) array of temporal kernels.
        delta (list or np.ndarray): 
            The time lags corresponding to the columns of the kernel array.
    """
    print("Generating population temporal kernel heatmap...")
    
    # Normalize each kernel to have a peak absolute value of 1 for better visualization
    max_abs_vals = np.max(np.abs(all_temporal_kernels), axis=1, keepdims=True)
    # Avoid division by zero for silent neurons
    max_abs_vals[max_abs_vals == 0] = 1
    normalized_kernels = all_temporal_kernels / max_abs_vals

    fig, ax = plt.subplots(figsize=(8, 12))
    
    # Use imshow to create the heatmap
    im = ax.imshow(normalized_kernels, cmap='bwr', aspect='auto', interpolation='nearest')
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label("Normalized Weight")
    
    # Set labels and title
    ax.set_title("Temporal Kernels for All Neurons")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Neuron ID")
    
    # Set the x-axis ticks to match your lags
    ax.set_xticks(np.arange(len(delta)))
    ax.set_xticklabels(delta)
    
    plt.tight_layout()
    plt.show()
    
    
def visualize_all_spatial_rfs(all_spatial_rfs, background_image=None, threshold_frac=0.6):
    """
    Visualizes all spatial receptive fields by overlaying their contours.

    Args:
        all_spatial_rfs (list of np.ndarray): List of 2D spatial RFs.
        background_image (np.ndarray, optional): A 2D image to plot contours on.
        threshold_frac (float, optional): Fraction of the max value to set the contour level.
    """
    print("Generating population spatial receptive field contour plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the background image if provided
    if background_image is not None:
        ax.imshow(background_image, cmap='gray', alpha=0.5)
    else:
        # If no background, set the limits manually based on RF shape
        ax.set_xlim(0, all_spatial_rfs[0].shape[1])
        ax.set_ylim(all_spatial_rfs[0].shape[0], 0)
        ax.set_aspect('equal')
        ax.set_facecolor('black')

    # Loop through each neuron's spatial RF
    for rf in all_spatial_rfs:
        # Find positive (excitatory) contours
        max_val = rf.max()
        if max_val > 1e-6: # Check for non-trivial positive values
            ax.contour(rf, levels=[max_val * threshold_frac], colors='red', linewidths=1.0)

        # Find negative (inhibitory) contours
        min_val = rf.min()
        if min_val < -1e-6: # Check for non-trivial negative values
            ax.contour(rf, levels=[min_val * threshold_frac], colors='blue', linewidths=1.0)

    ax.set_title("Population Spatial Receptive Fields (Red=Excitatory, Blue=Inhibitory)")
    ax.axis('off')
    plt.show()


def visualize_stimulus_movie(stim_movie, title="Stimulus Movie", interval=50):
    """
    Creates and displays an animation of the stimulus movie.

    Args:
        stim_movie (np.ndarray): A 3D array of shape (num_frames, height, width).
        title (str): The title for the plot.
        interval (int): The delay between frames in milliseconds.
    """
    if stim_movie.ndim != 3:
        print("Error: Input must be a 3D array (frames, height, width).")
        return

    num_frames = stim_movie.shape[0]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.title(title)
    ax.axis('off')

    # Initialize the image plot with the first frame
    # Use a grayscale colormap and set the limits for pixel values
    img = ax.imshow(stim_movie[0, :, :], cmap='gray', vmin=np.min(stim_movie), vmax=np.max(stim_movie))

    def update(frame_num):
        # This function is called for each frame of the animation
        img.set_data(stim_movie[frame_num, :, :])
        ax.set_title(f"Stimulus Movie (Frame {frame_num}/{num_frames})")
        return [img]

    # Create the animation object
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

    # Display the animation in the notebook
    # Note: This may take a moment to render
    plt.close(fig) # Close the static plot to only show the animation
    return anim

def preprocess_data(data):
    """
    Performs initial cleaning and preprocessing on the loaded data dictionary.
    - Renames mislabelled columns in 'stim_epoch_table'.
    - Converts time columns in 'stim_table' from ms to seconds.

    Args:
        data (dict): The raw data dictionary loaded from the .npz file.

    Returns:
        dict: The preprocessed data dictionary.
    """
    print("Performing initial data preprocessing...")

    # --- Rename mislabelled columns in stim_epoch_table ---
    # Make a copy to avoid modifying the original DataFrame in place
    epoch_table = data['stim_epoch_table'].copy()
    epoch_table = epoch_table.rename(
        columns={
            epoch_table.columns[0]: "stimulus",
            epoch_table.columns[1]: "start",
            epoch_table.columns[2]: "end",
        }
    )
    data['stim_epoch_table'] = epoch_table

    # --- Convert time columns in stim_table to seconds ---
    stim_table = data['stim_table'].copy()
    stim_table["start_s"] = stim_table["start"] / 1000.0
    stim_table["end_s"] = stim_table["end"] / 1000.0
    data['stim_table'] = stim_table
    data['preprocessing_done'] = True  
    print("Preprocessing complete.")
    return data

def filter_timeseries_by_epoch(data, stimulus_name='locally_sparse_noise'):
    """
    Filters time-series data (dff, t, running_speed) to specific stimulus epochs.

    Args:
        data (dict): The main data dictionary.
        stimulus_name (str): The name of the stimulus epoch to filter for.

    Returns:
        tuple: A tuple containing the filtered arrays:
            - t_filtered (np.ndarray)
            - dff_filtered (np.ndarray)
            - running_speed_filtered (np.ndarray)
    """
    print(f"Filtering time-series data for '{stimulus_name}' epochs...")
    if not data.get('preprocessing_done', False):
        logging.debug("Data preprocessing not done. Running preprocessing...")
        preprocess_data(data)  # Ensure preprocessing is done
        

    # Identify the relevant stimulus epochs from the table
    epoch_table = data['stim_epoch_table']
    print(f"Total epochs available: {len(epoch_table)}")
    print(f"Epochs columns: {epoch_table.columns.tolist()}")
    print(f"Looking for epochs with stimulus name: {stimulus_name}")
    print(f"Epochs with stimulus '{stimulus_name}': {epoch_table[epoch_table['stimulus'] == stimulus_name].shape[0]}  ")
    target_epochs = epoch_table[epoch_table['stimulus'] == stimulus_name].copy()

    if target_epochs.empty:
        raise ValueError(f"Stimulus epoch '{stimulus_name}' not found in stim_epoch_table.")

    # Convert start/end times to seconds for comparison
    target_epochs["start_s"] = pd.to_numeric(target_epochs["start_time"]) / 1000.0
    target_epochs["end_s"] = pd.to_numeric(target_epochs["end_time"]) / 1000.0

    # Create a boolean mask for all time points that fall within any of the target epochs
    time_mask = np.full(data["t"].shape, False, dtype=bool)
    for _, row in target_epochs.iterrows():
        time_mask |= ((data["t"] >= row["start_s"]) & (data["t"] < row["end_s"]))

    # Apply the time mask to the main time-series data
    t_filtered = data["t"][time_mask]
    dff_filtered = data["dff"][:, time_mask]
    running_speed_filtered = data["running_speed"][:, time_mask]
    
    print("Time-series filtering complete.")
    print(f"Filtered dff shape: {dff_filtered.shape}")
    print(f"Filtered t shape: {t_filtered.shape}")

    return t_filtered, dff_filtered, running_speed_filtered


def filter_stimulus_data(data, t_filtered):
    """
    Filters the stimulus frames and table to match the relevant time window.

    This function identifies which stimulus frames were presented during the
    time epochs defined by `t_filtered` and returns only those frames and
    their corresponding metadata.

    Args:
        data (dict): The main data dictionary containing 'stim' and 'stim_table'.
        t_filtered (np.ndarray): A 1D array of time points for the desired epoch.

    Returns:
        tuple: A tuple containing:
            - stim_filtered (np.ndarray): The filtered stimulus movie frames.
            - stim_table_filtered_df (pd.DataFrame): The filtered stimulus metadata table.
    """
    print("Filtering stimulus data to match the selected time epochs...")

    # Make a copy to avoid modifying the original data dictionary in place
    stim_table = data['stim_table'].copy()

    # Get the min and max time of the filtered calcium recording
    min_t_filtered = t_filtered.min()
    max_t_filtered = t_filtered.max()

    # Convert stim_table's 'start' and 'end' to seconds for comparison
    stim_table["start_s"] = stim_table["start"] / 1000.0
    stim_table["end_s"] = stim_table["end"] / 1000.0

    # Filter stim_table to include only frames whose presentation times
    # overlap with the filtered calcium recording time window.
    stim_table_filtered_df = stim_table[
        (stim_table["start_s"] >= min_t_filtered)
        & (stim_table["start_s"] < max_t_filtered)
    ].copy()

    # Get the unique frame indices from the filtered table
    unique_filtered_frames_indices = stim_table_filtered_df["frame"].unique()
    
    # Ensure indices are valid and within the bounds of the main stim array
    valid_frame_indices = unique_filtered_frames_indices[
        (unique_filtered_frames_indices >= 0)
        & (unique_filtered_frames_indices < data["stim"].shape[0])
    ]
    
    # Select the actual stimulus frames using these valid indices
    stim_filtered = data["stim"][valid_frame_indices]

    # Final cleanup of the filtered DataFrame to ensure consistency
    stim_table_filtered_df = (
        stim_table_filtered_df[stim_table_filtered_df["frame"].isin(valid_frame_indices)]
        .sort_values(by="start_s")
        .reset_index(drop=True)
    )
    
    print("Stimulus filtering complete.")
    print(f"Filtered stim_table_df shape: {stim_table_filtered_df.shape}")
    print(f"Filtered stim shape: {stim_filtered.shape}")
    
    return stim_filtered, stim_table_filtered_df

def plot_roi_masks_on_image(roi_masks, background_image, roi_centroids):
    """
    Overlays the contours and index of all ROI masks on a background image.

    Args:
        roi_masks (np.ndarray): A 3D array of shape (num_neurons, height, width).
        background_image (np.ndarray): The 2D max activity projection image.
        roi_centroids (np.ndarray): A 2D array of (row, col) centroid coordinates.
    """
    print("Visualizing all ROI mask outlines and indices...")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display the background image
    ax.imshow(background_image, cmap='gray')

    # Generate a set of distinct colors for the contours
    colors = plt.cm.get_cmap('gist_rainbow', roi_masks.shape[0])

    # Loop through each neuron's ROI mask
    for i in range(roi_masks.shape[0]):
        # Plot the contour of the mask
        ax.contour(roi_masks[i, :, :], levels=[0.5], colors=[colors(i)], linewidths=1.5)
        
        # Get the centroid coordinates for placing the text
        # Centroid is (row, col), which corresponds to (y, x) in plotting
        y, x = roi_centroids[i]
        
        # Add the neuron index as text.
        # We add a path effect (a simple outline) to make the text readable
        # against any background color.
        txt = ax.text(x, y, str(i), color='white', fontsize=8, ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])


    ax.set_title("Anatomical Map of All Neuron ROIs with Indices")
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

def calculate_roi_centroids(roi_masks):
    """
    Calculates the center (centroid) of each neuron's ROI mask.

    Args:
        roi_masks (np.ndarray): A 3D array of shape (num_neurons, height, width).

    Returns:
        np.ndarray: A 2D array of shape (num_neurons, 2) containing the 
                    (row, col) coordinates of each neuron's centroid.
    """
    print("Calculating centroids of ROI masks...")
    num_neurons = roi_masks.shape[0]
    centroids = np.zeros((num_neurons, 2))

    for i in range(num_neurons):
        # center_of_mass returns (row, col) which corresponds to (y, x) in plotting
        centroids[i, :] = center_of_mass(roi_masks[i, :, :])
        
    print("ROI centroid calculation complete.")
    return centroids    

def calculate_rf_centers(all_spatial_rfs):
    """
    Calculates the center of each spatial receptive field.
    The center is defined as the pixel with the maximum absolute weight.

    Args:
        all_spatial_rfs (list of np.ndarray): A list of 2D spatial RF arrays.

    Returns:
        np.ndarray: A 2D array of shape (num_neurons, 2) containing the
                    (row, col) coordinates of each RF's center.
    """
    print("Calculating centers of spatial receptive fields...")
    rf_centers = []

    for rf in all_spatial_rfs:
        # Find the index of the pixel with the largest absolute value
        max_abs_idx = np.argmax(np.abs(rf))
        
        # Convert the flat index to 2D (row, col) coordinates
        center_coords = np.unravel_index(max_abs_idx, rf.shape)
        rf_centers.append(center_coords)
        
    print("RF center calculation complete.")
    return np.array(rf_centers)

def plot_retinotopic_map(roi_centroids, rf_centers, background_image, axis_to_map='x'):
    """
    Generates a retinotopic map by plotting neuron locations color-coded 
    by their receptive field center positions.

    Args:
        roi_centroids (np.ndarray): (num_neurons, 2) array of physical neuron locations.
        rf_centers (np.ndarray): (num_neurons, 2) array of RF center locations.
        background_image (np.ndarray): The 2D max activity projection image.
        axis_to_map (str): Which axis of the RF to map to color ('x' or 'y').
    """
    if axis_to_map not in ['x', 'y']:
        raise ValueError("axis_to_map must be 'x' or 'y'")

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the max activity projection as the background
    ax.imshow(background_image, cmap='gray')

    # Get the physical locations (x, y for plotting)
    # Note: centroid is (row, col), so we plot (col, row)
    physical_x = roi_centroids[:, 1]
    physical_y = roi_centroids[:, 0]
    
    # Determine the color values based on the chosen RF axis
    if axis_to_map == 'x':
        color_values = rf_centers[:, 1] # RF center x-coordinate (column)
        cbar_label = "RF Center X-Position (pixels)"
        title = "Retinotopic Map of Visual Field (Horizontal Axis)"
    else: # axis_to_map == 'y'
        color_values = rf_centers[:, 0] # RF center y-coordinate (row)
        cbar_label = "RF Center Y-Position (pixels)"
        title = "Retinotopic Map of Visual Field (Vertical Axis)"

    # Create the scatter plot
    scatter = ax.scatter(physical_x, physical_y, c=color_values, cmap='coolwarm', s=80,
                         edgecolors='black', linewidth=1)
    
    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label, size=12)
    
    ax.set_title(title, size=14)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.show()



def visualize_all_strfs(all_rfs_spatiotemporal, delta, stim_dims, neurons_to_plot=None):
    """
    Visualizes the full spatio-temporal receptive fields for a population of neurons.

    Creates a grid where each row is a neuron and each column is a time lag.

    Args:
        all_rfs_spatiotemporal (list of np.ndarray): 
            A list of (num_pixels, num_lags) STRFs for all neurons.
        delta (list or np.ndarray): 
            The time lags used in the analysis.
        stim_dims (tuple): 
            The (height, width) of the stimulus.
        neurons_to_plot (list or range, optional): 
            A list of neuron indices to plot. If None, all neurons are plotted.
            This is useful for plotting a subset if the population is large.
    """
    print("Generating population spatio-temporal receptive field grid...")

    if neurons_to_plot is None:
        neurons_to_plot = range(len(all_rfs_spatiotemporal))
    
    num_neurons_to_plot = len(neurons_to_plot)
    num_lags = len(delta)
    stim_h, stim_w = stim_dims

    # Create a figure with a subplot for each neuron and each lag
    fig, axes = plt.subplots(num_neurons_to_plot, num_lags, 
                             figsize=(num_lags * 2, num_neurons_to_plot * 2))
    
    # Handle the case of plotting a single neuron
    if num_neurons_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle("Spatio-Temporal Receptive Fields for All Neurons", fontsize=16, y=1.0)

    # Find a global color scale limit for consistent visualization
    global_max = max(np.max(np.abs(all_rfs_spatiotemporal[i])) for i in neurons_to_plot)

    # Loop through the selected neurons and lags
    for row_idx, neuron_idx in enumerate(neurons_to_plot):
        strf = all_rfs_spatiotemporal[neuron_idx]
        
        for col_idx, lag in enumerate(delta):
            ax = axes[row_idx, col_idx]
            
            # Get the spatial RF at the current lag
            rf_at_lag = strf[:, col_idx].reshape(stim_h, stim_w)
            
            # Plot the RF
            ax.imshow(rf_at_lag, cmap='bwr', vmin=-global_max, vmax=global_max)
            ax.axis('off')
            
            # Add titles for the first row (lags)
            if row_idx == 0:
                ax.set_title(f"Lag {lag}")
        
        # Add labels for the rows (neuron IDs)
        axes[row_idx, 0].text(-5, stim_h / 2, f"Neuron {neuron_idx}", 
                              va='center', ha='right', fontsize=10, rotation=90)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()




def isolate_sparse_epochs(data, target='locally_sparse_noise', offset_samples=0):
    """
    Returns a boolean mask for the time vector 't', marking only the time points
    that fall within epochs of a specific stimulus type, using sample indices.

    This function is designed to work with a pandas DataFrame for the epoch table
    and uses the 'start' and 'end' columns as direct sample indices for masking.

    Args:
        data (dict): The main data dictionary, must contain 't' and 'stim_epoch_table'.
        target (str): The name of the stimulus epoch to isolate.
        offset_samples (int): An optional offset in samples (frames) to skip from 
                              the beginning of each epoch.

    Returns:
        np.ndarray: A boolean mask with the same shape as data['t'].
    """
    # Ensure the input data is in the expected format
    if 't' not in data or 'stim_epoch_table' not in data:
        raise ValueError("Input 'data' dictionary must contain 't' and 'stim_epoch_table'.")
    if not isinstance(data['stim_epoch_table'], pd.DataFrame):
        raise TypeError("'stim_epoch_table' must be a pandas DataFrame.")

    t = data['t']
    epochs_df = data['stim_epoch_table']
    # Initialize a boolean mask with all False values
    mask = np.zeros_like(t, dtype=bool)
    # --- Step 1: Filter the DataFrame for the target stimulus ---
    target_epochs = epochs_df[epochs_df['stimulus'] == target].copy()
    if target_epochs.empty:
        print(f"Warning: No epochs found with target='{target}'. Returning an all-False mask.")
        return mask

    # --- Step 2: Create the index-based mask ---
    # Iterate through each identified epoch and mark the corresponding indices as True.
    for _, row in target_epochs.iterrows():
        # Get start and end sample indices, ensuring they are integers
        start_index = int(row['start']) + offset_samples
        end_index = int(row['end'])
        
        # Check bounds to prevent errors
        if start_index >= len(mask) or end_index > len(mask):
            print(f"Warning: Epoch indices [{start_index}, {end_index}] are out of bounds for the time vector of length {len(mask)}. Skipping.")
            continue

        # Set the slice of the mask to True
        mask[start_index:end_index] = True
    return mask

def get_active_stimulus_per_timestep(stim_table, total_timesteps):
    """
    Creates an array mapping each timestep to the active stimulus frame ID.

    Args:
        stim_table (pd.DataFrame): DataFrame with 'frame', 'start', and 'end' columns,
                                   where 'start' and 'end' are sample indices.
        total_timesteps (int): The total number of timesteps in the recording
                               (e.g., the length of data['t']).

    Returns:
        np.ndarray: An array of length `total_timesteps` where the value at each
                    index `t` is the ID of the stimulus frame active at that time.
                    Timesteps with no stimulus are marked as -1.
    """
    print("Mapping active stimulus frame to each timestep...")
    
    # Initialize an array to hold the frame ID for each timestep.
    # Use -1 as a placeholder for times when no stimulus is active.
    active_stim_per_timestep = np.full(total_timesteps, -1, dtype=int)
    
    # Iterate through the stimulus table to fill the array
    for _, row in stim_table.iterrows():
        frame_id = int(row['frame'])
        start_index = int(row['start'])
        end_index = int(row['end'])
        
        # Check for valid indices to prevent errors
        if start_index < total_timesteps:
            # The end_index can be beyond the array length, so we clip it
            active_stim_per_timestep[start_index:min(end_index, total_timesteps)] = frame_id
            
    print("Mapping complete.")
    return active_stim_per_timestep


def gaussian_2d(xy_mesh, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    A 2D Gaussian function for fitting.
    
    Args:
        xy_mesh (tuple): A tuple of x and y meshgrid coordinates.
        amplitude (float): The amplitude of the Gaussian.
        x0, y0 (float): The center coordinates.
        sigma_x, sigma_y (float): The standard deviations (size).
        theta (float): The rotation angle.
        offset (float): The baseline offset.

    Returns:
        np.ndarray: The flattened 2D Gaussian.
    """
    x, y = xy_mesh
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g.ravel()

def fit_2d_gaussian_to_rf(rf_map):
    """
    Fits a 2D Gaussian to a spatial receptive field map.

    Args:
        rf_map (np.ndarray): The 2D spatial RF.

    Returns:
        tuple: A tuple containing:
            - params (list or None): The optimal parameters [amp, x0, y0, sx, sy, theta, offset].
            - r_squared (float or None): The R-squared value indicating goodness-of-fit.
    """
    h, w = rf_map.shape
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)

    # Initial guess for the parameters
    initial_guess = [
        rf_map.max(),      # amplitude
        w/2,               # x0
        h/2,               # y0
        w/4,               # sigma_x
        h/4,               # sigma_y
        0,                 # theta
        0                  # offset
    ]

    try:
        # Use curve_fit to find the best parameters
        popt, pcov = curve_fit(gaussian_2d, (x, y), rf_map.ravel(), p0=initial_guess)
        
        # Calculate R-squared for goodness-of-fit
        residuals = rf_map.ravel() - gaussian_2d((x, y), *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((rf_map.ravel() - np.mean(rf_map.ravel()))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return popt, r_squared
    except RuntimeError:
        # If the fit fails to converge, return None
        return None, None

def quantify_all_rfs(all_spatial_rfs):
    """
    Quantifies the localization of all spatial RFs by fitting 2D Gaussians.

    Args:
        all_spatial_rfs (list of np.ndarray): A list of 2D spatial RFs.

    Returns:
        pd.DataFrame: A DataFrame with quantification metrics for each neuron,
                      including the fitted RF center coordinates.
    """
    print("Quantifying localization for all spatial RFs...")
    results = []
    for i, rf in enumerate(all_spatial_rfs):
        params, r_squared = fit_2d_gaussian_to_rf(rf)
        
        if params is not None:
            # Calculate RF size as the geometric mean of the sigmas
            rf_size = np.sqrt(abs(params[3] * params[4]))
            results.append({
                'neuron_id': i,
                'r_squared': r_squared,
                'rf_size': rf_size,
                'amplitude': params[0],
                'rf_center_x': params[1], # The fitted x-center of the blob
                'rf_center_y': params[2]  # The fitted y-center of the blob
            })
        else:
            results.append({
                'neuron_id': i,
                'r_squared': 0, # Assign 0 if fit failed
                'rf_size': np.nan,
                'amplitude': 0,
                'rf_center_x': np.nan,
                'rf_center_y': np.nan
            })
            
    print("Quantification complete.")
    df = pd.DataFrame(results)
    df.set_index('neuron_id', inplace=True)
    df.sort_values(by='r_squared', ascending=False, inplace=True)
    df.reset_index(inplace=True)
    return df 