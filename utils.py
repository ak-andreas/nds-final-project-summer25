import numpy as np
import os
import datetime
import json
import utils as U
import logging

import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
                f, allow_pickle=True
            )  # allow_pickle=True is needed for object arrays

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