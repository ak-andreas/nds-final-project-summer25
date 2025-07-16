
# %% [markdown]
# # Neural Data Science Project 02
# 
# ## Working with Calcium data
# 
# In the following project you will recieve a data set, along with a set of questions. Use the methods that you have learned throughout this course to explore the data and to answer the questions. You are free to use tools, resources and libraries as you see fit. Use comments and markdown cells to document your thought process and to explain your reasoning. We encourage you to compare different algorithms or to implement state of the art solutions. The notebook should be self contained, although you may offload some functions to a `utils.py`. The notebook should be concluded with a final summary / conclusions section.

# %% [markdown]
# ## Context
# ![image.png](output/images/vcd-task.png)
# 
# The data set that goes along with this notebook was recorded using in vivo 2-photon calcium imaging to measure the activity of genetically identified neurons in the visual cortex of mice performing a go/no-go visual change detection task. The data recordings stem from primary visual cortex and a GCaMP6f indicator was used. The data was recorded as follows.
# 
# ![image-3.png](output/images/vcd-graph.png)
# 
# The data consists of:
# - the preprocessed activity traces (df/f)
# - the stimulus metadata
# - the stimulus frames
# - the ROI masks for each cell
# - a maximum activity projection of the recorded area
# - running speed
# - table of stimulus epochs
# 
# You will only work with a locally sparse noise stimulus.
# 
# Since the experiments were performed in sequence the calcium recordings that you receive also contain some other stimulus modalities (see `data["stim_epoch_table"]`). You can ignore these sections of the time-series data during analysis. Not all the data provided has to be used, however it can be incorporated into your analysis.

# %%
# import packages here

import numpy as np
import pandas as pd
import jupyter_black

jupyter_black.load()

# %%
# load data
def load_data(path="."):
    def array2df(d, key, cols):
        d[key] = pd.DataFrame(d[key], columns=cols)

    data = np.load(path + "/dff_data_rf.npz", allow_pickle=True)
    data = dict(data)
    array2df(data, "stim_table", ["frame", "start", "end"])
    array2df(data, "stim_epoch_table", ["start", "end", "stimulus"])

    return data


def print_info(data):
    data_iter = ((k, type(v), v.shape) for k, v in data.items())
    l = [f"[{k}] - {t}, - {s}" for k, t, s in data_iter]
    print("\n".join(l) + "\n")


data = load_data()

print("Overview of the data")
print_info(data)

# %% [markdown]
# ### What are locally sparse noise stimulus ?

# %% [markdown]
# **Locally Sparse**: In each frame, most of the screen is a neutral grey. Only a few, small, randomly 
# chosen locations ("local") have either a black or a white square. The rest of the screen is blank ("sparse").

# %%
import utils as U
import importlib

importlib.reload(U)  # Use this if you update your utils.py file

# # First, let's look at a few static frames again for context
U.plot_stim_frames(data, indices_to_display=[0, 50, 100, 150, 200])

# %%
from IPython.display import HTML
import utils as U
import importlib

importlib.reload(U)
# 0. Load the raw data
data = load_data()

# 1. Perform initial preprocessing (column renaming, time conversion)
# data = U.preprocess_data(data)

# 2. Filter the time-series data (dff, t, etc.) for the correct epochs
t_filtered, dff_filtered, running_speed_filtered = U.filter_timeseries_by_epoch(
    data, stimulus_name="locally_sparse_noise"
)

# 3. Filter the stimulus frames based on the new filtered time vector
stim_filtered, stim_table_filtered_df = U.filter_stimulus_data(data, t_filtered)

# # First, let's look at a few static frames again for context
U.plot_stim_frames(data, indices_to_display=[0, 50, 100, 150, 200])

# # Now, visualize the first 1000 frames of the filtered stimulus as a movie
# # (Using a slice like [:1000] makes the animation render faster)
HTML(U.visualize_stimulus_movie(stim_filtered[:1000]).to_jshtml())

# %% [markdown]
# # What is the maximal activity projection?
# 

# %% [markdown]
# The "Maximum Activity Projection" is a single summary image created by collapsing that entire movie into one picture.
# 
# For every single pixel in the 512x512 grid, the computer looks through all the frames of the movie and finds the highest (maximum) brightness value that pixel ever reached during the entire recording. It then creates a new, single image where each pixel's value is set to that maximum brightness.
# 
# The resulting image gives you a static, anatomical-like view of the recording area.
# 
# * **Bright Spots**: The brightest spots in the projection are typically the locations of neurons (or parts of neurons) that were highly active (i.e., had a large calcium influx, causing high fluorescence) at some point during the experiment.
# 
# * **Background**: The dimmer areas are parts of the tissue that were never highly fluorescent.
# 
# Why is it useful ?
# 
# It serves as an excellent anatomical reference map. It's the perfect background image to overlay  other spatial data on, such as:
# 
# The ROI masks to see the exact outlines of the detected neurons.
# 
# The receptive field centers, show where the "seeing" parts of the neurons are located relative to their physical bodies.

# %%
import importlib
import logging
import utils as U

importlib.reload(U)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# 2. Call the visualization functions
logging.info("\n--- Running Visualization Examples ---")
importlib.reload(U)

# Example 1: Plot max projection and show it on screen
U.plot_max_projection(data)

# %% [markdown]
# ## What are ROI masks ?

# %%
roi_masks = data["roi_masks"]
roi_masks.shape

# %%
# --- Phase 1, Task 1.2 (continued): Visualizing Anatomical Layout ---
# Before we calculate receptive fields, let's visualize the physical
# locations of the neurons we are analyzing.

import importlib
import utils as U

importlib.reload(U)  # Reload utils to make sure we have the new functions

# --- Step 1: Get the necessary data ---
# The ROI masks contain the shape and location of each neuron.
# The max activity projection provides a good background image.
roi_masks = data["roi_masks"]
max_projection_image = data["max_projection"]

# --- Step 2: Plot the ROI masks on the background image ---
# This function will draw an outline for each of the 189 neurons.
U.plot_roi_masks_on_image(roi_masks=roi_masks, background_image=max_projection_image)

# %%
# Add this cell to your practical02.ipynb notebook for data exploration

# --- Phase 1, Task 1.2 (continued): Visualizing Anatomical Layout ---
# Before we calculate receptive fields, let's visualize the physical
# locations of the neurons we are analyzing.

import importlib
import utils as U

importlib.reload(U)  # Reload utils to make sure we have the new functions

# --- Step 1: Get the necessary data ---
# The ROI masks contain the shape and location of each neuron.
# The max activity projection provides a good background image.
roi_masks = data["roi_masks"]
max_projection_image = data["max_projection"]

# --- Step 2: Calculate the physical center of each neuron ---
# This gives us the coordinates to place the index labels.
roi_centroids = U.calculate_roi_centroids(roi_masks)

# --- Step 3: Plot the ROI masks and their indices on the background image ---
# This function will draw an outline and a label for each of the 189 neurons.
U.plot_roi_masks_on_image(
    roi_masks=roi_masks,
    background_image=max_projection_image,
    roi_centroids=roi_centroids,
)

# %% [markdown]
# # Save Processed Dataset

# %%
# Add this cell to the end of your 00-background.ipynb notebook

# --- Persist the preprocessed data to a file ---
# This saves the key filtered variables so they can be loaded by the next notebook
# without having to re-run all the preprocessing steps.

import os

# Define the output directory and ensure it exists
namespace = "aakarsh"
output_dir = f"./data/{namespace}/background/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


from datetime import datetime

run_date_prefix_current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(f"Saving inferred spikes data to file: {run_date_prefix_current_time}")

output_filename = os.path.join(
    output_dir,
    f"cleaned_{run_date_prefix_current_time}.npz",
)

# We use np.savez_compressed to save multiple arrays into a single file.
# It's efficient and keeps the data organized.
# Note: For the pandas DataFrame, we save its values and columns separately
# so we can perfectly reconstruct it upon loading.
to_save = {
    "t_filtered": t_filtered,
    "dff_filtered": dff_filtered,
    "running_speed_filtered": running_speed_filtered,
    "stim_filtered": stim_filtered,
    "stim_table_filtered": stim_table_filtered_df.values,
    "stim_table_columns": stim_table_filtered_df.columns,
}

np.savez_compressed(output_filename, **to_save)

print(f"Successfully saved preprocessed data to: {output_filename}")

# %%



# %%
import numpy as np
import matplotlib.pyplot as plt

# Load the data you preprocessed
data = np.load('data/preprocessed/deconv_and_sta.npz', allow_pickle=True)

# These are the key arrays you'll use:
dff_inferred = data['dff_ds']      # The downsampled, cleaned calcium traces (cells x time)
t_inferred = data['t_ds']          # The corresponding time vector
fs_inferred = data['fs_ds'].item() # The sampling rate (~10 Hz)
dt = 1 / fs_inferred               # The time step between samples

print(f"Data loaded successfully!")
print(f"Shape of calcium traces: {dff_inferred.shape}")
print(f"Sampling rate: {fs_inferred:.2f} Hz")


# %%


from scipy.signal import deconvolve as sp_deconvolve

def get_exponential_decay_kernel(tau: float, dt: float) -> np.ndarray:
    """Generates a normalized exponential decay kernel."""
    kernel_len = int(np.ceil(5 * tau / dt))
    t = np.arange(kernel_len) * dt
    kernel = np.exp(-t / tau)
    return kernel / kernel.sum()

def deconv_ca(ca: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """Deconvolves a calcium trace using an exponential kernel."""
    kernel = get_exponential_decay_kernel(tau, dt)
    sp_hat, _ = sp_deconvolve(ca, kernel)
    sp_hat = np.pad(sp_hat, (0, ca.shape[0] - sp_hat.shape[0]), 'constant')
    sp_hat = np.clip(sp_hat, 0, None) # No negative spike rates!
    return sp_hat

# --- Parameters for your data ---
cell_to_analyze = 50                 # Pick a cell to analyze
tau_gcamp = 0.15                     # Decay constant for GCaMP6f (in seconds)

# --- Run the deconvolution ---
calcium_trace = dff_inferred[cell_to_analyze, :]
inferred_spikes = deconv_ca(calcium_trace, tau=tau_gcamp, dt=dt)


# only look at first 100 seconds for clarity
max_time = 10  # seconds
max_index = int(max_time / dt)
t_inferred = t_inferred[:max_index]
calcium_trace = calcium_trace[:max_index]
inferred_spikes = inferred_spikes[:max_index]


import matplotlib.pyplot as plt
import numpy as np

# (Assuming all your previous code, including data loading and deconvolution, is here)

# --- Plot the results ---
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# only look at first 100 seconds for clarity
max_time = 100  # seconds
max_index = int(max_time / dt)
t_inferred_segment = t_inferred[:max_index]
calcium_trace_segment = calcium_trace[:max_index]
inferred_spikes_segment = inferred_spikes[:max_index]

# Plot 1: Calcium Trace
ax[0].plot(t_inferred_segment, calcium_trace_segment, color='dodgerblue', label='Cleaned ΔF/F')
ax[0].set_title(f'Cell {cell_to_analyze}: Cleaned Calcium Signal')
ax[0].set_ylabel('ΔF/F')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Plot 2: Inferred Spikes using a Stem Plot
ax[1].stem(
    t_inferred_segment,
    inferred_spikes_segment,
    linefmt='crimson',    # Style for the vertical lines
    markerfmt='o',        # Remove markers at the top of the stem
    basefmt=" ",          # Remove the baseline
    label='Inferred Spikes'
)
ax[1].set_title('Inferred Spiking Activity (Simple Deconvolution)')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Estimated Activity')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()# Ensure the data is within the time limit


# %%
import numpy as np
from scipy.signal import deconvolve as sp_deconvolve
from tqdm import tqdm

# %%
# --- Load your preprocessed data ---
# This file should contain 'dff_ds', 't_ds', and 'fs_ds'
preprocessed_data = np.load('data/preprocessed/deconv_and_sta.npz', allow_pickle=True)
dff_clean = preprocessed_data['dff_ds']
t_vector = preprocessed_data['t_ds']
fs = preprocessed_data['fs_ds'].item()
dt = 1 / fs

# --- Process all cells ---
n_cells = dff_clean.shape[0]
inferred_spikes = np.zeros_like(dff_clean)
tau_gcamp = 0.15

print(f"Running deconvolution on {n_cells} cells...")
for i in tqdm(range(n_cells), desc="Inferring spikes"):
    inferred_spikes[i, :] = deconv_ca(dff_clean[i, :], tau=tau_gcamp, dt=dt)


# %%

# --- Persist the data to a file ---
from datetime import datetime
name = "aakarsh"  # Change this to your namespace if needed
output_dir = f'./data/{name}/spike_inference/'
run_date_prefix_current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(f"Saving inferred spikes data to file: {run_date_prefix_current_time}")
output_filename = f'{output_dir}/inferred_spikes_data_{run_date_prefix_current_time}.npz'

np.savez_compressed(
    output_filename,    
    inferred_spikes=inferred_spikes,
    sampling_frequency=fs,
    time_vector=t_vector,
    tau_gcamp=tau_gcamp,
    dff_clean=dff_clean # Good practice to save the source data too
)



