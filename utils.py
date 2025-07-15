import numpy as np
import os
import datetime
import json
import utils as U
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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
