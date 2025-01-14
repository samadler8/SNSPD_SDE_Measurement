import os
import hashlib
import logging

import numpy as np

from pathlib import Path

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("processing_helpers.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

def compute_sha1_hash(filename):
    if filename is not None and os.path.isfile(filename):
        sha1 = hashlib.sha1()
        with open(filename, 'rb') as stream:
            for chunk in iter(lambda: stream.read(4096), b''):
                sha1.update(chunk)
        logger.info(f"SHA1hash({os.path.basename(filename)}): {sha1.hexdigest()}")
        return sha1.hexdigest()
    else:
        logger.warning("compute_sha1_hash: File does not exist. Returning None.")
        return None

def get_plateau(current_array, count_array):
    """
    Extracts the plateau region from the given current and count arrays.

    Args:
        current_array (np.ndarray): Array of current values.
        count_array (np.ndarray): Array of count values.

    Returns:
        tuple: A tuple containing the plateau region of the current and count arrays.
               If no plateau is found, returns (None, None).
    """
    # Calculate the threshold for identifying the plateau
    count_array = np.array(count_array)
    count_array = count_array.mean(axis=1)
    threshold = 0.97 * np.max(count_array)

    # Return the plateau region if indices are found
    plateau_cur = []
    plateau_counts = []
    for i in range(current_array.size):
        if count_array[i] >= threshold:
            plateau_cur.append(current_array[i])
            plateau_counts.append(count_array[i])

    return np.array(plateau_cur), np.array(plateau_counts)