import os
import hashlib
import logging

import numpy as np

from pathlib import Path

logger = logging.getLogger(__name__)

current_file_dir = Path(__file__).parent

def compute_sha1_hash(filename):
    if os.path.isfile(filename):
        sha1 = hashlib.sha1()
        with open(filename, 'rb') as stream:
            for chunk in iter(lambda: stream.read(4096), b''):
                sha1.update(chunk)
        logger.debug(f"SHA1hash({os.path.basename(filename)}): {sha1.hexdigest()}")
        return sha1.hexdigest()
    else:
        logger.debug("compute_sha1_hash: File does not exist. Returning None.")
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
    sorted_counts = np.sort(count_array)
    max_average_counts = np.mean(sorted_counts[-4:-1])  # Mean of the top 3 values (ignoring the highest)
    threshold = 0.95 * max_average_counts

    # Identify indices where counts meet or exceed the threshold
    plateau_indices = np.where(count_array >= threshold)[0]

    # Return the plateau region if indices are found
    if len(plateau_indices) > 0:
        plateau_cur = current_array[plateau_indices[0]:plateau_indices[-1] + 1]
        plateau_counts = count_array[plateau_indices[0]:plateau_indices[-1] + 1]
        return plateau_cur, plateau_counts

    # Return None if no plateau is found
    return None, None