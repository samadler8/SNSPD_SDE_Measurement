import os
import hashlib
import logging

import numpy as np


logger = logging.getLogger(__name__)

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

def get_mean_uncertainty(rawdata):
    """
    Estimate the uncertainty for each row of raw data. 

    The function uses the standard deviation as the primary uncertainty measure. 
    If the standard deviation is too small, a minimum uncertainty is calculated
    based on a uniform distribution for quantization error, specific to ANDO 
    power meters with limited resolution.

    Parameters:
        rawdata (numpy.ndarray): 2D array of raw data, where each row represents
                                a set of measurements.

    Returns:
        numpy.ndarray: 1D array of uncertainties for each row.
    """
    import numpy as np

    # Calculate standard deviation and mean for each row
    std = rawdata.std(axis=1, ddof=1)
    avg = rawdata.mean(axis=1)

    # Initialize the minimum uncertainty array
    min_unc = np.zeros_like(avg)

    # Define minimum uncertainty based on quantization error for different ranges
    min_unc[avg > 1e-9] = 1e-12 * 0.5 / (3**0.5)
    min_unc[avg > 1e-6] = 1e-9 * 0.5 / (3**0.5)
    min_unc[avg > 1e-3] = 1e-6 * 0.5 / (3**0.5)

    # Replace small standard deviations with the minimum uncertainty
    unc = np.maximum(std, min_unc)

    return avg, unc


def get_plateau(data):
    Cur_Array = np.array(data['Cur_Array'])
    Count_Array = np.array(data['Count_Array'])
    Dark_Count_Array = np.array(data['Dark_Count_Array'])
    Real_Count_Array = np.maximum(Count_Array - Dark_Count_Array, 0)

    max_counts = np.sort(Real_Count_Array)[-2]

    threshold = 0.94 * max_counts
    plateau_indices = np.where(Real_Count_Array >= threshold)[0]

    if len(plateau_indices) > 0:
        return Cur_Array[plateau_indices[-1]] - Cur_Array[plateau_indices[0]]
    return 0