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

def get_uncertainty(rawdata):
    """
    From the rawdata provided make an estimate of the uncertainty for each
    row of data Use the std, but if it is too small, use an estimate based
    on assuming a uniform distribution on a quantizer. This works for ANDO
    power meters because they only have 4 digit resolution on the higher
    ranges.

    """
    # Nfit = rawdata.shape[0]
    # N = rawdata.shape[1]
    std = rawdata.std(axis=1, ddof=1)
    avg = rawdata.mean(axis=1)
    min_unc = np.zeros(avg.shape)

    # Use estimate quantization error as lower bound
    #   this works for the ando power meters we are using to monitor
    min_unc[avg > 1e-9] = 1e-12 * 0.5 / (3**0.5)
    min_unc[avg > 1e-6] = 1e-9 * 0.5 / (3**0.5)
    min_unc[avg > 1e-3] = 1e-6 * 0.5 / (3**0.5)

    # replace std with uncertainty from a uniform distribution in the
    # quantization of the power meter if the std is too small
    unc = np.where(std < min_unc, min_unc, std)
    return unc

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