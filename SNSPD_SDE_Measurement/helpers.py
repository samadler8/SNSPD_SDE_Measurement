import os

import pickle
import logging

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from uncertainties import unumpy as unp
from uncertainties import ufloat, correlated_values

logger = logging.getLogger(__name__)

current_file_dir = Path(__file__).parent
         
def get_ic(pickle_filepath, ic_threshold=1e-4):
    df = pd.read_pickle(pickle_filepath)
    filtered_df = df[df['Voltage'] > ic_threshold]  # Filter rows where Voltage > threshold
    if not filtered_df.empty:
        ic = filtered_df['Current'].iloc[0]  # Get the first current value
    else:
        ic = None
    return ic

def get_mean_uncertainty(rawdata):
    """
    Estimate the uncertainty for each row of raw data. 

    The function uses the standard deviation as the primary uncertainty measure. 
    If the standard deviation is too small, a minimum uncertainty is calculated
    based on a uniform distribution for quantization error, specific to ANDO 
    power meters with limited resolution.

    Parameters:
        rawdata (numpy.ndarray): 2D array where each element is a list (or array) 
                                  representing a set of measurements. 

    Returns:
        numpy.ndarray: 1D array of uncertainties for each row.
    """

    if isinstance(rawdata, np.ndarray) and rawdata.ndim == 2:
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
    
    else:
        # Initialize arrays to store means and uncertainties
        avg = []
        unc = []

        # Iterate through each row in rawdata
        for row in rawdata:
            if isinstance(row, np.ndarray) and row.size > 0:  # Ensure the row has data
                # Calculate mean and standard deviation for non-empty data
                avg_row = row.mean()
                std_row = row.std(ddof=1)

                # Apply minimum uncertainty based on quantization error for different ranges
                if avg_row > 1e-9:
                    min_unc = 1e-12 * 0.5 / (3**0.5)
                elif avg_row > 1e-6:
                    min_unc = 1e-9 * 0.5 / (3**0.5)
                elif avg_row > 1e-3:
                    min_unc = 1e-6 * 0.5 / (3**0.5)
                else:
                    min_unc = 0

                # If standard deviation is small, use the minimum uncertainty
                unc_row = max(std_row, min_unc)

                avg.append(avg_row)
                unc.append(unc_row)
            else:
                # If row is empty or not a valid array, assign NaN or other placeholder
                avg.append(np.nan)
                unc.append(np.nan)

        # Convert lists to numpy arrays
        avg = np.array(avg)
        unc = np.array(unc)

    return avg, unc


def extract_nonlinearity_data(filepath):
    """
    Extract nonlinearity data from a file and organize it by ranges and attenuation settings.

    Parameters:
        filepath (str): Path to the pickled dataframe file containing the data.

    Returns:
        dict: A dictionary with data organized by range, containing uncertainty arrays.
    """

    # Load the dataframe from the pickle file
    df = pd.read_pickle(filepath)

    # Get unique and sorted ranges from the dataframe
    ranges = df['Range'].unique()

    # Initialize a dictionary to store the results
    d = {}


    # Loop through each range
    for rng in ranges:
        
        filtered_df = df[(df['Range'] == rng)]
        atts = filtered_df['Attenuation Setting'].unique()
        threshold = 10**((rng-0.1)/10) * 1e-3
        filtered_df = filtered_df.groupby('Attenuation Setting').filter(
            lambda x: (x['Power'] <= threshold).sum() > len(x) / 2
        )

        d[rng] = {}

        # Get unique and sorted attenuation steps
        taus = filtered_df['Attenuation Step'].unique()
        taus.sort()

        # Filter data for the two attenuation steps
        filtered_df_tau0 = filtered_df[filtered_df['Attenuation Step'] == taus[0]]
        filtered_df_tau1 = filtered_df[filtered_df['Attenuation Step'] == taus[1]]

        # Get unique and sorted attenuation settings
        atts = filtered_df['Attenuation Setting'].unique()
        atts.sort()
        d[rng]['att'] = atts

        # Determine the number of iterations
        N = len(filtered_df['Iteration'].unique())

        # Initialize temporary arrays for storing data
        v_temp = np.empty(len(atts), dtype=object)
        vt_temp = np.empty(len(atts), dtype=object)

        # Loop through each attenuation setting and populate data
        for i, att in enumerate(atts):
            v_temp[i] = filtered_df_tau0[filtered_df_tau0['Attenuation Setting'] == att]['Power'].values
            vt_temp[i] = filtered_df_tau1[filtered_df_tau1['Attenuation Setting'] == att]['Power'].values

        # Calculate mean and uncertainty for v and vt
        v_avg_temp, v_unc_temp = get_mean_uncertainty(v_temp)
        vt_avg_temp, vt_unc_temp = get_mean_uncertainty(vt_temp)

        # Store the data with uncertainties
        d[rng]['v'] = unp.uarray(v_avg_temp, v_unc_temp)
        d[rng]['vt'] = unp.uarray(vt_avg_temp, vt_unc_temp)

    return d

def get_param_name(rng, order):
    return f"b{-rng}{order}"

def P_range(params, rng, v):
    """
    Compute linearized power P given the parameters of the polynomial,
    power meter range setting 'rng', and the readings 'v'
    """
    #  assumes params is an lmfit Parameters
    order = 2
    out = v + 0
    name = get_param_name(rng, order)
    while name in params:
        out += params[name]*(v**order)
        order += 1
        name = get_param_name(rng, order)
    return out

def P_range_unc(params, covar, rng, v):
    """
    Compute linearized power P with uncertainties and covariances
    given the parameters of the polynomial 'params',
    their covariance matrix 'covar',
    power meter range setting 'rng', and the readings 'v'
    """
    #  assumes params is an lmfit Parameters
    order = 2
    out = v
    name = get_param_name(rng, order)
    params_unc = correlated_values([params[name].value for name in params], covar)
    while name in params:
        coeff = params_unc[list(params.keys()).index(name)]
        out += coeff*(v**order)
        # logging.debug(coeff*(v**order))
        order += 1
        name = get_param_name(rng, order)
    return out