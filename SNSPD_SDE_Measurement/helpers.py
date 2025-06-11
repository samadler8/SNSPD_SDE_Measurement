import os
import pickle
import logging
import math
import pytesseract
import pyautogui
import time

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from uncertainties import unumpy as unp
from uncertainties import ufloat, correlated_values
from PIL import Image

current_file_dir = Path(__file__).parent
logger = logging.getLogger(__name__)

# If Tesseract is not in your PATH, specify the path manually
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_screen_and_extract_text(x, y, width, height):
    time.sleep(0.1)
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    text = pytesseract.image_to_string(screenshot)
    logger.info(f"\nExtracted Text: {text}\n")
    return
         
def get_ic(filepath, threshold=1e-4):
    with open(filepath, 'rb') as file:
        data_dict = pickle.load(file)
    index = np.argmax(data_dict['Volt_Meas_Array'] > threshold)
    if data_dict['Volt_Meas_Array'][index] > threshold:
        ic = data_dict['Cur_Array'][index]
    logger.info(f" Critical current found: {ic}")
    return ic

def get_uncertainty(rawdata):
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
        rawdata = rawdata.astype(float)

        # Calculate standard deviation and mean for each row
        std = rawdata.std(axis=1, ddof=1)
        avg = rawdata.mean(axis=1)

        # Initialize the minimum uncertainty array
        min_unc = np.zeros_like(avg)

        # Define minimum uncertainty based on quantization error for different rng_settings
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

                # Apply minimum uncertainty based on quantization error for different rng_settings
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

    return unp.uarray(avg, unc)


def extract_nonlinearity_data(filepath, filtered=True, tau=None):
    """
    Extract nonlinearity data from a file and organize it by rng_settings and attenuation settings.

    Parameters:
        filepath (str): Path to the pickled dataframe file containing the data.

    Returns:
        dict: A dictionary with data organized by range, containing uncertainty arrays.
    """
    # Load the dataframe from the pickle file
    df = pd.read_pickle(filepath)

    if tau is not None:
        taus = [0, tau]
    else:
        taus = df['Attenuator 2'].unique()
    taus.sort()
    taus = taus[:2]
    logger.info(f" taus: {taus}")

    df = df[df['Attenuator 2'].isin(taus)].copy()

    if filtered:
        df = df[df['Attenuator 1'] <= 60].copy()
        rng_settings = df['Range'].unique()
        for rng in rng_settings:
            # Define power thresholds for the current range
            max_power_threshold = 9.75*10**(-4 + rng/10)
            min_power_threshold = 5*10**(-5 + rng/10)
            
            # Filter out invalid power values within the 'Power' column directly in the original DataFrame
            df.loc[df['Range'] == rng, 'Power'] = df.loc[df['Range'] == rng, 'Power'].apply(
                lambda powers: [power for power in powers if min_power_threshold < power < max_power_threshold]
            )

    # Remove rows where 'Power' lists are empty
    df = df[df['Power'].apply(len) > 0]

    

    # Now that filtering is done, we can get to the meat of the code
    rng_settings = df['Range'].unique()
    data_dict = {}
    for rng in rng_settings:
        filtered_df = df[(df['Range'] == rng)].copy()
        
        # Filter data for the two attenuation steps
        filtered_df_tau0 = filtered_df[filtered_df['Attenuator 2'] == taus[0]]
        filtered_df_tau1 = filtered_df[filtered_df['Attenuator 2'] == taus[1]]

        # Get unique and sorted attenuation settings
        att_settings_tau0 = filtered_df_tau0['Attenuator 1'].unique()
        att_settings_tau1 = filtered_df_tau1['Attenuator 1'].unique()
        att_settings = list(set(att_settings_tau0) & set(att_settings_tau1))
        att_settings.sort()
        filtered_df_tau0 = filtered_df_tau0[filtered_df_tau0['Attenuator 1'].isin(att_settings)]
        filtered_df_tau1 = filtered_df_tau1[filtered_df_tau1['Attenuator 1'].isin(att_settings)]
        if filtered_df_tau0.empty or filtered_df_tau1.empty:
            continue

        # Initialize temporary arrays for storing data
        v_temp = np.empty(len(att_settings), dtype=object)
        vt_temp = np.empty(len(att_settings), dtype=object)
        for i, a in enumerate(att_settings):
            filtered_df_tau0_a = filtered_df_tau0[filtered_df_tau0['Attenuator 1'] == a]
            v_temp[i] = np.array([value for sublist in filtered_df_tau0_a['Power'] for value in sublist])
            filtered_df_tau1_a = filtered_df_tau1[filtered_df_tau1['Attenuator 1'] == a]
            vt_temp[i] = np.array([value for sublist in filtered_df_tau1_a['Power'] for value in sublist])

        # Calculate mean and uncertainty for v and vt
        data_dict[rng] = {}
        data_dict[rng]['att'] = att_settings
        data_dict[rng]['v'] = get_uncertainty(v_temp)
        data_dict[rng]['vt'] = get_uncertainty(vt_temp)
    
    return data_dict

def get_param_name(rng, order):
    return f"b{-rng}{order}"

def nonlinear_power_corrections(params, rng, v):
    """
    Compute linearized power P given the parameters of the polynomial,
    power meter range setting 'rng', and the readings 'v'
    """
    order = 2
    out = v
    name = get_param_name(rng, order)
    while name in params:
        out += params[name]*(v**order)
        order += 1
        name = get_param_name(rng, order)
    return out

def nonlinear_power_corrections_unc(params, covar, rng, v):
    """
    Compute linearized power P with uncertainties and covariances
    given the parameters of the polynomial 'params',
    their covariance matrix 'covar',
    power meter range setting 'rng', and the readings 'v'
    """
    order = 2
    out = v
    name = get_param_name(rng, order)
    params_unc = correlated_values([params[name].value for name in params], covar)
    while name in params:
        coeff = params_unc[list(params.keys()).index(name)]
        out += coeff*(v**order)
        order += 1
        name = get_param_name(rng, order)
    return out

def nonlinear_power_corrections_unc_plotting(params, covar, rng, v):
    """
    Compute linearized power P with uncertainties and covariances
    given the parameters of the polynomial 'params',
    their covariance matrix 'covar',
    power meter range setting 'rng', and the readings 'v'
    """
    order = 2
    out = v
    name = get_param_name(rng, order)
    params_unc = correlated_values([params[name] for name in params.keys()], covar, )
    while name in params:
        coeff = params_unc[list(params.keys()).index(name)]
        out += coeff*(v**order)
        order += 1
        name = get_param_name(rng, order)
    return out
