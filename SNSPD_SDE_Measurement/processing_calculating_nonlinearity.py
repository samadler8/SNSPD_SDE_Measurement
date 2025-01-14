import os
import logging
import pickle
import json

import pickle
import lmfit
import math

import numpy as np

from uncertainties import unumpy as unp
from scipy.optimize import brute
from uncertainties import ufloat
from pathlib import Path

from helpers import *
from processing_helpers import *

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("calculating_nonlinearity_log.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

def initialize_parameters(poly_order_list, rng_settings):
    """
    Initialize lmfit.Parameters for fitting polynomial coefficients.

    Args:
        poly_order_list (list): List of maximum polynomial orders for each range.
        rng_settings (list): List of rng_settings corresponding to the data.

    Returns:
        lmfit.Parameters: Initialized parameters for the fitting process.
    """
    params = lmfit.Parameters()
    params.add('tau', value=math.log10(tau))  # Add the initial parameter tau

    for i, rng in enumerate(rng_settings):
        max_order = poly_order_list[i]
        for order in range(2, max_order + 1):
            param_name = get_param_name(rng, order)
            params.add(param_name, value=0)

    return params

def calculate_residuals(params, processed_nonlinearity_data):
    """
    Calculate residuals for the polynomial fit.

    Args:
        params (lmfit.Parameters): Parameters to optimize.
        processed_nonlinearity_data (dict): Averaged data structured by rng_settings.

    Returns:
        np.ndarray: Residuals for the fit.
    """
    residuals = []

    for rng, data in processed_nonlinearity_data.items():
        # Extract nominal values and uncertainties
        v = unp.nominal_values(data['v'])
        vt = unp.nominal_values(data['vt'])
        v_std = unp.std_devs(data['v'])
        vt_std = unp.std_devs(data['vt'])
        
        # Initialize the model
        model = vt - params['tau'] * v
        order = 2
        param_name = get_param_name(rng, order)

        # Add higher-order terms if available
        while param_name in params:
            model += params[param_name] * (vt**order - params['tau'] * v**order)
            order += 1
            param_name = get_param_name(rng, order)

        # Normalize residuals by estimated uncertainty
        uncertainty = np.sqrt(vt_std**2 + (params['tau'] * v_std)**2)
        residuals.append(model / uncertainty)
    # Return residuals as a flattened array
    return np.hstack(residuals)

def reduced_chi_squared(poly_order_list, processed_nonlinearity_data, rng_settings):
    """
    Compute the reduced chi-squared for the polynomial fit.

    Args:
        poly_order_list (list): Polynomial orders for each range.
        d (dict): Averaged data structured by rng_settings.
        rng_settings (list): List of rng_settings corresponding to the data.

    Returns:
        float: Reduced chi-squared value.
    """
    poly_order_list = np.array(poly_order_list, dtype=int)
    params = initialize_parameters(poly_order_list, rng_settings)
    fit_result = lmfit.minimize(calculate_residuals, params, method='leastsq', args=(processed_nonlinearity_data,))
    return fit_result.redchi

def find_poly_fit(processed_nonlinearity_data, rng_settings, poly_order_list):
    # Polynomial orders should minimize reduced chi-square
    params = initialize_parameters(poly_order_list, rng_settings)
    fit = lmfit.minimize(calculate_residuals, params, method='leastsq', args=(processed_nonlinearity_data,))
    logger.info(lmfit.fit_report(fit.params))
    return fit




if __name__ == '__main__':
    fnames = ['nonlinear_calibration_data_tau2.5__20250110-210258.pkl', 'nonlinear_calibration_data_tau3__20250110-210258.pkl', 'nonlinear_calibration_data_tau2__20250110-210258.pkl', 'nonlinear_calibration_data_tau1.5__20250110-210258.pkl']
    for fname in fnames:
        fpath = os.path.join(current_file_dir, 'data_sde', fname)
        logger.info(f'Processing file: {fpath}')

        # Extract nonlinearity data and rng_settings
        processed_nonlinearity_data = extract_nonlinearity_data(fpath)
        rng_settings = list(processed_nonlinearity_data.keys())

        # Optimize polynomial orders
        poly_order_param_space = (slice(1, 6, 1),) * len(rng_settings)  # order range: 1 to 5 inclusive
        optimized_orders = brute(reduced_chi_squared, poly_order_param_space, args=(processed_nonlinearity_data, rng_settings)) # Optimize polynomial orders for each range using a brute force approach.
        poly_order_list = optimized_orders.astype(int).tolist()
        logger.info(f'Optimized polynomial orders: {poly_order_list}')

        # Fit polynomial and calculate range discontinuities
        fit = find_poly_fit(processed_nonlinearity_data, rng_settings, poly_order_list)
        logger.info(f'Fit reduced chi-squared: {fit.redchi}')

        rng_disc = {}
        rng_disc[-10] = ufloat(1, 0)  # Initialize for the base range
        rng_disc_factor = ufloat(1, 0)

        for rng in rng_settings:
            if rng + 10 not in rng_settings:
                logger.warning(f"Skipping range {rng} as {rng + 10} is not in rng_settings.")
                continue

            # Check for overlapping data
            overlap = set(processed_nonlinearity_data[rng]['att']).intersection(processed_nonlinearity_data[rng + 10]['att'])

            idx1 = [list(processed_nonlinearity_data[rng]['att']).index(x) for x in overlap]
            idx2 = [list(processed_nonlinearity_data[rng + 10]['att']).index(x) for x in overlap]

            ratio = nonlinear_power_corrections_unc(fit.params, fit.covar, rng, processed_nonlinearity_data[rng]['v'][idx1]) / \
                    nonlinear_power_corrections_unc(fit.params, fit.covar, rng + 10, processed_nonlinearity_data[rng + 10]['v'][idx2])

            ratio_nominal = unp.nominal_values(ratio)
            ratio_std_dev = unp.std_devs(ratio)

            rng_disc_factor *= ufloat(np.mean(ratio_nominal), np.std(ratio_std_dev))
        
            rng_disc[rng] = rng_disc_factor


        # Save calibration data to a pickle file
        nonlinear_calibration_data = {
            "fit_params": fit.params.valuesdict(),
            "covar": fit.covar,
            "rng_disc": rng_disc,
        }
        logger.info(f"nonlinear_calibration_data: {nonlinear_calibration_data}")

        output_dir = os.path.join(current_file_dir, 'data_sde')
        
        os.makedirs(output_dir, exist_ok=True)
        _, data_filename = os.path.split(os.path.splitext(fpath)[0])
        filename = f'calculation_v1_{data_filename}.pkl'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(nonlinear_calibration_data, f)
        logger.info(f"Calibration calculation saved to {filepath}")

        readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
        os.makedirs(readable_output_dir, exist_ok=True)
        json_filepath = os.path.join(readable_output_dir, )
        with open(json_filepath, 'w') as f:
            json.dump(nonlinear_calibration_data, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        