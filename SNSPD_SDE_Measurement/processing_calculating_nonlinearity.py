import os
import logging

import pickle
import lmfit

import numpy as np
import pandas as pd

from uncertainties import unumpy as unp
from scipy.optimize import brute
from uncertainties import ufloat, correlated_values
from pathlib import Path
from datetime import datetime

from helpers import *
from processing_helpers import *

logger = logging.getLogger(__name__)
current_file_dir = Path(__file__).parent

def initialize_parameters(poly_order_list, ranges):
    """
    Initialize lmfit.Parameters for fitting polynomial coefficients.

    Args:
        poly_order_list (list): List of maximum polynomial orders for each range.
        ranges (list): List of ranges corresponding to the data.

    Returns:
        lmfit.Parameters: Initialized parameters for the fitting process.
    """
    params = lmfit.Parameters()
    params.add('tau', value=0.5)  # Add the initial parameter tau (set to 0.5 becasue tau refers to the second attenuator being set to 3 dBmW which effectively halves the power)

    for i, rng in enumerate(ranges):
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
        processed_nonlinearity_data (dict): Averaged data structured by ranges.

    Returns:
        np.ndarray: Residuals for the fit.
    """
    residuals = []

    convert_to_dBmW_before_fitting = 0

    for rng, data in processed_nonlinearity_data.items():
        # Extract nominal values and uncertainties
        v = unp.nominal_values(data['v'])
        vt = unp.nominal_values(data['vt'])
        v_std = unp.std_devs(data['v'])
        vt_std = unp.std_devs(data['vt'])
        if convert_to_dBmW_before_fitting:
            # Convert nominal values from W to dBmW
            v = 10 * np.log10(v / 1e-3)  # Convert Watts to dBmW
            vt = 10 * np.log10(vt / 1e-3)  # Convert Watts to dBmW
            # Convert uncertainties from W to dBmW
            v_std = (v_std / v) * (10 / np.log(10))  # Uncertainty in dBmW
            vt_std = (vt_std / vt) * (10 / np.log(10))  # Uncertainty in dBmW

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


def reduced_chi_squared(poly_order_list, processed_nonlinearity_data, ranges):
    """
    Compute the reduced chi-squared for the polynomial fit.

    Args:
        poly_order_list (list): Polynomial orders for each range.
        d (dict): Averaged data structured by ranges.
        ranges (list): List of ranges corresponding to the data.

    Returns:
        float: Reduced chi-squared value.
    """
    poly_order_list = np.array(poly_order_list, dtype=int)
    params = initialize_parameters(poly_order_list, ranges)
    fit_result = lmfit.minimize(calculate_residuals, params, method='leastsq', args=(processed_nonlinearity_data,))
    return fit_result.redchi

def find_poly_fit(processed_nonlinearity_data, ranges, poly_order_list):
    # Polynomial orders should minimize reduced chi-square
    params = initialize_parameters(poly_order_list, ranges)
    fit = lmfit.minimize(calculate_residuals, params, method='leastsq', args=(processed_nonlinearity_data,))
    logger.debug(lmfit.fit_report(fit.params))
    return fit




if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    fpath = os.path.join(current_file_dir, 'data_sde', 'nonlinearity_factor_raw_power_meaurements_data_20241210-174441.pkl')
    logger.info(f'Processing file: {fpath}')

    # Extract nonlinearity data and ranges
    processed_nonlinearity_data = extract_nonlinearity_data(fpath)
    ranges = processed_nonlinearity_data.keys()

    # Optimize polynomial orders
    poly_order_param_space = (slice(1, 6, 1),) * len(ranges)  # order range: 1 to 5 inclusive
    optimized_orders = brute(reduced_chi_squared, poly_order_param_space, args=(processed_nonlinearity_data, ranges)) # Optimize polynomial orders for each range using a brute force approach.
    poly_order_list = optimized_orders.astype(int).tolist()
    logger.info(f'Optimized polynomial orders: {poly_order_list}')

    # Fit polynomial and calculate range discontinuities
    fit = find_poly_fit(processed_nonlinearity_data, ranges, poly_order_list)
    logger.info(f'Fit reduced chi-squared: {fit.redchi}')


    rng_disc = {}
    rng_disc[-10] = ufloat(1, 0)  # Initialize for the base range
    rng_disc_factor = ufloat(1, 0)

    for rng in ranges:
        logger.debug(f'rng in ranges: {rng}')

        if rng + 10 not in ranges:
            logger.warning(f"Skipping range {rng} as {rng + 10} is not in ranges.")
            continue

        # Check for overlapping data
        overlap = set(processed_nonlinearity_data[rng]['att']).intersection(processed_nonlinearity_data[rng + 10]['att'])

        if overlap:
            # Use overlapping data
            idx1 = [list(processed_nonlinearity_data[rng]['att']).index(x) for x in overlap]
            idx2 = [list(processed_nonlinearity_data[rng + 10]['att']).index(x) for x in overlap]

            ratio = P_range_unc(fit.params, fit.covar, rng, processed_nonlinearity_data[rng]['v'][idx1]) / \
                    P_range_unc(fit.params, fit.covar, rng + 10, processed_nonlinearity_data[rng + 10]['v'][idx2])

            ratio_nominal = unp.nominal_values(ratio)
            ratio_std_dev = unp.std_devs(ratio)

            rng_disc_factor *= ufloat(np.mean(ratio_nominal), np.std(ratio_std_dev))
        else:
            # No overlap: Extrapolate from the previous range
            logger.warning(f"No overlapping attenuation data between ranges {rng} and {rng + 10}. Extrapolating.")

            # Predict the first value of the next range using the previous range's polynomial model
            v_next = processed_nonlinearity_data[rng + 10]['v']  # Measurements in the next range
            v_predicted = P_range(fit.params, rng, v_next)  # Predicted values from the previous range

            ratio = P_range_unc(fit.params, fit.covar, rng, v_predicted) / \
                    P_range_unc(fit.params, fit.covar, rng + 10, v_next)

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
    logger.debug(f"nonlinear_calibration_data: {nonlinear_calibration_data}")
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    calibration_filepath = os.path.join(current_file_dir, "data_sde", f"nonlinear_calibration_data__{now_str}.pkl")
    pd.to_pickle(nonlinear_calibration_data, calibration_filepath)
    logger.info(f"Calibration data saved to {calibration_filepath}")