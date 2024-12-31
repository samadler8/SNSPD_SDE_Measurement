import os
import logging

import lmfit

import numpy as np
import pandas as pd

from uncertainties import unumpy as unp
from scipy.optimize import brute
from uncertainties import ufloat, correlated_values

from processing_helpers import *

logger = logging.getLogger(__name__)

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
    ranges.sort()

    # Initialize a dictionary to store the results
    d = {}

    # Loop through each range
    for rng in ranges:
        filtered_df = df[df['Range'] == rng]
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
        v_temp = np.empty((len(atts), N), dtype=float)
        vt_temp = np.empty((len(atts), N), dtype=float)

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





def initialize_parameters(N_list, ranges):
    """
    Initialize lmfit.Parameters for fitting polynomial coefficients.

    Args:
        N_list (list): List of maximum polynomial orders for each range.
        ranges (list): List of ranges corresponding to the data.

    Returns:
        lmfit.Parameters: Initialized parameters for the fitting process.
    """
    params = lmfit.Parameters()
    params.add('tau', value=0.5)  # Add the initial parameter tau

    for idx, rng in enumerate(ranges):
        max_order = N_list[idx]
        for order in range(2, max_order + 1):
            param_name = f"b{-rng}{order}"
            params.add(param_name, value=0)

    return params


def calculate_residuals(params, d):
    """
    Calculate residuals for the polynomial fit.

    Args:
        params (lmfit.Parameters): Parameters to optimize.
        d (dict): Averaged data structured by ranges.

    Returns:
        np.ndarray: Residuals for the fit.
    """
    residuals = []

    for rng, data in d.items():
        v = data['v']
        vt = data['vt']
        v_unc = data['vstd']
        vt_unc = data['vtstd']

        model = vt - params['tau'] * v
        order = 2
        param_name = f"b{-rng}{order}"

        while param_name in params:
            model += params[param_name] * (vt**order - params['tau'] * v**order)
            order += 1
            param_name = f"b{-rng}{order}"

        # Normalize residuals by estimated uncertainty
        uncertainty = np.sqrt(vt_unc**2 + (params['tau'] * v_unc)**2)
        residuals.append(model / uncertainty)

    return np.hstack(residuals)


def reduced_chi_squared(N_list, d, ranges):
    """
    Compute the reduced chi-squared for the polynomial fit.

    Args:
        N_list (list): Polynomial orders for each range.
        d (dict): Averaged data structured by ranges.
        ranges (list): List of ranges corresponding to the data.

    Returns:
        float: Reduced chi-squared value.
    """
    N_list = np.array(N_list, dtype=int)
    params = initialize_parameters(N_list, ranges)
    fit_result = lmfit.minimize(calculate_residuals, params, method='leastsq', args=(d,))
    return fit_result.redchi


def optimize_polynomial_orders(d, ranges):
    """
    Optimize polynomial orders for each range using a brute force approach.

    Args:
        d (dict): Averaged data structured by ranges.
        ranges (list): List of ranges corresponding to the data.

    Returns:
        np.ndarray: Optimized polynomial orders for each range.
    """
    param_space = (slice(1, 6, 1),) * len(ranges)  # Order range: 1 to 5 inclusive
    optimized_orders = brute(reduced_chi_squared, param_space, args=(d, ranges))
    return optimized_orders.astype(int)



def find_poly_fit(d, ranges, orders=np.array([4, 3, 2, 2, 2, 2])):
    # Polynomial orders should minimize reduced chi-square
    # See methods named 'redchi' and 'optimize_orders' below
    params = initialize_parameters(orders, ranges)
    fit = lmfit.minimize(calculate_residuals, params, method='leastsq', args=(d,))
    logger.debug(lmfit.fit_report(fit.params))
    return fit


def P_range(params, rng, v):
    """
    Compute linearized power P given the parameters of the polynomial,
    power meter range setting 'rng', and the readings 'v'
    """
    #  assumes params is an lmfit Parameters
    k = 2
    out = v + 0
    name = f'b{k-rng*10}'
    while name in params:
        out += params[name]*(v**k)
        k += 1
        name = f'b{k-rng*10}'
    return out


def P_range_unc(params, covar, rng, v):
    """
    Compute linearized power P with uncertainties and covariances
    given the parameters of the polynomial 'params',
    their covariance matrix 'covar',
    power meter range setting 'rng', and the readings 'v'
    """
    #  assumes params is an lmfit Parameters
    k = 2
    out = v + 0
    params_unc = correlated_values([params[name].value for name in params], covar)
    name = f'b{k-rng*10}'
    while name in params:
        # coeff = ufloat(params[name].value, params[name].stderr)
        coeff = params_unc[list(params.keys()).index(name)]
        out += coeff*(v**k)
        # print(coeff*(v**k))
        k += 1
        name = f'b{k-rng*10}'
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    fpath = 'data_sde/SK3_data_dict__20241212-225454.pkl'
    logger.info(f'Processing file: {fpath}')

    # Extract nonlinearity data and ranges
    d = extract_nonlinearity_data(fpath)
    ranges = d.keys()

    # Optimize polynomial orders
    N_list = optimize_polynomial_orders(d, ranges)
    N_list = N_list.astype(int).tolist()
    logger.info(f'Optimized polynomial orders: {N_list}')

    # Fit polynomial and calculate range discontinuities
    fit = find_poly_fit(d, ranges, N_list)
    logger.info(f'Fit reduced chi-squared: {fit.redchi}')

    rng_disc = {}
    rng_disc[-10] = ufloat(1, 0)
    rng_disc_factor = ufloat(1, 0)

    for rng in ranges:
        overlap = set(d[rng]['att']).intersection(d[rng + 10]['att'])
        idx1 = [list(d[rng]['att']).index(x) for x in overlap]
        idx2 = [list(d[rng + 10]['att']).index(x) for x in overlap]
        ratio = (P_range_unc(fit.params, fit.covar, rng, d[rng]['v'][idx1])/P_range_unc(fit.params, fit.covar, rng + 10, d[rng + 10]['v'][idx2]))
        rng_disc_factor *= ufloat(np.mean(ratio), np.std(ratio))
        rng_disc[rng] = rng_disc_factor

    # Save calibration data to a pickle file
    nonlinear_calibration_data = {
        "fit_params": fit.params.valuesdict(),
        "covar": fit.covar,
        "rng_disc": rng_disc,
    }
    calibration_filepath = os.path.join("data_sde", "nonlinear_calibration_data__.pkl")
    pd.to_pickle(nonlinear_calibration_data, calibration_filepath)
    logger.info(f"Calibration data saved to {calibration_filepath}")