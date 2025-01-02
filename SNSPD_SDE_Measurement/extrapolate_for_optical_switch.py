import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
from datetime import datetime
from helpers import *

logger = logging.getLogger(__name__)
current_file_dir = Path(__file__).parent

def linear_func(x, a, b):
    return a * x + b

import matplotlib.pyplot as plt

def extrapolate_optical_switch_data(nonlinearity_data_filepath, optical_switch_data_filepath):
    processed_nonlinearity_data = extract_nonlinearity_data(nonlinearity_data_filepath)
    range_setting = -10
    attenuation_setting = 0

    try:
        v = unp.nominal_values(processed_nonlinearity_data[range_setting]['v'])
        v_std = unp.std_devs(processed_nonlinearity_data[range_setting]['v'])
        
        v_dBmW = 10 * np.log10(v / 1e-3)  # Convert Watts to dBmW
        v_std_dBmW = abs((v_std / v) * (10 / np.log(10)))  # Uncertainty in dBmW

        logging.info(f"v: {v}")
        logging.info(f"v_dBmW: {v_dBmW}")

        atts = processed_nonlinearity_data[range_setting]['att']
    except KeyError as e:
        logger.error(f"Range setting {range_setting} data not found: {e}")
        raise

    popt, pcov = curve_fit(linear_func, atts, v_dBmW, sigma=v_std_dBmW, absolute_sigma=True)

    a = ufloat(popt[0], np.sqrt(pcov[0, 0]))
    b = ufloat(popt[1], np.sqrt(pcov[1, 1]))
    logger.info(f"Fitted parameters: a = {a}, b = {b}")

    y_new = a * attenuation_setting + b

    # Plot the data and the fitted line
    plt.errorbar(atts, v_dBmW, yerr=v_std_dBmW, fmt='o', label='Data (with uncertainties)', capsize=5)
    atts_fit = np.linspace(-2, max(atts), 100)
    v_fit = linear_func(atts_fit, *popt)
    plt.plot(atts_fit, v_fit, label='Best Fit Line', color='red')

    plt.axvline(x=attenuation_setting, linestyle='--', color='green', label=f'Extrapolation at x={attenuation_setting}')
    plt.scatter(attenuation_setting, unp.nominal_values(y_new), color='green', label=f'Extrapolated Point: y={y_new:.2uS}')

    plt.xlabel('Attenuation')
    plt.ylabel('Voltage (V)')
    plt.title('Linear Fit and Extrapolation')
    plt.legend()
    plt.grid(True)
    plt.show()

    y_new_mean_W = unp.pow(10, unp.nominal_values(y_new) / 10) / 1000
    y_new_std_W = (unp.std_devs(y_new) / 10) * 10**(unp.nominal_values(y_new) / 10) * np.log(10)
    logger.info(f"Extrapolated value (y_new): {y_new}")
    logger.info(f"Extrapolated value (y_new_mean_W): {y_new_mean_W}")
    logger.info(f"Extrapolated value (y_new_std_W): {y_new_std_W}")


    df = pd.read_pickle(optical_switch_data_filepath)
    power_cpm = np.array(df['power_cpm']).reshape(1, -1)
    power_cpm_mean, power_cpm_std = get_mean_uncertainty(power_cpm)
    power_cpm = ufloat(power_cpm_mean, power_cpm_std)
    power_mpm = ufloat(y_new_mean_W, y_new_std_W/10000)


    switch_data = {
        'power_mpm': [power_mpm],  # Wrap it in a list if it's a scalar or a ufloat
        'power_cpm': [power_cpm]  # Wrap in a list
    }
    result_df = pd.DataFrame(switch_data)

    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'optical_switch_calibration_extrapolated_processed_{now_str}.pkl'
    output_filepath = current_file_dir / "data_sde" / output_filename
    output_filepath.parent.mkdir(exist_ok=True)

    result_df.to_pickle(output_filepath)
    logger.info(f"Processed data saved to: {output_filepath}")
    return output_filepath


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    optical_switch_filepath = current_file_dir / 'data_sde' / 'optical_switch_calibration_data.pkl'
    nonlinearity_data_filepath = current_file_dir / 'data_sde' / 'nonlinearity_factor_raw_power_meaurements_data_20241210-174441.pkl'
    extrapolate_optical_switch_data(nonlinearity_data_filepath, optical_switch_filepath)