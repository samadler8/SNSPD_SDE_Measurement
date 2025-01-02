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


def extrapolate_nonlinearity_data(nonlinearity_data_filepath, rngs, atts):
    processed_nonlinearity_data = extract_nonlinearity_data(nonlinearity_data_filepath)

    extraploated_powers = []
    for rng, att in zip(rngs, atts):
        try:
            v = unp.nominal_values(processed_nonlinearity_data[rng]['v'])
            v_std = unp.std_devs(processed_nonlinearity_data[rng]['v'])
            
            v_dBmW = 10 * np.log10(v / 1e-3)  # Convert Watts to dBmW
            v_std_dBmW = abs((v_std / v) * (10 / np.log(10)))  # Uncertainty in dBmW

            logging.info(f"v: {v}")
            logging.info(f"v_dBmW: {v_dBmW}")

            att_settings = processed_nonlinearity_data[rng]['att']
        except KeyError as e:
            logger.error(f"Range setting {rng} data not found: {e}")
            raise

        popt, pcov = curve_fit(linear_func, att_settings, v_dBmW, sigma=v_std_dBmW, absolute_sigma=True)

        a = ufloat(popt[0], np.sqrt(pcov[0, 0]))
        b = ufloat(popt[1], np.sqrt(pcov[1, 1]))
        logger.info(f"Fitted parameters: a = {a}, b = {b}")

        y_new = a * att + b

        # Plot the data and the fitted line
        plt.errorbar(att_settings, v_dBmW, yerr=v_std_dBmW, fmt='o', label='Data (with uncertainties)', capsize=5)
        att_settings_range = max(att_settings) - min(att_settings)
        atts_fit = np.linspace(min(att_settings) - 3*att_settings_range, max(att_settings) + 3*att_settings_range, 100)
        v_fit = linear_func(atts_fit, *popt)
        plt.plot(atts_fit, v_fit, label='Best Fit Line', color='red')

        plt.axvline(x=att, linestyle='--', color='green', label=f'Extrapolation at x={att}')
        plt.scatter(att, unp.nominal_values(y_new), color='green', label=f'Extrapolated Point: y={y_new:.2uS}')

        plt.xlabel('Attenuation')
        plt.ylabel('Voltage (V)')
        plt.title('Linear Fit and Extrapolation')
        plt.legend()
        plt.grid(True)
        plt.show()

        y_new_mean_W = unp.pow(10, unp.nominal_values(y_new) / 10) / 1000
        y_new_std_W = (unp.std_devs(y_new) / 10) * 10**(unp.nominal_values(y_new) / 10) * np.log(10) / 10000
        logger.info(f"Extrapolated value (y_new): {y_new}")
        logger.info(f"Extrapolated value (y_new_mean_W): {y_new_mean_W}")
        logger.info(f"Extrapolated value (y_new_std_W): {y_new_std_W}")

        extraploated_powers.append(ufloat(y_new_mean_W, y_new_std_W))

    return extraploated_powers


if __name__ == '__main__':
    # filepath = current_file_dir / 'data_sde' / 'attenuator_calibration_data__20241212-142132.pkl'
    # data = pd.read_pickle(filepath)
    # print(data)

    


    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Filepaths
    nonlinearity_data_filepath = current_file_dir / 'data_sde' / 'nonlinearity_factor_raw_power_meaurements_data_20241210-174441.pkl'
    target_att = 29
    
    # Extrapolate nonlinearity data
    extrapolated_powers = extrapolate_nonlinearity_data(nonlinearity_data_filepath, [-10, -40], [0, target_att])

    # Initialize data for DataFrame
    attenuators = [1, 2, 3]
    columns = ['Attenuator', 'Attenuation (dB)', 'Range', 'Power Mean', 'Power Std']
    powers_df = pd.DataFrame(columns=columns)

    # Add data for no attenuator
    powers_df = pd.concat([
        powers_df,
        pd.DataFrame({
            'Attenuator': [None],
            'Attenuation (dB)': [0],
            'Range': [-10],
            'Power Mean': [unp.nominal_values(extrapolated_powers[0])],
            'Power Std': [unp.std_devs(extrapolated_powers[0])]
        })
    ], ignore_index=True)

    # Add data for each attenuator
    for attenuator in attenuators:
        powers_df = pd.concat([
            powers_df,
            pd.DataFrame({
                'Attenuator': [attenuator],
                'Attenuation (dB)': [target_att],
                'Range': [-40],
                'Power Mean': [unp.nominal_values(extrapolated_powers[1])],
                'Power Std': [unp.std_devs(extrapolated_powers[1])]
            })
        ], ignore_index=True)

    # Save the calibration data to a file
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    attenuator_calibration_filename = f"attenuator_calibration_extrapolated_processed__{now_str}.pkl"
    attenuator_calibration_filepath = current_file_dir / "data_sde" / attenuator_calibration_filename

    powers_df.to_pickle(attenuator_calibration_filepath)
    logging.debug(f"powers_df: {powers_df}")
    logging.info(f"Calibration data saved to {attenuator_calibration_filepath}")