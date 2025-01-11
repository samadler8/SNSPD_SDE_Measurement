import sys
import os
import logging
import pickle

import numpy as np
import pandas as pd
import scipy.constants as codata

from datetime import datetime
from pathlib import Path

from scipy.interpolate import interp1d
from uncertainties import ufloat, correlated_values
from uncertainties import unumpy as unp

from helpers import *
from processing_helpers import *

logger = logging.getLogger(__name__)
current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("script_log.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

wavelength = 1566.314e-9 #m
init_rng = 0

class PMCorrectLinearUnc:
    """
    Class for correcting power meter readings using linearization
    and range discontinuity adjustments.
    """

    def __init__(self, fname):
        """
        Initialize the correction class with a pickle file.
        """
        self.fname = fname
        self.fit_params = None
        self.covar = None
        self.rng_disc = {}
        self.load()

    def load(self):
        """
        Load calibration data from a pickle file.
        """

        try:
            # Load data from pickle file
            data = pd.read_pickle(self.fname)
            logging.info(f"PMCorrectLinearUnc load data: {data}")
            self.fit_params = data["fit_params"]
            self.covar = data["covar"]
            self.rng_disc = data["rng_disc"]
            logger.info(f"Calibration data successfully loaded from {self.fname}")
        except Exception as e:
            logger.error(f"Failed to load calibration data from {self.fname}: {e}")

    def nonlinear_power_correction(self, v, rng=None):
        """
        Correct the power reading, applying linearization and range discontinuities.
        Apply corrections for both nonlinearity and range discontinuity.
        Compute linearized power with uncertainties for a given power reading and meter range setting.
        """
        if rng is None:
            rng = self.estimate_range(v)
            logger.info(f"Estimated range: {rng} for reading: {v}")

        if not self.fit_params or self.covar is None or self.covar.size == 0:
            logger.error("Fit parameters or covariance matrix not loaded.")
            return ufloat(v, 0)
        
        order = 2
        corrected_power = v
        name = get_param_name(rng, order)
        params_unc = correlated_values([self.fit_params[name] for name in self.fit_params.keys()], self.covar, )
        
        while name in self.fit_params:
            coeff = params_unc[list(self.fit_params.keys()).index(name)]
            corrected_power += coeff * (v ** order)
            order += 1
            name = get_param_name(rng, order)

        return corrected_power / self.rng_disc[rng]

    # def fix_power_unc(self, v, rng=None):
    #     """
    #     Apply corrections for both nonlinearity and range discontinuity.
    #     """
    #     if rng is None:
    #         rng = self.estimate_range(v)
    #         logger.info(f"Estimated range: {rng} for reading: {v}")

    #     corrected_power = self.fix_power_nonlinearity_unc(v, rng)
    #     return corrected_power / self.rng_disc[rng]

    # def fix_power_nonlinearity_unc(self, v, rng=-10):
    #     """
    #     Compute linearized power with uncertainties for a given power reading and meter range setting.
    #     """
    #     if not self.fit_params or self.covar is None or self.covar.size == 0:
    #         logger.error("Fit parameters or covariance matrix not loaded.")
    #         return ufloat(v, 0)
    #     order = 2
    #     result = v
    #     name = get_param_name(rng, order)
    #     params_unc = correlated_values([self.fit_params[name] for name in self.fit_params.keys()], self.covar, )
        
    #     while name in self.fit_params:
    #         coeff = params_unc[list(self.fit_params.keys()).index(name)]
    #         result += coeff * (v ** order)
    #         order += 1
    #         name = get_param_name(rng, order)

    #     return result

    @staticmethod
    def estimate_range(v):
        """
        Estimate the power meter range based on the reading.
        """
        rng = int(np.floor(np.log10(v) + 1))
        if v <= 10 ** (rng - 1):
            rng -= 1
        return 10 * (rng + 3)


def extract_counts_unc(filename):
    '''
    Extracts counts with uncertainties from SNSPD run files.
    '''
    with open(filename, 'rb') as file:
        data_dict = pickle.load(file)

    # Ensure all relevant entries in data_dict are NumPy arrays
    Maxpol_Count_Array = np.array(data_dict['Maxpol_Count_Array'])
    Minpol_Count_Array = np.array(data_dict['Minpol_Count_Array'])
    Dark_Count_Array = np.array(data_dict['Dark_Count_Array'])
    Cur_Array = np.array(data_dict['Cur_Array'])

    # Compute the Averagepol_Count_Array
    Light_Count_Array = (Maxpol_Count_Array + Minpol_Count_Array) / 2

    # Compute bias, light, dark, and net with uncertainties
    if Dark_Count_Array.ndim == 1:
        _, plateau_dark_counts = get_plateau(Cur_Array, Dark_Count_Array)
        _, plateau_light_counts = get_plateau(Cur_Array, Light_Count_Array)
        dark = unp.uarray(Dark_Count_Array, Dark_Count_Array*(np.std(plateau_dark_counts))/np.mean(plateau_dark_counts))
        light = unp.uarray(Light_Count_Array, Light_Count_Array*(np.std(plateau_light_counts))/np.mean(plateau_light_counts))
    else:
        dark = unp.uarray(np.mean(Dark_Count_Array, axis=1), np.std(Dark_Count_Array, axis=1))
        light = unp.uarray(np.mean(Light_Count_Array, axis=1), np.std(Light_Count_Array, axis=1))
    net = light - dark

    return np.array(data_dict['Cur_Array']), net


def extract_raw_powers_unc(att_cal_filepath):
    '''
    Extracts powers with uncertainty from the attenuation calibration file.
    '''
    logging.info(f"extract_raw_powers_unc")

    with open(att_cal_filepath, 'rb') as file:
        powers_data = pickle.load(file)

    if not isinstance(powers_data, pd.DataFrame):
        powers_data = pd.DataFrame(powers_data)

    columns = ['Attenuator', 'Attenuation (dB)', 'Range', 'Power Mean', 'Power Std']
    new_powers_data = pd.DataFrame(columns=columns)

    # Extract unique attenuation and range values (assuming they are consistent for each attenuator)
    attval = powers_data['Attenuation (dB)'].iloc[0]
    rng = powers_data['Range'].iloc[0]

    # Process each attenuator
    attenuators_list = [None, 1, 2, 3]
    for attenuator in attenuators_list:
        filtered_data = powers_data[powers_data['Attenuator'] == attenuator]
        all_powers = []

        # Collect all power values for this attenuator
        for powers in filtered_data['Powers']:
            if isinstance(powers, list):
                all_powers.extend(powers)  # Add all values in the list
            else:
                all_powers.append(powers)  # Add single value

        # Calculate mean and standard deviation of the collected powers
        powers_mean = np.mean(all_powers)
        powers_std = np.std(all_powers, ddof=1)  # Use ddof=1 for sample std deviation

        # Append the calculated values to the new dataframe using pd.concat
        new_powers_data = pd.concat([new_powers_data, pd.DataFrame([{
            'Attenuator': attenuator,
            'Attenuation (dB)': attval,
            'Range': rng,
            'Power Mean': powers_mean,
            'Power Std': powers_std,
        }])], ignore_index=True)

    return new_powers_data

def calculate_switch_correction_unc(filepath, correction):
    """
    Computes switching ratio with errors and applies nonlinearity correction if provided.
    
    Parameters:
        filepath (str): Path to the pickle file produced by `optical_switch_calibration`.
        correction (PMCorrectLinearUnc, optional): Object for applying nonlinearity corrections.
        
    Returns:
        ufloat: Switching correction with uncertainty.
    """
    # Load calibration data from the pickle file
    switchdata = pd.read_pickle(filepath)

    # Extract measurement data from the DataFrame
    power_mpm = np.array(switchdata['power_mpm'])
    power_cpm = np.array(switchdata['power_cpm'])
    
    power_mpm = np.array([value for d in power_mpm for value in d.values()])
    power_mpm = power_mpm.reshape(1, -1)
    power_mpm_unc = get_uncertainty(power_mpm)
    
    power_cpm = np.array(switchdata['power_cpm'])  # Convert to NumPy array
    power_cpm = power_cpm.reshape(1, -1)
    power_cpm_unc = get_uncertainty(power_cpm)        

    # Apply nonlinearity corrections if provided
    if correction is not None:
        logger.info("Applying nonlinearity corrections.")
        power_mpm_unc = correction.fix_power_unc(power_mpm_unc, init_rng,)

    switch_correction_unc = power_mpm_unc / power_cpm_unc

    return switch_correction_unc



def compute_efficiency_unc(config):
    filename = config['filename']
    bias, net = extract_counts_unc(filename)
    
    correction = PMCorrectLinearUnc(config['nonlinear_processed_path'])

    switch_correction = calculate_switch_correction_unc(config['switch_file'], correction)

    # att_cal_data = extract_raw_powers_unc(config['attcal_file'])
    att_cal_data = pd.read_pickle(config['attcal_file'])
    atts = [1, 2, 3]

    filtered0_att_cal_data = att_cal_data[att_cal_data['Attenuation (dB)'] == 0]
    power_0 = ufloat(filtered0_att_cal_data['Power Mean'].iloc[0], filtered0_att_cal_data['Power Std'].iloc[0])
    rng_0 = filtered0_att_cal_data['Range'].iloc[0]
    power_0 = correction.nonlinear_power_correction(power_0, rng_0)
    power_atts = []
    for att in atts:
        filteredAtt_att_cal_data = att_cal_data[att_cal_data['Attenuator'] == att]
        power_att = ufloat(filteredAtt_att_cal_data['Power Mean'].iloc[0], filteredAtt_att_cal_data['Power Std'].iloc[0])
        rng_att = filteredAtt_att_cal_data['Range'].iloc[0]
        # Something is (or at least was) going very wrong here
        power_atts.append(correction.nonlinear_power_correction(power_att, rng_att))
        power_atts.append(power_att)

    logging.info(f"Corrected attenuation powers")
    logging.info(f"power_0: {power_0}")
    logging.info(f"power_atts: {power_atts}")
    # Energy per photon
    E = codata.h * codata.c / (wavelength)
    logging.info(f"E: {E}")


    # Calculate expected counts
    counts_expected = unp.nominal_values(power_0) / E
    logging.info(f"counts_expected without attenuation: {counts_expected}")
    for power_att in power_atts:
        counts_expected *= unp.nominal_values(power_att) / unp.nominal_values(power_0)
    logging.info(f"counts_expected post attenuation: {counts_expected}")

    # Apply switch correction and calibration factor
    counts_expected = counts_expected / switch_correction
    counts_expected = counts_expected / ufloat(config['CF'], config['CF_err'])
    logging.info(f"counts_expected post corrections: {counts_expected}")


    # Store results in the configuration dictionary
    config['counts_expected'] = f'{unp.nominal_values(counts_expected)}'
    config['counts_expected_std_err'] = f'{unp.std_devs(counts_expected)}'

    # Log the expected counts
    logger.info(f'counts_expected: {counts_expected}')


    return bias, net/counts_expected, counts_expected


if __name__ == '__main__':
    self_filename = os.path.basename(__file__)
    self_sha1hash = compute_sha1_hash(__file__)

    NIST_pm_calib_path = os.path.join(current_file_dir, 'calibration_power_meter', 'SWN HP81521B 2933G05261.xlsx')
    calib_df = pd.read_excel(NIST_pm_calib_path, sheet_name='Data')
    wl = calib_df['Wav (nm)'].values
    cf_list = calib_df['Cal. Factor'].values
    cf_err_list = calib_df['St. Dev'].values
    cf_err = max(cf_err_list*cf_list)
    cf_interp = interp1d(wl, cf_list, kind='cubic')

    nonlinear_calibration_path = os.path.join(current_file_dir, 'data_sde', '.pkl')
    switch_path = os.path.join(current_file_dir, 'data_sde', '.pkl')
    attcal_path = os.path.join(current_file_dir, 'data_sde', '.pkl')
    fpath = os.path.join(current_file_dir, 'data_sde', '.pkl')

    config = {}
    logger.info(f' Counts data file: {fpath}')
    logger.info(f' Wavelength: {wavelength} nm')

    if os.path.isfile(NIST_pm_calib_path):
        config['CF'] = cf_interp(wavelength)  # NIST power-meter calibration factor
        config['CF_err'] = cf_err  # Standard error
    else:
        config['CF'] = 1.0
        config['CF_er'] = 0.0
    if not os.path.exists(switch_path):
        logger.info(f' No switch file found')
        sys.exit(1)
    else:
        logger.info(f' Switching ratio file: {switch_path}')
    if not os.path.exists(nonlinear_calibration_path):
        logger.info(f' No nonlinearity analysis file found')
        sys.exit(1)
    else:
        logger.info(f' Nonlinearity analysis file: {nonlinear_calibration_path}')
    
    config['filename'] = fpath
    config['file_sha1hash'] = compute_sha1_hash(fpath)
    config['script_filename'] = self_filename
    config['script_sha1hash'] = self_sha1hash
    config['nonlinear_calibration_path'] = nonlinear_calibration_path
    config['nonlinear_calibration_path_sha1hash'] = compute_sha1_hash(nonlinear_calibration_path)
    config['attcal_file'] = attcal_path
    config['attcal_file_sha1hash'] = compute_sha1_hash(attcal_path)
    config['switch_file'] = switch_path
    config['switch_file_sha1hash'] = compute_sha1_hash(switch_path)

    bias, eff, counts_expected = compute_efficiency_unc(config)

    data = {
        "Bias": bias,
        "Efficiency": eff,
        "Counts_Expected": counts_expected,
    }
    df = pd.DataFrame(data)

    # Define the file path for saving the CSV
    csv_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(csv_dir, exist_ok=True)

    csv_filepath = os.path.join(csv_dir, f'final_results__{"{:%Y%m%d-%H%M%S}".format(datetime.now())}.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filepath, index=False)
