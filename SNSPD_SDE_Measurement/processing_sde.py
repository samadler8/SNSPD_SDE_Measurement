import sys
import os
import logging
import pickle
import json

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
        logging.FileHandler("processing_sde.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

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
        self.rng_disc = None
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

    def nonlinear_power_correction_unc(self, v, rng=None):
        """
        Correct the power reading, applying linearization and range discontinuities.
        Apply corrections for both nonlinearity and range discontinuity.
        Compute linearized power with uncertainties for a given power reading and meter range setting.
        """
        if self.fit_params is None or self.covar is None or self.rng_disc is None:
            logger.warning("Correction not loaded.")
            return v
        
        if rng is None:
            rng = self.estimate_range(v)
            logger.info(f"Estimated range: {rng} for reading: {v}")

        if rng not in self.rng_disc:
            logger.warning(f"Range {rng} not found in rng_disc. Setting to 1 ± 1e-4.")
            # Pull out max rng_disc std_dev
            self.rng_disc[rng] = ufloat(1.0, 1e-4)
            
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

    Maxpol_Count_Array_unc = get_uncertainty(Maxpol_Count_Array)
    Minpol_Count_Array_unc = get_uncertainty(Minpol_Count_Array)
    Dark_Count_Array_unc = get_uncertainty(Dark_Count_Array)

    Net_Count_Array_unc = ((Maxpol_Count_Array_unc + Minpol_Count_Array_unc) / 2) - Dark_Count_Array_unc

    return np.array(data_dict['Cur_Array']), Net_Count_Array_unc


def calculate_att_correction_unc(att_cal_filepath, correction):
    '''
    Extracts powers with uncertainty from the attenuation calibration file.
    '''
    logging.info(f"extract_raw_powers_unc")

    with open(att_cal_filepath, 'rb') as file:
        att_cal_data = pickle.load(file)

    atts = [0, 1, 2]

    filtered0_att_cal_data = att_cal_data[att_cal_data['Attenuation (dB)'] == 0]
    power_0 = np.array(filtered0_att_cal_data['Power Measurement'].iloc[0])
    power_0 = power_0.reshape(1, -1)
    power_0_unc = get_uncertainty(power_0)[0]
    rng_0 = filtered0_att_cal_data['Range'].iloc[0]
    power_0_unc_corrected = correction.nonlinear_power_correction_unc(power_0_unc, rng_0)
    power_atts_unc_corrected = []
    for att in atts:
        filteredAtt_att_cal_data = att_cal_data[att_cal_data['Attenuator'] == att]
        power_att = np.array(filteredAtt_att_cal_data['Power Measurement'].iloc[0])
        power_att = power_att.reshape(1, -1)
        power_att_unc = get_uncertainty(power_att)[0]
        rng_att = filteredAtt_att_cal_data['Range'].iloc[0]
        power_atts_unc_corrected.append(correction.nonlinear_power_correction_unc(power_att_unc, rng_att))

    return power_0_unc_corrected, power_atts_unc_corrected

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
    power_mpm = power_mpm.reshape(1, -1)
    power_mpm_unc = get_uncertainty(power_mpm)
    
    power_cpm = np.array(switchdata['power_cpm'])  # Convert to NumPy array
    power_cpm = power_cpm.reshape(1, -1)
    power_cpm_unc = get_uncertainty(power_cpm)        

    power_mpm_unc = correction.nonlinear_power_correction_unc(power_mpm_unc, init_rng,)

    switch_correction_unc = power_mpm_unc / power_cpm_unc

    return switch_correction_unc



def compute_efficiency_unc(config):
    filename = config['filename']
    
    bias, net = extract_counts_unc(filename)
    
    correction = PMCorrectLinearUnc(config['nonlinear_calculation_path'])
   
    switch_correction = calculate_switch_correction_unc(config['switch_file'], correction)[0]

    power_0_unc_corrected, power_atts_unc_corrected = calculate_att_correction_unc(config['attcal_file'], correction)
    
    
    # Energy per photon
    wavelength = config['wavelength']
    E = codata.h * codata.c / (wavelength)
    logging.info(f"E: {E}")

    # Calculate expected counts
    counts_expected = unp.nominal_values(power_0_unc_corrected) / E
    logging.info(f"counts_expected without attenuation: {counts_expected}")
    for power_att_unc_corrected in power_atts_unc_corrected:
        counts_expected *= unp.nominal_values(power_att_unc_corrected) / unp.nominal_values(power_0_unc_corrected)
    logging.info(f"counts_expected post attenuation: {counts_expected}")

    # Apply switch correction and calibration factor
    logging.info(f"switch_correction: {switch_correction}")
    counts_expected = counts_expected / switch_correction
    counts_expected = counts_expected / ufloat(config['CF'], config['CF_err'])
    logging.info(f"counts_expected post all corrections: {counts_expected}")

    return bias, net/counts_expected, counts_expected


if __name__ == '__main__':
    self_filename = os.path.basename(__file__)
    self_sha1hash = compute_sha1_hash(__file__)

    wavelength = 1566.314e-9 #m

    config = {}

    NIST_pm_calib_path = os.path.join(current_file_dir, 'calibration_power_meter', 'SWN HP81521B 2933G05261.xlsx')
    # nonlinear_calculation_path = os.path.join(current_file_dir, 'data_sde', 'calculation_0_nonlinear_calibration_data_tau2__20250110-210258.pkl')
    nonlinear_calculation_path = None
    switch_path = os.path.join(current_file_dir, 'data_sde', 'optical_switch_calibration_data_cpm_splice2__20250109-180754.pkl')
    attcal_path = os.path.join(current_file_dir, 'data_sde', 'attenuator_calibration_data_attval29__20250111-180558.pkl')
    fpath = os.path.join(current_file_dir, 'data_sde', 'SK3_counts_data_snspd_splice1_attval29__20250111-180558.pkl')
    
    logger.info(f' Counts data file: {fpath}')
    config['wavelength']
    logger.info(f' Wavelength: {wavelength*1e9} nm')

    if not os.path.exists(NIST_pm_calib_path):
        logger.warning(f' No NIST calibrated power meter file found')
        config['CF'] = 1.0
        config['CF_er'] = 0.0
    else:
        logger.info(f' NIST_pm_calib_path: {NIST_pm_calib_path}')

        calib_df = pd.read_excel(NIST_pm_calib_path, sheet_name='Data')
        wl = calib_df['Wav (nm)'].values
        cf_list = calib_df['Cal. Factor'].values
        cf_err_list = calib_df['St. Dev'].values
        cf_err = max(cf_err_list*cf_list)
        cf_interp = interp1d(wl, cf_list, kind='cubic')
        
        config['CF'] = cf_interp(wavelength*1e9)  # NIST power-meter calibration factor
        config['CF_err'] = cf_err  # Standard error
    if not os.path.exists(switch_path):
        logger.error(f' No switch file found')
        sys.exit(1)
    else:
        logger.info(f' Switching ratio file: {switch_path}')
    if nonlinear_calculation_path is None or not os.path.exists(nonlinear_calculation_path):
        logger.warning(f' No nonlinearity analysis file found')
    else:
        logger.info(f' Nonlinearity analysis file: {nonlinear_calculation_path}')
    
    config['filename'] = fpath
    config['file_sha1hash'] = compute_sha1_hash(fpath)
    config['script_filename'] = self_filename
    config['script_sha1hash'] = self_sha1hash
    config['nonlinear_calculation_path'] = nonlinear_calculation_path
    config['nonlinear_calculation_path_sha1hash'] = compute_sha1_hash(nonlinear_calculation_path)
    config['attcal_file'] = attcal_path
    config['attcal_file_sha1hash'] = compute_sha1_hash(attcal_path)
    config['switch_file'] = switch_path
    config['switch_file_sha1hash'] = compute_sha1_hash(switch_path)

    bias, eff, counts_expected = compute_efficiency_unc(config)

    data = {
        "Bias": bias,  # Normal data
        "Efficiency_nominal": unp.nominal_values(eff),  # Extract nominal values
        "Efficiency_stddev": unp.std_devs(eff),         # Extract uncertainties
        "Counts_Expected_nominal": unp.nominal_values(counts_expected),
        "Counts_Expected_stddev": unp.std_devs(counts_expected),
    }

    nonlinear_correction_exists = (nonlinear_calculation_path is not None)

    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'final_results_nonlinear_correction{nonlinear_correction_exists}__{"{:%Y%m%d-%H%M%S}".format(datetime.now())}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    json_filepath = f'{os.path.splitext(filepath)[0]}.json'
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
