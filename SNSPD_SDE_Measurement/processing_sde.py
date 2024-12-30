import sys
import os
import glob
import re
import yaml
import lmfit
import scipy.constants as codata
import numpy as np
import pandas as pd
import logging
import hashlib
import datetime
import logging
import pickle

from datetime import datetime

from scipy.interpolate import interp1d
from uncertainties import ufloat, correlated_values
from uncertainties import unumpy as unp

from processing_helpers import *



logger = logging.getLogger(__name__)

class PMCorrectLinearUnc:
    """
    Class for correcting power meter readings using linearization
    and range discontinuity adjustments.
    """
    
    def __init__(self, fname=None):
        """
        Initialize the correction class with a YAML configuration file.
        """
        self._fname = fname
        self.fit_params = None
        self.covar = None
        self.rng_disc = {}
        if self._fname:
            self.load()

    def load(self):
        """
        Load calibration data from a YAML file. This includes:
        - Fit parameters for linearization.
        - Covariance matrix for uncertainties.
        - Range discontinuity factors.
        """
        if not self._fname:
            logger.error("No file provided for loading calibration data.")
            return

        try:
            with open(self._fname, 'r') as stream:
                yaml_data = {i: g for i, g in enumerate(yaml.load_all(stream, Loader=yaml.Loader))}
            
            # Extract data from YAML
            self.rng_disc = {
                key: ufloat(1, 0) if key == -10 else ufloat(value[-2], value[-1])
                for key, value in yaml_data[1].items()
            }
            self.fit_params = lmfit.Parameters()
            self.fit_params.loads(yaml_data[2]['fit_params'])
            self.covar = yaml_data[3]['covar']
        except Exception as e:
            logger.error(f"Failed to load calibration data from {self._fname}: {e}")

    def power(self, reading, rng=None):
        """
        Correct the power reading, applying linearization and range discontinuities.
        If no file is loaded, return the reading as is with zero uncertainty.
        """
        if not self._fname:
            return ufloat(reading, 0)
        return self.fix_power_unc(reading, rng)

    def fix_power_unc(self, reading, rng=None):
        """
        Apply corrections for both nonlinearity and range discontinuity.
        """
        # Estimate range if not provided
        if rng is None:
            rng = self.estimate_range(reading)
            logger.debug(f"Estimated range: {rng} for reading: {reading}")

        # Correct for nonlinearity
        corrected_power = self.fix_power_nonlinearity_unc(reading, rng)
        
        # Adjust for range discontinuity
        return corrected_power / self.rng_disc.get(rng, ufloat(1, 0))

    def fix_power_nonlinearity_unc(self, reading, rng=-10):
        """
        Compute linearized power with uncertainties and covariances
        for a given power reading and meter range setting.
        """
        if not self.fit_params or not self.covar:
            logger.error("Fit parameters or covariance matrix not loaded.")
            return ufloat(reading, 0)

        # Linearize power using polynomial coefficients
        params_unc = correlated_values(
            [self.fit_params[name].value for name in self.fit_params],
            self.covar
        )
        result = reading
        k = 2
        name = f'b{k - rng * 10}'

        while name in self.fit_params:
            coeff = params_unc[list(self.fit_params.keys()).index(name)]
            result += coeff * (reading ** k)
            k += 1
            name = f'b{k - rng * 10}'

        return result

    @staticmethod
    def estimate_range(reading):
        """
        Estimate the power meter range based on the reading.
        """
        rng = int(np.floor(np.log10(reading) + 1))  # Determine the decade of the reading
        if reading <= 10 ** (rng - 1):
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

    # Compute the Averagepol_Count_Array
    Light_Count_Array = (Maxpol_Count_Array + Minpol_Count_Array) / 2

    # Compute bias, light, dark, and net with uncertainties
    if Dark_Count_Array.ndim == 1:
        plateau_dark = get_plateau(Dark_Count_Array)
        plateau_light = get_plateau(Light_Count_Array)
        dark = unp.uarray(Dark_Count_Array, Dark_Count_Array*(np.std(plateau_dark))/np.mean(plateau_dark))
        light = unp.uarray(Light_Count_Array, Light_Count_Array*(np.std(plateau_light))/np.mean(plateau_light))
    else:
        dark = unp.uarray(np.mean(Dark_Count_Array, axis=1), np.std(Dark_Count_Array, axis=1))
        light = unp.uarray(np.mean(Light_Count_Array, axis=1), np.std(Light_Count_Array, axis=1))
    net = light - dark

    return np.array(data_dict['Cur_Array']), net


def extract_raw_powers_unc(att_cal_filepath):
    '''
    Extracts powers with uncertainty from the attenuation calibration file.
    '''

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

        # Append the calculated values to the new dataframe
        new_powers_data = new_powers_data.append({
            'Attenuator': attenuator,
            'Attenuation (dB)': attval,
            'Range': rng,
            'Power Mean': powers_mean,
            'Power Std': powers_std,
        }, ignore_index=True)

    return new_powers_data


def calculate_switch_correction_unc(filename, correction=None):
    '''
    Computes switching ratio with errors and nonlinearity correction.
    '''
    switchdata = np.loadtxt(filename)
    n_avg = np.where(np.diff(switchdata[:, 1]))[0][0]+1
    pm1_power = switchdata[:, -3]  # Monitoring power meter
    pmcal_power = switchdata[:, -2]  # NIST calibrated power meter
    # Reshape array so each row is a measurement
    pm1_power.shape = (-1, n_avg)
    pmcal_power.shape = (-1, n_avg)
    pm1_power = pm1_power[0::2]
    pmcal_power = pmcal_power[1::2]
    pm1_power_mean = pm1_power.mean(axis=1)
    pm1_power_std = get_uncertainty(pm1_power)
    pm1_power_unc = unp.uarray(pm1_power_mean, pm1_power_std)
    pmcal_power_mean = pmcal_power.mean(axis=1)
    pmcal_power_std = get_uncertainty(pmcal_power)
    pmcal_power_unc = unp.uarray(pmcal_power_mean, pmcal_power_std)
    if correction is not None:
        for i in range(len(pm1_power_unc)):
            pm1_power_unc[i] = correction.power(pm1_power_unc[i], -10)
    # Assume pmcal_power is ~100 uW so it does not need correction
    # We aren't measuring nonlinearity of pmcal in this experiment
    try:
        ratio = pm1_power_unc/pmcal_power_unc
    except ValueError:
        ratio = pm1_power_unc[:-1]/pmcal_power_unc
    # Combine statistical and systematic errors assuming independence
    switch_correction_std = np.sqrt(unp.std_devs(ratio.mean())**2 +
                                    unp.nominal_values(ratio).std()**2)
    switch_correction_unc = ufloat(unp.nominal_values(ratio.mean()),
                                   switch_correction_std)
    return switch_correction_unc


def compute_efficiency_unc(config):

    filename = config['filename']
    basename = filename.rsplit(os.sep, 1)
    if len(basename) == 1:
        basename = filename.rsplit('/', 1)
    basename = basename[0]+os.sep+basename[1][0:6]
    att_cal_filepath = 'data_sde/attenuator_calibration_data__20241212-225454.pkl'
    logger.debug(f'compute_efficiency: optics_filename: {att_cal_filepath}')

    bias, net = extract_counts_unc(filename)
    att_cal_data = extract_raw_powers_unc(att_cal_filepath)

    correction = None
    if config['nonlinear_yaml'] is not None:
        nonlinfile = config['nonlinear_yaml']
        correction = PMCorrectLinearUnc(nonlinfile)

        for i, p in enumerate(power_att):
            power_att[i] = correction.power(p, rng_cal[1+2*i])
            logger.debug('compute_eff: power before, power corrected, range')
            logger.debug(f'compute_eff: {p}, {power_att[i]}, {rng_cal[1+2*i]}')
        for i in range(len(power_0)):
            power_in = power_0[i]
            power_0[i] = correction.power(power_0[i], rng_cal[0])
            logger.debug(
                f'compute_eff: power_0 correction: {i}, {power_in/power_0[i]}')

    if config['switch_file'] is not None:
        switchfile = config['switch_file']
        switch_correction = calculate_switch_correction_unc(switchfile,
                                                            correction)
        pass
    else:
        switch_correction = ufloat(1, 0)

    # Energy per photon
    E = codata.h * codata.c / wavelength
    counts_expected = power_0[-1] / E
    for idx, p in enumerate(power_att):
        counts_expected *= power_att[idx] / power_0[idx]
    counts_expected = counts_expected / switch_correction
    counts_expected = counts_expected / ufloat(config['CF'], config['CF_err'])
    config['counts_expected'] = f'{unp.nominal_values(counts_expected)}'
    config['counts_expected_std_err'] = f'{unp.std_devs(counts_expected)}'
    logger.info(f' counts_expected: {counts_expected}')

    if config['save_results']:  # Save efficiency results
        savefile = config['filename']
        savefile = savefile.rsplit('.dat', 1)[0] + '_de_analysis.neo.dat'
        logging.info(f' Saving efficiencies to: {savefile}')

        CF = ufloat(config['CF'], config['CF_err'])
        config['NIST calibration factor'] = f'{CF}'
        config['input_file'] = os.path.basename(config['filename'])
        config['input_file_sha1hash'] = config['file_sha1hash']
        config['nonlinear_yaml'] = os.path.basename(config['nonlinear_yaml'])
        config['switch_file'] = os.path.basename(config['switch_file'])
        del config['CF'], config['CF_err'], config['save_results']
        del config['filename'], config['file_sha1hash']

        header = '#  Calculate SDE \n#\n'
        header += '#  ' + datetime.datetime.now().ctime() + '\n'
        header += '#\n# config used\n'
        y = yaml.dump(config, default_flow_style=False)
        y = '# ' + y
        y = y.replace('\n', '\n# ')
        header += y
        header += "#\n# bias,\t efficiency,\t std. err.\n"

        save_fp = open(savefile, 'w')
        save_fp.write(header)
        save_fp.close()
        save_fp = open(savefile, 'ab')
        data = np.vstack([bias, unp.nominal_values(net/counts_expected),
                          unp.std_devs(net/counts_expected)])
        data = np.array(data.T, dtype='float')
        np.savetxt(save_fp, data, fmt='%2.4f, %f, %f')
        save_fp.flush()
        save_fp.close()

    return bias, net/counts_expected, counts_expected


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    logger.debug('hello debug mode')

    self_filename = os.path.basename(__file__)
    self_sha1hash = compute_sha1_hash(__file__)

    NIST_pm_calib_path = os.path.join('calibration', 'SWN HP81521B 2933G05261.xlsx')
    # Calibration factor
    if os.path.isfile(NIST_pm_calib_path):
        logger.info(
            f' Found PM calibration {os.path.basename(NIST_pm_calib_path)}')
        calib_df = pd.read_excel(NIST_pm_calib_path, sheet_name='Data')
        wl_list = calib_df['Wav (nm)'].values
        cf_list = calib_df['Aver'].values
        cf_err_list = calib_df['St. Dev'].values
        cf_err = max(cf_err_list*cf_list)
        cf_interp = interp1d(wl_list, cf_list, kind='cubic')


    nonlin_path = 'data_sde/nonlinearity_factor_raw_power_meaurements_data_20241210-174441.pkl'
    switch_path = 'data_sde/optical_switch_calibration_data,pkl'
    fpath = 'data_sde/SK3_data_dict__20241212-225454.pkl'


    config = {}
    logger.info(f' Counts data file: {fpath}')
    wl = int(re.search('(.{4})nm', fpath).group(1))
    logger.info(f' Wavelength: {wl} nm')
    if not os.path.exists(switch_path):
        logger.info(f' No switch file found for {wl} nm')
        sys.exit(1)
    else:
        logger.info(f' Switching ratio file: {switch_path}')

    if not os.path.exists(nonlin_path):
        logger.info(f' No switch file found for {wl} nm')
        sys.exit(1)
    else:
        logger.info(f' Nonlinearity analysis file: {nonlin_path}')
    config['filename'] = fpath
    config['file_sha1hash'] = compute_sha1_hash(fpath)
    config['script_filename'] = self_filename
    config['script_sha1hash'] = self_sha1hash
    config['nonlinear_yaml'] = nonlin_path
    config['nonlinear_yaml_sha1hash'] = compute_sha1_hash(nonlin_path)
    config['switch_file'] = switch_path
    config['switch_file_sha1hash'] = compute_sha1_hash(switch_path)
    config['save_results'] = True
    if os.path.isfile(NIST_pm_calib_path):
        config['CF'] = cf_interp(wl)  # NIST power-meter calibration factor
        config['CF_err'] = cf_err  # Standard error
    else:
        config['CF'] = 1.0
        config['CF_er'] = 0.0

    bias, eff, counts_expected = compute_efficiency_unc(config)

    final_results_filepath = os.path.join('data_sde', f'final_results__{"{:%Y%m%d-%H%M%S}".format(datetime.now())}.txt')

    with open(final_results_filepath, 'w') as file:
        file.write(f"Bias: {bias}\n")
        file.write(f"Efficiency: {eff}\n")
        file.write(f"Counts Expected: {counts_expected}\n")
