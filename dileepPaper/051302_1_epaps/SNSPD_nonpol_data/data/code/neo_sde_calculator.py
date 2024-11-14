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
from scipy.interpolate import interp1d
from uncertainties import ufloat, correlated_values
from uncertainties import unumpy as unp

logger = logging.getLogger(__name__)


def get_uncertainty(rawdata):
    """
    From the rawdata provided make an estimate of the uncertainty for each
    row of data Use the std, but if it is too small, use an estimate based
    on assuming a uniform distribution on a quantizer. This works for ANDO
    power meters because they only have 4 digit resolution on the higher
    ranges.

    """
    # Nfit = rawdata.shape[0]
    # N = rawdata.shape[1]
    std = rawdata.std(axis=1, ddof=1)
    avg = rawdata.mean(axis=1)
    min_unc = np.zeros(avg.shape)

    # Use estimate quantization error as lower bound
    #   this works for the ando power meters we are using to monitor
    min_unc[avg > 1e-9] = 1e-12 * 0.5 / (3**0.5)
    min_unc[avg > 1e-6] = 1e-9 * 0.5 / (3**0.5)
    min_unc[avg > 1e-3] = 1e-6 * 0.5 / (3**0.5)

    # replace std with uncertainty from a uniform distribution in the
    # quantization of the power meter if the std is too small
    unc = np.where(std < min_unc, min_unc, std)
    return unc


def compute_sha1_hash(filename):

    if os.path.isfile(filename):
        stream = open(filename, 'rb')  # Open in binary mode for hashing
        file_content = stream.read()  # Memory intensive on buffer?
        sha1hash = hashlib.sha1(file_content)
        stream.close()
        logger.debug(
            f' SHA1hash({os.path.basename(filename)}): {sha1hash.hexdigest()}')
        return sha1hash.hexdigest()
    else:
        logger.debug(
            f'compute_sha1_hash: File does not exist. Returning None.')
        return None


def find_instrument(filename, name='laser'):
    lines = []
    found = False
    name = name.lower()
    with open(filename) as f:
        for line in f:
            if name in line.lower():
                found = True
            if found:
                ll = line.strip().lstrip('#')
                ll = ll.replace('\t', ' ')
                if ':' in ll:
                    lines.append(ll)
                elif name in ll.lower():
                    found = True
                else:
                    found = False
                if '##' in line:
                    found = False
    logger.debug(f'find_instrument: {name}')
    logger.debug(f'find_instrument: {lines}')
    return lines


def extract_pm_serial_number(filename):
    pm_config = find_instrument(filename, 'pm')
    for line in pm_config:
        if 'slot' in line.lower():
            logger.debug(f'extract_pm_serial_number: {line.strip()}')
            serial_number = line.split(':', 1)[1]
    return serial_number


def find_switch_file(wl, path_regex):
    switch_files = glob.glob(path_regex + str(wl) + 'nm.dat')
    if len(switch_files) > 0:
        basename = os.path.basename(switch_files[0])
    if len(switch_files) > 1:
        logger.info(f' Multiple switch files found at {wl} nm')
        logger.info(f' Using file: {basename}')
    try:
        switch_files[0]
        return switch_files[0]
    except IndexError:
        return None


def find_neo_nonlinear_file(wl, path_regex):
    nonlinear_files = glob.glob(
        path_regex + str(wl) + 'nm.dat.analysis.neo.yaml')
    if len(nonlinear_files) > 0:
        basename = os.path.basename(nonlinear_files[0])
    if len(nonlinear_files) > 1:
        logger.info(f' Multiple nonlinear files found at {wl} nm')
        logger.info(f' Using file: {basename}')
    try:
        nonlinear_files[0]
        return nonlinear_files[0]
    except IndexError:
        return None


class pm_correct_linear_unc:
    def __init__(self, fname=None):
        self._fname = fname
        self.load()

    def load(self):
        # logger.info(f' Trying to load {self._fname}')
        stream = open(self._fname, 'r')
        gen = yaml.load_all(stream, Loader=yaml.Loader)
        yaml_data = {}
        for i, g in enumerate(gen):
            yaml_data[i] = g
        stream.close()
        rng_disc = yaml_data[1]
        fit_json = yaml_data[2]['fit_params']
        self.fit_params = lmfit.Parameters()
        self.fit_params.loads(fit_json)  # lmfit parameters
        self.covar = yaml_data[3]['covar']  # covariance matrix for above
        self.rng_disc = {}  # range discontinuity factor (cumulative)
        for key in rng_disc.keys():
            if key == -10:
                self.rng_disc[key] = ufloat(1, 0)
            else:
                self.rng_disc[key] = ufloat(rng_disc[key][-2],
                                            rng_disc[key][-1])

    def power(self, reading, rng=None):
        if self._fname is None:
            return ufloat(reading, 0)
        else:
            return self.fix_power_unc(reading, rng=rng)

    def fix_power_unc(self, reading, rng=None):
        if rng is None:
            # Try to guess range:
            # add 3 for mW, +1 b/c of floor
            rng = int(np.floor(np.log10(reading)+1))
            if reading <= 10.**(rng-1):
                rng -= 1
            rng = 10*(rng+3)
            logger.debug(
                f'fix_power_unc: estimate rng: {rng}, {reading}')
        out = self.fix_power_nonlinearity_unc(reading, rng)
        # Remember, self.rng_disc is cumulative now
        out = out/self.rng_disc[rng]  # Correct for range discontinuity

        return out

    def fix_power_nonlinearity_unc(self, reading, rng=-10):
        """
        Compute linearized power with uncertainties and covariances
        given the power reading and power meter range setting 'rng'.
        This does not correct for range discontinuities.
        """
        #  assumes params is an lmfit Parameters
        k = 2
        out = reading
        params_unc = correlated_values(
            [self.fit_params[name].value for name in self.fit_params],
            self.covar)
        name = f'b{k-rng*10}'
        while name in self.fit_params:
            coeff = params_unc[list(self.fit_params.keys()).index(name)]
            out += coeff*(reading**k)
            k += 1
            name = f'b{k-rng*10}'
        return out



def extract_counts_unc(filename):
    '''
    Extracts counts with uncertainties from SNSPD run files.
    '''
    # Find number of data aquisitions per bias setting ('daqloop')
    daqloop = 1
    with open(filename, 'r') as fid:
        for line in fid:
            if '\'daqloop\':' in line:
                daqloop = int(line.split('\'daqloop\': ')[1].split(',')[0])
                break
    data = np.loadtxt(filename)
    basename = os.path.basename(filename)
    # Find where the bais voltage repeats, this will be where the data
    # changes from dark counts to light counts
    first_point = (np.where(data[0, 1] == data[daqloop-1:, 1]))
    # first_point is s tuple, the point of interest is the second entry
    # in the first row
    split_point = first_point[0][1] + daqloop - 1
    logger.debug(f'extract_counts_unc: {basename} split point: {split_point}')
    dark_data = data[:split_point, [1, 2, 4]]
    light_data = data[split_point:, [1, 2, 4]]
    if daqloop > 1:
        dark_reshaped = dark_data.reshape(-1, daqloop, 3)
        dark_var = np.var(dark_reshaped[:, :, 2], axis=1)
        dark_data = np.mean(dark_reshaped, axis=1)
        dark_var += dark_data[:, 2]  # Add Poisson variance to systematic
        light_reshaped = light_data.reshape(-1, daqloop, 3)
        light_var = np.var(light_reshaped[:, :, 2], axis=1)
        light_data = np.mean(light_reshaped, axis=1)
        light_var += light_data[:, 2]  # Add Poisson variance to systematic
    else:
        # Just use Poisson variance
        dark_var = dark_data[:, 2]
        light_var = light_data[:, 2]

    # A bool list that finds the biases where the device was latched
    good = light_data[:, 1] < 10e-1
    # If, for some reason, the bais data did not return to < 10e-3
    # at begining of the data run (due to some previous state)
    if not good[0]:
        good[0] = True   # (only happens in some sets of data)

    # sets dark_data and light_data equal to there repective sets of data
    # without the timestamp and treshold
    light_data = light_data[good, :]
    lastbias = min(light_data[-1, 0], dark_data[-1, 0])
    light_data = light_data[light_data[:, 0] <= lastbias, :]
    dark_data = dark_data[dark_data[:, 0] <= lastbias, :]
    dark_var = dark_var[:dark_data.shape[0]]
    light_var = light_var[:dark_data.shape[0]]

    bias = light_data[:, 0]  # a list that is the range of the bias voltage
    dark = unp.uarray(dark_data[:, 2], np.sqrt(dark_var))
    light = unp.uarray(light_data[:, 2], np.sqrt(light_var))
    net = light - dark

    return bias, light, dark, net


def extract_raw_powers_unc(optics_filename):
    '''
    Extracts powers with uncertainty from the attenuation calibration file.
    '''
    caldata = np.loadtxt(optics_filename)
    laser_config = find_instrument(optics_filename, 'laser')
    for line in laser_config:
        if 'wavelength' in line:
            wavelength = float(line.split(':')[-1])
    if wavelength > 1:
        wavelength *= 1e-9
    logger.debug(f'extract_raw_powers_unc: wavelength {wavelength} meters.')

    #  Figure out how many measurements per setting in the cal file
    #     Look at the last column (range)
    n_avg = np.where(np.diff(caldata[:, -1]))[0][0]+1
    rng_cal = caldata[:, -1]
    power_cal = caldata[:, -2]
    power_cal.shape = (-1, n_avg)
    rng_cal.shape = (-1, n_avg)
    rng_cal = rng_cal[:, 0].astype(dtype='int')
    logger.debug(
        f'{power_cal.mean(axis=1)} {power_cal.std(axis=1, ddof=1)}')
    power_cal_mean = power_cal.mean(axis=1)
    power_cal_std = get_uncertainty(power_cal)
    power_cal_unc = unp.uarray(power_cal_mean, power_cal_std)
    power_0 = power_cal_unc[::2]
    power_att = power_cal_unc[1::2]
    power_att = power_att[:-1]
    power_0 = (power_0[:-1]+power_0[1:])/2.
    last_power_0 = (power_cal_unc[-2] + power_cal_unc[-1]) / 2.
    power_0 = np.hstack([power_0, last_power_0])

    return wavelength, power_0, power_att, rng_cal


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
    optics_filename = basename + 'console_cal.dat'
    logger.debug(f'compute_efficiency: optics_filename: {optics_filename}')

    bias, light, dark, net = extract_counts_unc(filename)
    wavelength, power_0, \
        power_att, rng_cal = extract_raw_powers_unc(optics_filename)
    pm_serial_number = extract_pm_serial_number(optics_filename)
    logger.debug(f'attenuation calib: {power_0}, {power_att}')
    logger.info(f' pm serial number: {pm_serial_number}')

    correction = None
    if config['nonlinear_yaml'] is not None:
        nonlinfile = config['nonlinear_yaml']
        correction = pm_correct_linear_unc(nonlinfile)

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


def read_yaml_file(yaml_fname):
    out_yaml = {}
    if os.path.isfile(yaml_fname):
        logger.info(f' Reading file {yaml_fname}')
        stream = open(yaml_fname, 'r')
        gen = yaml.load_all(stream, Loader=yaml.Loader)
        for i, g in enumerate(gen):
            out_yaml[i] = g
        stream.close()
    else:
        logger.info(f' Yaml file {yaml_fname} does not exist.')
    return out_yaml


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    logger.debug('hello debug mode')

    self_filename = os.path.basename(__file__)
    self_sha1hash = compute_sha1_hash(__file__)

    nonlin_path_regex = switch_path_regex = fpath_regex = '.'
    NIST_pm_calib_file = 'calib.xlsx'
    config_yaml = read_yaml_file('neo_config.yaml')
    if 'nonlinearity_folder' in config_yaml[0].keys():
        nonlin_path_regex = config_yaml[0]['nonlinearity_folder']
    if 'switch_calib_folder' in config_yaml[0].keys():
        switch_path_regex = config_yaml[0]['switch_calib_folder']
    if 'counts_folder' in config_yaml[0].keys():
        fpath_regex = config_yaml[0]['counts_folder']
    if 'CF_calib_file' in config_yaml[0].keys():
        NIST_pm_calib_file = config_yaml[0]['CF_calib_file']

    # Calibration factor
    if os.path.isfile(NIST_pm_calib_file):
        logger.info(
            f' Found PM calibration {os.path.basename(NIST_pm_calib_file)}')
        calib_df = pd.read_excel(NIST_pm_calib_file, sheet_name='Data')
        wl_list = calib_df['Wav (nm)'].values
        cf_list = calib_df['Aver'].values
        cf_err_list = calib_df['St. Dev'].values
        cf_err = max(cf_err_list*cf_list)
        cf_interp = interp1d(wl_list, cf_list, kind='cubic')

    switch_path_regex += '/*console_switch_cal_'
    nonlin_path_regex += '/*console_nonlinear_'

    fpath_regex += '/*nm.dat'
    filelist = glob.glob(fpath_regex)

    for fname in filelist:
        config = {}
        logger.info(f' Counts data file: {fname}')
        wl = int(re.search('(.{4})nm', fname).group(1))
        logger.info(f' Wavelength: {wl} nm')
        switch_file = find_switch_file(wl, switch_path_regex)
        if not switch_file:
            logger.info(f' No switch file found for {wl} nm')
            # break
        else:
            logger.info(f' Switching ratio file: {switch_file}')
        nonlinear_file = find_neo_nonlinear_file(wl, nonlin_path_regex)
        if not nonlinear_file:
            logger.info(f' No switch file found for {wl} nm')
            # break
        else:
            logger.info(f' Nonlinearity analysis file: {nonlinear_file}')
        config['filename'] = fname
        config['file_sha1hash'] = compute_sha1_hash(fname)
        config['script_filename'] = self_filename
        config['script_sha1hash'] = self_sha1hash
        config['nonlinear_yaml'] = nonlinear_file
        config['nonlinear_yaml_sha1hash'] = compute_sha1_hash(nonlinear_file)
        config['switch_file'] = switch_file
        config['switch_file_sha1hash'] = compute_sha1_hash(switch_file)
        config['save_results'] = True
        if os.path.isfile(NIST_pm_calib_file):
            config['CF'] = cf_interp(wl)  # NIST power-meter calibration factor
            config['CF_err'] = cf_err  # Standard error
        else:
            config['CF'] = 1.0
            config['CF_er'] = 0.0

        bias, eff, counts_expected = compute_efficiency_unc(config)
