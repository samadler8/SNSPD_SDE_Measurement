import matplotlib.pyplot as plt
import numpy as np
import glob
import lmfit
import os
import re
import yaml
import logging
import copy
import scipy
import hashlib
import datetime
import argparse
from uncertainties import ufloat, correlated_values
from uncertainties import unumpy as unp

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-wl", type=str, default='',
                    help="wavelength (in nm)", dest='wl')
parser.set_defaults(plot=False)
parser.add_argument("--plot", action="store_true",
                    help="plot fit and data from last file processed",
                    dest='plot')

args = parser.parse_args()


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


def extract_nonlinearity_data(filename):

    data = np.loadtxt(filename)
    # column format of text file... there are 7 columns
    # time, 1, monitor power, tau_setting, att_setting, range, power_reading
    #
    # The data is also taken in blocks of N at each setting...
    #
    # At the end of the file, there may be extra data to look for drift
    #

    rngIdx = -2  # Use to be -2, when there was no "splitter" measurements
    tauIdx = -3  # column number which indicates the setting for "tau"
    attIdx = -4  # column for att setting
    powIdx = -1

    # determine N from the data file
    N = (data[:, tauIdx] > 0).nonzero()[0][0]

    # delete extra data at the end of the file that could be used to check
    # for drift
    extra = data.shape[0] % (2*N)
    data = data[:-extra, :]

    ranges = np.unique(data[:, rngIdx])
    ranges.sort()
    ranges = ranges[::-1].astype(int)
    d = {}
    for rng in ranges:
        d[rng] = {}
        d[rng]['v'] = data[
            (data[:, rngIdx] == rng) & (data[:, tauIdx] == 0), powIdx].reshape(
                (-1, N))
        d[rng]['vt'] = data[
            (data[:, rngIdx] == rng) & (data[:, tauIdx] == 3), powIdx].reshape(
                (-1, N))
        d[rng]['att'] = data[
            (data[:, rngIdx] == rng) & (data[:, tauIdx] == 3), attIdx][::N]
        d[rng]['vstd'] = get_uncertainty(d[rng]['v'])
        d[rng]['vtstd'] = get_uncertainty(d[rng]['vt'])
        d[rng]['range'] = rng
        d[rng]['N'] = N
    davg = copy.deepcopy(d)
    for rng in ranges:
        davg[rng]['v'] = davg[rng]['v'].mean(axis=1)
        davg[rng]['vt'] = davg[rng]['vt'].mean(axis=1)
        # v with uncertainties
        davg[rng]['v+unc'] = unp.uarray(davg[rng]['v'], davg[rng]['vstd'])
        # vt with uncertainties
        davg[rng]['vt+unc'] = unp.uarray(davg[rng]['vt'], davg[rng]['vtstd'])

    return d, davg, ranges


def init_params_ranges(N=[], ranges=[]):
    """
    Create a new set of parameters for fitting...
    this will cover all ranges simultaneously
    e.g.
        parameter b102 is the v**2 coefficeint for the -10 range
        parameter b303 is the v**3 coefficient for the -30 range
    """
    params = lmfit.Parameters()
    # using fixed (tau) and unknown attenuator technique
    params.add('tau', value=0.5)
    N_idx = 0
    for rng in ranges:
        for i in range(2, (N[N_idx]+1)):
            params.add(f"b{-rng}{i}", 0)  # , min=-1e6, max=1e6)
        N_idx += 1
    return params


def residuals3(params, davg):
    #  assumes params is an lmfit Parameters
    results = []
    for rng in davg.keys():
        v = davg[rng]['v']
        vt = davg[rng]['vt']
        vunc = davg[rng]['vstd']
        vtunc = davg[rng]['vtstd']
        k = 2
        out = vt - params['tau']*v
        name = f'b{k-rng*10}'
        # print(name)
        while name in params:
            # print(name,params[name])
            out += params[name]*(vt**k - params['tau'] * (v**k))
            k += 1
            name = f'b{k-rng*10}'
        # Estimate uncertainty for out and divide the residual by it.
        # Estimate is rough 1st order estimate
        out /= (vtunc**2 + (params['tau']*vunc)**2)**0.5
        results.append(out)
    results = np.hstack(results)
    return results


def find_poly_fit(davg, ranges, orders=np.array([4, 3, 2, 2, 2, 2])):
    # Polynomial orders should minimize reduced chi-square
    # See methods named 'redchi' and 'optimize_orders' below
    params = init_params_ranges(orders, ranges)
    fit = lmfit.minimize(residuals3, params, method='leastsq', args=(davg,))
    logger.debug(lmfit.fit_report(fit.params))
    return fit


def redchi(N_list, davg, ranges):
    N_list = N_list.astype(int)
    params = init_params_ranges(N_list, ranges)
    fit = lmfit.minimize(residuals3, params, method='leastsq', args=(davg,))
    # logger.debug(fit.redchi)
    return fit.redchi


def optimize_orders(davg, ranges):
    # N_fit = scipy.optimize.differential_evolution(redchi,
    #                                               args=(davg, ranges),
    #                                               bounds=[(1, 6)]*6)
    # N_fit = scipy.optimize.minimize(redchi, np.array([3, 3, 3, 3, 3, 3]),
    #                                 args=(davg, ranges),
    #                                 bounds=[(1, 6)]*6,
    #                                 method='nelder-mead',
    #                                 options={'xatol': 1e-1})
    myparamspace = (slice(1, 5, 1),)*6
    N_fit = scipy.optimize.brute(redchi, myparamspace,
                                 args=(davg, ranges))
    # N_list = np.round(N_fit.x).astype(int)
    # logger.info([N_fit, N_list])
    return N_fit


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
    params_unc = correlated_values(
        [params[name].value for name in params], covar)
    name = f'b{k-rng*10}'
    while name in params:
        # coeff = ufloat(params[name].value, params[name].stderr)
        coeff = params_unc[list(params.keys()).index(name)]
        out += coeff*(v**k)
        # print(coeff*(v**k))
        k += 1
        name = f'b{k-rng*10}'
    return out


def write_yaml_list_to_file(name, out_yaml_list):
    logger.info(' Writing to: %s' % name)
    stream = open(name, 'w')
    yaml.dump_all(out_yaml_list, stream, default_flow_style=True,
                  explicit_start=True)
    stream.close()
    return


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
    logger.debug('hello')

    self_filename = os.path.basename(__file__)
    self_sha1hash = compute_sha1_hash(__file__)

    config_yaml = read_yaml_file('neo_config.yaml')
    if 'nonlinearity_folder' in config_yaml[0].keys():
        fpath_regex = config_yaml[0]['nonlinearity_folder']

    fpath_regex += '/*_console_nonlinear_*'
    fpath_regex = fpath_regex + args.wl + 'nm.dat'
    filelist = glob.glob(fpath_regex)
    filelist.sort()
    output_dir = os.path.dirname(filelist[0])
    ordersfile = os.path.join(output_dir, 'nonlinear_fit_orders.yaml')
    orders_dict = {}
    orders_dict_changed = False
    if os.path.exists(ordersfile):
        stream = open(ordersfile, 'r')
        gen = yaml.load_all(stream, Loader=yaml.Loader)
        for ii, orders in enumerate(gen):
            orders_dict = orders
        stream.close()

    for fname in filelist:
        logger.info(' file: %s' % fname)
        file_sha1hash = compute_sha1_hash(fname)
        logger.info(f' SHA-1 hash: {file_sha1hash}')
        sha1_yaml = {
            'input file': os.path.basename(fname),
            'input file SHA-1 hash': file_sha1hash,
            'analysis script file': self_filename,
            'analysis script file SHA-1 hash': self_sha1hash,
            'Date of analysis': datetime.datetime.now().ctime()}
        output_dir = os.path.dirname(fname)
        logger.debug('output_dir: %s' % output_dir)
        wl = int(re.search('(.{4})nm', fname).group(1))
        [d, davg, ranges] = extract_nonlinearity_data(fname)

        # Find polynomial fit for data within each range
        if wl not in orders_dict.keys():
            orders_dict_changed = True
            logger.info(f' Optimizing polynomial fit orders for {fname}')
            N_list = optimize_orders(davg, ranges)
            N_list = N_list.astype(int).tolist()
            orders_dict[wl] = N_list
        else:
            N_list = orders_dict[wl]
        logger.info(f' Polynomial fit coefficients: {N_list}')
        fit = find_poly_fit(davg, ranges, N_list)
        logger.info(f' fit.redchi = {fit.redchi}')

        # Find range discontinuities
        rng_disc = {}
        rng_disc[-10] = [1, 0]
        rng_disc_factor = ufloat(1, 0)  # Cumulative
        for rng in ranges[1:]:  # [-20, -30, ...]
            overlap = set(davg[rng]['att']).intersection(davg[rng+10]['att'])
            idx1 = [list(davg[rng]['att']).index(x) for x in overlap]
            idx2 = [list(davg[rng+10]['att']).index(x) for x in overlap]
            ratio = P_range_unc(fit.params, fit.covar, rng, np.hstack(
                [davg[rng]['v+unc'][idx1], davg[
                    rng]['vt+unc'][idx1]])) / P_range_unc(
                        fit.params, fit.covar, rng+10, np.hstack(
                            [davg[rng+10]['v+unc'][idx2], davg[
                                rng+10]['vt+unc'][idx2]]))
            # Combine statistical and systematic errors assuming independence
            fact_err = np.sqrt(unp.std_devs(ratio.mean())**2 +
                               unp.nominal_values(ratio).std()**2)
            rng_disc_factor *= ufloat(unp.nominal_values(ratio.mean()),
                                      fact_err)
            logger.info(' Range {}: {:1.6f}, {:1.5f}, {:1.6f}'.format(
                rng, ratio.mean(), unp.nominal_values(ratio).std(),
                        rng_disc_factor))

            # Why does yaml make me do np.array().tolist() for dump_all() ?
            rr_list = [np.array([rr.nominal_value,
                                 np.array(rr.std_dev).tolist()]).tolist()
                       for rr in ratio]
            rr_list.extend([rng_disc_factor.nominal_value,
                            np.array(rng_disc_factor.std_dev).tolist()])
            rng_disc[rng] = rr_list

        rng_disc_yaml = {}
        for key in rng_disc.keys():
            rng_disc_yaml[int(key)] = np.array(rng_disc[key],
                                               dtype=object).tolist()

        outname = os.path.join(output_dir, os.path.basename(fname) +
                               '.analysis.neo.yaml')
        params_yaml = {}
        params_yaml['fit_params'] = fit.params.dumps()
        covar_yaml = {}
        covar_yaml['covar'] = np.array(fit.covar).tolist()
        write_yaml_list_to_file(outname,
                                [sha1_yaml, rng_disc_yaml,
                                 params_yaml, covar_yaml])

    if orders_dict_changed:
        stream = open(ordersfile, 'w')
        yaml.dump_all([orders_dict], stream, default_flow_style=True,
                      explicit_start=True)
        stream.close()

    if args.plot and len(filelist) > 0:
        fsize = 24
        plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
        fig = plt.figure(figsize=(16, 10))
        as_scale = 3e2
        ax = fig.add_subplot(1, 1, 1, aspect=as_scale)
        for rng in ranges[:-3]:  # [-10, -20, -30] dBm
            P_corrected_unc = P_range_unc(fit.params, fit.covar, rng,
                                          davg[rng]['v+unc'])
            P_corrected_unc /= ufloat(rng_disc[rng][-2], rng_disc[rng][-1])
            P_corrected_unc = davg[rng]['v+unc'] / P_corrected_unc
            ax.errorbar(davg[rng]['v']*1e6,
                        unp.nominal_values(P_corrected_unc),
                        unp.std_devs(P_corrected_unc),
                        linestyle='-', marker='.',
                        label=f'{rng} dBm')

        ax.set_xscale("log", nonpositive='clip')
        ax.legend(loc='upper left',
                  prop={'weight': 'normal', 'size': fsize},
                  shadow=True, edgecolor='black')
        ax.set_xlabel('Power reading ($\mu$W)')
        ax.set_ylabel('Nonlinearity correction')
        ax.grid()
        plt.show()
