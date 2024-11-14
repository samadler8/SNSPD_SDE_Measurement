import matplotlib.pyplot as plt
import matplotlib.ticker as mtic
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse, Rectangle
import re
import glob
from scipy.interpolate import interp1d
import allantools
import yaml
import copy
import lmfit
from uncertainties import ufloat, correlated_values
from uncertainties import unumpy as unp


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


def P_range(params, rng, v):
    """
    Compute linearized power P
    given the parameters of the polynomial 'params',
    power meter range setting 'rng', and the readings 'v'
    """
    #  assumes params is an lmfit Parameters
    k = 2
    out = v + 0
    name = f'b{k-rng*10}'
    while name in params:
        coeff = params[name].value
        out += coeff*(v**k)
        k += 1
        name = f'b{k-rng*10}'
    return out


def alphalabel(ax, xx=0.05, yy=0.95, fsize=12, mycol='black', alph='(a)'):
    ax.text(xx, yy, alph, fontsize=fsize, color=mycol, weight='bold',
            ha='center', va='center')
    return ax


def fig02(fnum):

    filelist = []
    for ii in range(1, 10):
        filelist.append('data/pulse_traces/SDS0000{}.csv'.format(ii))

    for ii in range(10, 38):
        filelist.append('data/pulse_traces/SDS000{}.csv'.format(ii))

    # jitterfile = 'data/timetagger_jitter_data/october_29_21_1800.dat'

    fluxpaths = ['data/flux_scan_last_2223/*nm_de_analysis.neo.dat',
                 'data/flux_scan_last_2226/*nm_de_analysis.neo.dat',
                 'data/flux_scan_last_2234/*nm_de_analysis.neo.dat']
    fluxfiles = [glob.glob(fluxpaths[0]),
                 glob.glob(fluxpaths[1]),
                 glob.glob(fluxpaths[2])]

    satfile = glob.glob('data/counts_last_2223/*maxpol_1550nm.dat')[0]

    def decay_func(tt, A, tau, tdisp, ydisp):
        return A*np.exp(-(tt-tdisp)/tau)+ydisp

    fsize = 18
    as_scale = 1.0
    msize = 10
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(3, 1, 1, aspect=as_scale*1.5e-3)
    for ii in range(0, fnum-1):
        mydat = np.loadtxt(filelist[ii], delimiter=',', skiprows=12)
        ax1.plot(mydat[:, 0]*1e6-0.35, mydat[:, 1]*1e2,
                 color='crimson', alpha=0.5, label='_nolegend_')
    mydat = np.loadtxt(filelist[fnum-1], delimiter=',', skiprows=12)
    ax1.plot(mydat[:, 0]*1e6-0.35, mydat[:, 1]*1e2,
             color='crimson', alpha=0.8, label='cond.')
    mydat = np.loadtxt(filelist[fnum], delimiter=',', skiprows=12)
    ax1.plot(mydat[:, 0]*1e6-0.1, mydat[:, 1]*1e3+40,
             color='green', label='raw')
    mytimes = mydat[110:500, 0]*1e6-0.1
    myvolts = mydat[110:500, 1]*1e3+40
    popt, popc = curve_fit(decay_func, mytimes, myvolts)
    ax1.plot(mytimes, decay_func(mytimes, *popt), color=(0.5, 0.5, 0.5),
             label='fit', linewidth=8, alpha=0.5)
    ax1.set_xlim(-0.1, 0.35)
    ax1.set_ylim(-30, 100)
    ax1.set_yticks([0, 50])
    ax1.set_xticks([0, 0.1, 0.2, 0.3])
    ax1.set_xlabel('Time ($\mu$s)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.grid()
    leg = ax1.legend(loc='upper right',
                     prop={'weight': 'normal', 'size': fsize},
                     shadow=True, edgecolor='black')
    for line in leg.get_lines():
        line.set_linewidth(5.0)
    ax1.text(0.07, 10, '$t_r$ ={:.3f} $\mu$s'.format(popt[1]),
             fontsize=fsize, color=(0.5, 0.5, 0.5),
             weight='bold', ha='center', va='center')
    ax1 = alphalabel(ax=ax1, xx=-0.07, yy=85, fsize=fsize+4, alph='(a)')

    fullcnts = np.genfromtxt(satfile)
    yscale = 1e5
    daqloop = 1
    with open(satfile) as fid:
        for line in fid:
            if '\'daqloop\':' in line:
                daqloop = int(line.split('\'daqloop\':')[1].split(',')[0])
                fid.close()
                break
    spindx = (np.where(fullcnts[0, 1] == fullcnts[daqloop:, 1]))
    splitpt = spindx[0][1] + daqloop - 1
    biases = fullcnts[:splitpt, 1]
    darks = fullcnts[:splitpt, [1, 2, 4]]
    lights = fullcnts[splitpt:, [1, 2, 4]]
    if daqloop > 1:
        biases_reshaped = biases.reshape(-1, daqloop, 1)
        biases = []
        for sublist in biases_reshaped[:, 1]:
            biases.extend(sublist*10)
        dark_reshaped = darks.reshape(-1, daqloop, 3)
        darks = np.mean(dark_reshaped, axis=1)/yscale
        lights_reshaped = lights.reshape(-1, daqloop, 3)
        lights = np.mean(lights_reshaped, axis=1)/yscale
    minsize = min(len(lights), len(darks)) - 1
    nets = lights[:minsize, 2] - darks[:minsize, 2]
    biases = biases[:minsize]
    lights = lights[:minsize, 2]
    darks = darks[:minsize, 2]
    satanalysisfile = satfile.split('.dat')[0] + '_de_analysis.neo.dat'
    with open(satanalysisfile) as fid:
        for line in fid:
            if 'counts_expected:' in line:
                counts_expected = float(
                    line.split(': ')[1].rstrip('\'\n').lstrip('\''))
                fid.close()
                break

    def get_SDE(net_counts):
        return yscale*net_counts/counts_expected

    def get_raw(SDE):
        return SDE * counts_expected
    err = 0.3

    ax2 = fig.add_subplot(3, 1, 2, aspect=as_scale*4.1e-1)
    color = (0, 0.4, 0)  # 'tab:green'
    ax2.set_xlabel('Bias Current ($\mu$A)')
    ax2.set_ylabel(r'Counts ($\times 10^5$/sec.)')
    ax2.errorbar(biases, darks, np.sqrt(darks/yscale),
                 markersize=msize, fmt='o',
                 label='Dark counts')
    ax2.errorbar(biases, lights, np.sqrt(lights/yscale),
                 markersize=msize, fmt='x',
                 label='Raw counts')
    ax2.errorbar(biases, nets, err*1e-2*nets,
                 markersize=msize+2, fmt='s', markerfacecolor='none',
                 label='SDE', color=color)
    ax2.grid()
    ax2.legend(loc='center right', prop={'weight': 'normal'},
               shadow=True, edgecolor='black')
    ax2b = ax2.secondary_yaxis('right', functions=(get_SDE, get_raw))
    ax2b.set_ylabel('SDE', color=color)
    ax2b.tick_params(axis='y', labelcolor=color)
    ax2 = alphalabel(ax=ax2, xx=3.1, yy=2.25, fsize=fsize+4, alph='(b)')
    # mydat = np.loadtxt(filelist[-1], delimiter=',', skiprows=12)
    # ax2.plot(mydat[:, 0]*1e6-7, mydat[:, 1]*1e3+40,
    #          color='green')
    # ax2.set_xlim(-1, 3)
    # ax2.set_ylim(-30, 100)
    # ax2.set_yticks([0, 50])
    # ax2.set_xlabel('Time ($\mu$s)')
    # ax2.grid()
    # ax2 = alphalabel(ax=ax2, xx=-0.7, yy=85, fsize=20, alph='(c)')

    # def jitter_func(tt, A, tau, tdisp, ydisp):
    #     return A*np.exp(-((tt-tdisp)/tau)**2/2)+ydisp

    # jdat = np.loadtxt(jitterfile, delimiter='\t', skiprows=10)
    # ax3 = fig.add_subplot(4, 1, 3, aspect=as_scale*2.15)
    # ns_bin = 0.004  # HydraHarp nanoseconds per bin
    # indx = 3150  # 12.6 ns / ns_bin (laser rep. rate)
    # folded = np.zeros(indx)  # Laser signal divided by 800 for sync pulse
    # for ii in range(0, indx):
    #     for shift in range(0, 10):  # folding
    #         folded[ii] += sum(jdat[shift*indx+ii, :])
    # tseries = np.linspace(0, indx*ns_bin, indx)
    # ax3.plot(tseries-7, folded*1e-5, linewidth=3)
    # ax3.set_xlim(-5, 5)
    # ax3.set_ylabel('counts (x$10^5$)')
    # ax3.set_xlabel('time (ns)')
    # ax3.grid()
    # popt, popc = curve_fit(jitter_func, tseries-7, folded*1e-5)
    # ax3.annotate(text=' FWHM = {:.2f} ns'.format(2.355*popt[1]),
    #              xy=(-2.355*popt[1]/2+popt[2], 1.375),
    #              xytext=(2.355*popt[1]/2+popt[2], 1.375),
    #              arrowprops=dict(linewidth=4, arrowstyle='<->'))
    # # print(2.355*popt[1])
    # ax3 = alphalabel(ax=ax3, xx=-4.3, yy=2.19, fsize=fsize, alph='(c)')

    def eff(file):
        # print(file)
        stream = open(file, 'r')
        data = np.genfromtxt(stream, dtype=None, delimiter=',')
        # for ii in range(-3, -1):
        #     if(data[-1, 1] < data[ii, 1]):
        #         data = data[:len(data)-1]
        # top_values = np.sort(data[:, 1])
        # top_values = top_values[len(top_values)-4:-1]
        myindx = np.where(data[:, 0] == 0.5)
        effy = data[myindx, 1]
        effy_err = data[myindx, 2]
        dat_filename = re.search(
            '(.+)_de_analysis.neo.dat', file).group(1) + '.dat'
        stream.close()
        fullcnts = np.genfromtxt(dat_filename)
        daqloop = 1
        with open(dat_filename) as fid:
            for line in fid:
                if '\'daqloop\':' in line:
                    daqloop = int(line.split('\'daqloop\':')[1].split(',')[0])
                    fid.close()
                    break
        # finds where the bais voltage repeats, this will be where the data
        # changes from dark counts to light counts
        spindx = (np.where(fullcnts[0, 1] == fullcnts[daqloop:, 1]))
        # b/c first_point is a tuple, the point of interest is the second
        # entry in the first row
        splitpt = spindx[0][1] + daqloop - 1
        darks = fullcnts[:splitpt, [1, 2, 4]]
        lights = fullcnts[splitpt:, [1, 2, 4]]
        if daqloop > 1:
            dark_reshaped = darks.reshape(-1, daqloop, 3)
            darks = np.mean(dark_reshaped, axis=1)
            light_reshaped = lights.reshape(-1, daqloop, 3)
            lights = np.mean(light_reshaped, axis=1)
        minsize = min(len(lights), len(darks))
        nets = lights[:minsize] - darks[:minsize]
        wl = re.search('_(.{4})nm', file).group(1)
        return [int(wl), effy, np.max(nets[myindx]), effy_err]

    ax4 = fig.add_subplot(3, 1, 3, aspect=as_scale*15)
    dets = ['D1', 'D2', 'D3']
    mymarkers = ['o', 'x', '^']
    for det in [0, 1, 2]:
        fluxes = np.array([eff(f) for f in fluxfiles[det]], dtype=object)
        fluxes = fluxes[np.argsort(fluxes[:, 2])]
        ax4.errorbar(fluxes[:, 2]*1e-5/fluxes[:, 1], fluxes[:, 1],
                     fluxes[:, 3],
                     marker=mymarkers[det],
                     label=dets[det], markersize=msize+2)
        # if det > 0:
        #     print([fluxes[0, 2]/fluxes[0, 1], fluxes[4, 2]]/fluxes[4, 1])
        #     print([fluxes[0, 1], fluxes[0, 3], fluxes[4, 1], fluxes[4, 3]])
        # else:
        #     print([fluxes[0, 2]/fluxes[0, 1], fluxes[3, 2]]/fluxes[3, 1])
        #     print([fluxes[0, 1], fluxes[0, 3], fluxes[3, 1], fluxes[3, 3]])
        print(dets[det])
        for ii in range(len(fluxes[:, 1])):
            print(fluxes[ii, 2]*1e-5/fluxes[ii, 1],
                  fluxes[ii, 1][0], fluxes[ii, 3][0])
    ax4.set_xlabel('Photon rate (x$10^5$ per second)')
    ax4.set_ylabel('SDE')
    ax4.set_ylim(0.85, 1)
    ax4.set_xlim(0, 5.2)
    ax4.grid()

    leg = ax4.legend(loc='upper right',
                     prop={'weight': 'normal', 'size': fsize},
                     shadow=True, edgecolor='black')
    ax4 = alphalabel(ax=ax4, xx=0.45, yy=0.98, fsize=fsize+4, alph='(c)')

    fig.tight_layout()
    return fig


def fig03(fwlpath):

    fsize = 24
    msize = 14
    as_scale = 1.0
    err = 0.4
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(16, 10))

    def eff(file, vb):
        stream = open(file, 'r')
        data = np.genfromtxt(stream, dtype=None, delimiter=',')
        try:
            indx = list(data[:, 0]).index(vb)
        except ValueError:
            for ii in range(-3, -1):
                if(data[-1, 1] < data[ii, 1]):
                    data = data[:len(data) - 1]
            top_values = np.sort(data[:, 1])
            top_values = top_values[len(top_values)-4:-1]
            top_err = err
        else:
            top_values = data[indx, 1]
            top_err = data[indx, 2]
        dat_filename = re.search(
            '(.+)_de_analysis.neo.dat', file).group(1) + '.dat'
        stream = open(dat_filename, 'r')
        pol = bool(re.search('maxpol', file))
        stream = open(file, 'r')
        wl = re.search('_(.{4})nm', file).group(1)
        return [int(wl), np.max(top_values), pol, top_err]

    vbias = 0.5
    mymarkers = [['^', 'x'], ['^', 'x'], ['^', 'x']]
    dets = ['D1', 'D2', 'D3']
    vshifts = [0, 0.0, 0.0]
    for detnum in range(3):
        filelist = glob.glob(fwlpath[detnum])
        result = np.array([eff(f, vbias) for f in filelist])
        maxpol = np.array([i for i in result if i[2]])
        minpol = np.array([i for i in result if not i[2]])
        maxpol = maxpol[maxpol[:, 0].argsort()]
        minpol = minpol[minpol[:, 0].argsort()]
        # print([min(maxpol[:, 3]), max(maxpol[:, 3])])
        # print([min(minpol[:, 3]), max(minpol[:, 3])])
        ax = fig.add_subplot(3, 1, detnum+1, aspect=as_scale*2e2)

        ax.errorbar(maxpol[:, 0], maxpol[:, 1]+vshifts[detnum],
                    maxpol[:, 3],  # err*1e-2*maxpol[:, 1],
                    markersize=msize+2, marker=mymarkers[detnum][0],
                    capsize=10,
                    label='max. pol.')
        ax.errorbar(minpol[:, 0], minpol[:, 1]+vshifts[detnum],
                    minpol[:, 3],  # err*1e-2*minpol[:, 1],
                    markersize=msize+2, marker=mymarkers[detnum][1],
                    capsize=10,
                    markerfacecolor='none',
                    label='min. pol.')
        # print(dets[detnum])
        # for ii in range(len(maxpol[:, 0])):
        #     print(maxpol[ii, 0], maxpol[ii, 1], maxpol[ii, 3],
        #           minpol[ii, 1], minpol[ii, 3])
        ax.grid()
        ax.set_ylim([0.89, 0.96])
        ax.yaxis.set_ticks([0.90, 0.92, 0.94])
        if detnum < 2:
            ax.tick_params(axis='x', labelbottom=False)
        if detnum == 1:
            ax.legend(loc='lower center',
                      prop={'weight': 'normal', 'size': fsize},
                      shadow=True, edgecolor='black')
            ax.set_ylabel('SDE')
        ax = alphalabel(ax=ax, xx=1525, yy=0.95, fsize=24,
                        alph=dets[detnum])

    ax.set_xlabel('wavelength (nm)')

    # ax.set_ylim(0.89, 0.99)

    fig.tight_layout()
    return fig


def figS2(fWSink):
    WSi_nk = np.genfromtxt(fWSink)
    ll = WSi_nk[:, 0]
    nn = WSi_nk[:, 1]
    kk = WSi_nk[:, 2]
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(5, 4))
    plt.xlabel('wavelength (nm)', fontweight='bold')
    plt.ylabel('Refractive index of WSi', fontweight='bold')
    plt.plot(ll, kk, linewidth=3, linestyle='--',
             label=r'$\kappa$ (imaginary part)', color='red')
    plt.plot(ll, nn, linewidth=3, label='n (real part)', color='black')

    plt.grid()
    plt.gca().legend(loc='center',  # bbox_to_anchor=(0.4, 0.4),
                     prop={'weight': 'normal'},
                     shadow=True, edgecolor='black', fontsize='medium')
    # plt.xlim(1300, 1700)
    # plt.ylim(5, 6.5)
    fig.tight_layout()
    return fig


def newfigtest2d(fpolscan, fpolcalib, fnum):

    fsize = 18
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(16, 12))
    mypolscan = fpolscan[fnum]
    mypolcalib = fpolcalib[fnum]
    stream = open(mypolscan, 'r')
    dev_data = np.genfromtxt(stream, dtype=None, delimiter='\t')
    stream.close()
    stream = open(mypolcalib, 'r')
    calib_data = np.genfromtxt(stream, dtype=None, delimiter='\t')
    stream.close()

    darks = np.array([])
    lights = np.array([])
    powers = np.array([])
    pmcal = np.array([])
    pm2 = np.array([])

    for ii in range(0, 441):
        darks = np.append(darks, dev_data[ii][2])
        lights = np.append(lights, dev_data[ii][3])
        powers = np.append(powers, dev_data[ii][4])
        pmcal = np.append(pmcal, calib_data[ii][2])
        pm2 = np.append(pm2, calib_data[ii][3])

    pmratio = pmcal/pm2
    lights2d = np.reshape(lights, (21, 21))
    darks2d = np.reshape(darks, (21, 21))
    powers2d = np.reshape(powers, (21, 21))
    pmratio2d = np.reshape(pmratio, (21, 21))

    my2darr = (lights2d-darks2d)*1e-3/powers2d/pmratio2d
    # my2darr = sp.gaussian_filter(my2darr, [0, 0])  # smoothing commented out

    ax0 = fig.add_subplot(1, 3, 1)
    dmap = ax0.imshow((lights2d-darks2d)*1e-5, extent=[0, 360, 360, 0])
    fig.colorbar(dmap, ax=ax0, format='%.2f', fraction=0.046, pad=0.04)
    ax0.set_xlabel(r'2$\epsilon_r$ (deg.)')
    ax0.set_ylabel(r'2$\theta_r$ (deg.)')
    ax0.set_title('(lights - darks)*1e-5')

    ax1 = fig.add_subplot(1, 3, 2)
    emap = ax1.imshow(pmratio2d, extent=[0, 360, 360, 0])
    fig.colorbar(emap, ax=ax1, format='%.2f', fraction=0.046, pad=0.04)
    ax1.set_xlabel(r'2$\epsilon_r$ (deg.)')
    ax1.set_ylabel(r'2$\theta_r$ (deg.)')
    ax1.set_title('pmratio')

    ax2 = fig.add_subplot(1, 3, 3)
    fmap = ax2.imshow(my2darr, extent=[0, 360, 360, 0])
    fig.colorbar(fmap, ax=ax2, format='%.2f', fraction=0.046, pad=0.04)
    ax2.set_xlabel(r'2$\epsilon_r$ (deg.)')
    ax2.set_ylabel(r'2$\theta_r$ (deg.)')
    ax2.set_title('(lights - darks)*1e-3/pmratio')

    polmax = max(np.reshape(my2darr, (441, 1)))
    polmin = min(np.reshape(my2darr, (441, 1)))
    polsen = polmax/polmin

    ax2 = alphalabel(ax=ax2, xx=30, yy=400, fsize=16,
                     alph='{:.3f}'.format(polsen[0]))
    fig.tight_layout()
    return fig


def fig04(fpolscan, fpolcalib):

    fsize = 18
    as_scale = 1.0
    msize = 10
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(3, 1, 1, aspect=as_scale*1.5e-3)
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(16, 14))
    dets = ['D1', 'D2', 'D3']
    for fnum in range(0, 3):
        mypolscan = fpolscan[fnum]
        mypolcalib = fpolcalib[fnum]
        stream = open(mypolscan, 'r')
        dev_data = np.genfromtxt(stream, dtype=None, delimiter='\t')
        stream.close()
        stream = open(mypolcalib, 'r')
        calib_data = np.genfromtxt(stream, dtype=None, delimiter='\t')
        stream.close()

        darks = np.array([])
        lights = np.array([])
        powers = np.array([])
        pmcal = np.array([])
        pm2 = np.array([])

        for ii in range(0, 441):
            darks = np.append(darks, dev_data[ii][2])
            lights = np.append(lights, dev_data[ii][3])
            powers = np.append(powers, dev_data[ii][4])
            pmcal = np.append(pmcal, calib_data[ii][2])
            pm2 = np.append(pm2, calib_data[ii][3])

        pmratio = pmcal/pm2
        lights2d = np.reshape(lights, (21, 21))
        darks2d = np.reshape(darks, (21, 21))
        powers2d = np.reshape(powers, (21, 21))
        pmratio2d = np.reshape(pmratio, (21, 21))

        my2darr = (lights2d-darks2d)*1e-3/powers2d/pmratio2d
        # my2darr = sp.gaussian_filter(my2darr, [0, 0])  # smoothing if desired
        polmax = max(max(np.reshape(my2darr, (441, 1))))
        polmin = min(min(np.reshape(my2darr, (441, 1))))
        polsen = polmax/polmin

        ax = fig.add_subplot(1, 3, fnum+1)
        fmap = ax.imshow(my2darr*100/polmax, extent=[0, 360, 360, 0])
        fig.colorbar(fmap, ax=ax, format='%d', fraction=0.046, pad=0.04,
                     ticks=[98, 99, 100])
        ax.set_xlabel(r'2$\epsilon_r$ (deg.)')
        if fnum == 0:
            ax.set_ylabel(r'2$\theta_r$ (deg.)')
        ax.set_title('{}, pol. sens. {:.3f}'.format(dets[fnum],
                                                    polsen),
                     fontsize=fsize)

    fig.tight_layout()
    return fig


def figSRefl(fRefl, Refl_pos, arrowy):

    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(16, 10))

    xlims = [1450, 1650]
    ylims = [0, 0.6]
    ax = fig.add_subplot(1, 1, 1,
                         aspect=(xlims[1]-xlims[0])/(ylims[1]))
    wafer_xy = [1537, 0.4]
    wafer_height = 0.35
    rect_xy = [1515, 0.22]
    wafer_width = wafer_height*(xlims[1]-xlims[0])/(ylims[1]-ylims[0])

    draw_ellipse = Ellipse(wafer_xy, wafer_width,
                           wafer_height, alpha=0.5)
    draw_rect = Rectangle(rect_xy,
                          wafer_width*0.5, wafer_height*0.05,
                          color='white')
    plt.gcf().gca().add_artist(draw_ellipse)
    plt.gcf().gca().add_artist(draw_rect)
    plt.grid()

    def draw_die(x, y, color):
        return Rectangle((x-wafer_width*2/76.2, y-wafer_height*3/76.2),
                         wafer_width*4/76.2,
                         wafer_height*6/76.2, facecolor=color,
                         edgecolor='black')

    for ii in range(len(fRefl)):
        mydat = np.loadtxt(fRefl[ii], delimiter=',', skiprows=1)
        line, = ax.plot(mydat[:, 0], mydat[:, 1], linewidth=1)
        mydie = draw_die(Refl_pos[ii][0]*wafer_width*4/76.2+wafer_xy[0],
                         Refl_pos[ii][1]*wafer_height*6/76.2+wafer_xy[1],
                         line.get_color())
        plt.gcf().gca().add_artist(mydie)
        plt.annotate(
            '', xy=(arrowy[ii],
                    mydat[np.searchsorted(mydat[:, 0], arrowy[ii]), 1]),
            arrowprops=dict(
                arrowstyle='-|>'),
            xytext=((Refl_pos[ii][0])*wafer_width*4/76.2+wafer_xy[0],
                    (Refl_pos[ii][1]-0.5)*wafer_height*6/76.2+wafer_xy[1]))

    fig.tight_layout()
    plt.xlim(xlims)
    plt.ylim(ylims)
    ax = alphalabel(ax=ax, xx=wafer_xy[0], yy=rect_xy[1]+0.325,
                    fsize=16,
                    alph='76.2 mm wafer')
    ax = alphalabel(ax=ax, xx=wafer_xy[0]+30, yy=rect_xy[1]+0.24,
                    fsize=16,
                    alph='(4 mm x6 mm)\ndies')
    ax.set_ylabel('Reflectance', fontweight='bold')
    ax.set_xlabel('wavelength (nm)', fontweight='bold')

    return fig


def figpolgrid(fpolscan, fpolcalib):

    fsize = 12
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(12, 10))
    mytext = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    for fnum in range(3):
        mypolscan = fpolscan[fnum]
        mypolcalib = fpolcalib[fnum]
        stream = open(mypolscan, 'r')
        dev_data = np.genfromtxt(stream, dtype=None, delimiter='\t')
        stream.close()
        stream = open(mypolcalib, 'r')
        calib_data = np.genfromtxt(stream, dtype=None, delimiter='\t')
        stream.close()

        darks = np.array([])
        lights = np.array([])
        powers = np.array([])
        pmcal = np.array([])
        pm2 = np.array([])

        for ii in range(0, 441):
            darks = np.append(darks, dev_data[ii][2])
            lights = np.append(lights, dev_data[ii][3])
            powers = np.append(powers, dev_data[ii][4])
            pmcal = np.append(pmcal, calib_data[ii][2])
            pm2 = np.append(pm2, calib_data[ii][3])

            pmratio = pmcal/pm2
        lights2d = np.reshape(lights, (21, 21))
        darks2d = np.reshape(darks, (21, 21))
        powers2d = np.reshape(powers, (21, 21))
        pmratio2d = np.reshape(pmratio, (21, 21))

        # my2darr = (lights2d-darks2d)*1e-3/powers2d/pmratio2d
        my2darr = (lights2d-darks2d)*1e-5/pmratio2d

        ax0 = fig.add_subplot(3, 3, 3*fnum+1)
        dmap = ax0.imshow((lights2d-darks2d)*1e-5, extent=[0, 360, 360, 0])
        fig.colorbar(dmap, ax=ax0, format='%.2f', fraction=0.046, pad=0.04)
        if fnum == 2:
            ax0.set_xlabel(r'2$\epsilon_r$ (deg.)')
        ax0.set_ylabel(r'2$\theta_r$ (deg.)')
        if fnum == 0:
            ax0.set_title(r'Counts ($\times 10^5$/sec.)')
        ax0 = alphalabel(ax=ax0, xx=30, yy=30, fsize=16,
                         alph=mytext[3*fnum], mycol='white')

        ax1 = fig.add_subplot(3, 3, 3*fnum+2)
        emap = ax1.imshow(pmratio2d, extent=[0, 360, 360, 0])
        fig.colorbar(emap, ax=ax1, format='%.2f', fraction=0.046, pad=0.04)
        if fnum == 2:
            ax1.set_xlabel(r'2$\epsilon_r$ (deg.)')
        # ax1.set_ylabel(r'2$\theta_r$ (deg.)')
        if fnum == 0:
            ax1.set_title('pmratio')
        ax1 = alphalabel(ax=ax1, xx=30, yy=30, fsize=16,
                         alph=mytext[3*fnum+1], mycol='white')

        ax2 = fig.add_subplot(3, 3, 3*fnum+3)
        fmap = ax2.imshow(my2darr, extent=[0, 360, 360, 0])
        fig.colorbar(fmap, ax=ax2, format='%.2f', fraction=0.046, pad=0.04)
        if fnum == 2:
            ax2.set_xlabel(r'2$\epsilon_r$ (deg.)')
        # ax2.set_ylabel(r'2$\theta_r$ (deg.)')
        if fnum == 0:
            ax2.set_title('Counts/pmratio')
        ax2 = alphalabel(ax=ax2, xx=30, yy=30, fsize=16,
                         alph=mytext[3*fnum+2], mycol='white')

    fig.tight_layout()
    return fig


def powermeter_figs(fnonlinpaths):

    # See 'data/calibration' folder
    wl_list = np.array(range(1470, 1631, 20))
    cf_list = np.array([0.9995, 1.0014, 1.0015, 1.0085,
                        1.0138, 1.0132, 1.0124, 1.0099, 1.0124])
    cf_err = 1e-2*np.array([0.14, 0.11, 0, 0.06, 0.12, 0.07,
                            0.02, 0.02, 0.12])
    cfinterp = interp1d(wl_list, cf_list, kind='cubic')

    fsize = 16
    msize = 8
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(16, 10))
    ax0 = fig.add_subplot(1, 2, 1, aspect=4e3)
    fullwl = np.linspace(1471, 1630, 159)
    fullcf = cfinterp(fullwl)

    ax0.errorbar(wl_list, cf_list, cf_err, marker='o', linestyle='',
                 capsize=10,
                 markersize=msize, label='NIST calibration')
    ax0.plot(fullwl, fullcf, linewidth=2, linestyle='--',
             label='cubic spline fit')
    ax0.set_xlabel('wavelength (nm)')
    ax0.set_ylabel('Calibration factor at 100 $\mu$W')
    ax0.legend(loc='lower right',
               prop={'weight': 'normal', 'size': fsize},
               shadow=True, edgecolor='black')
    ax0.grid()

    ax1 = fig.add_subplot(1, 2, 2, aspect=2.2e2)
    for fnum in range(1):  # len(fnonlinpaths)):
        fname = glob.glob(fnonlinpaths[fnum])[0]
        fp = open(fname, 'r')
        gen = yaml.load_all(fp, Loader=yaml.Loader)
        yaml_data = {}
        for i, g in enumerate(gen):
            yaml_data[i] = g
        fp.close()
        rng_disc = yaml_data[1]
        fit_json = yaml_data[2]['fit_params']
        fit_params = lmfit.Parameters()
        fit_params.loads(fit_json)  # lmfit parameters
        covar = yaml_data[3]['covar']  # covariance matrix for above
        [d, davg, ranges] = extract_nonlinearity_data(
            fname.split('.analysis')[0])
        for rng in ranges[:-3]:  # [-10, -20, -30] dBm
            P_corrected_unc = P_range_unc(fit_params, covar, rng,
                                          davg[rng]['v+unc'])
            P_corrected_unc /= ufloat(rng_disc[rng][-2], rng_disc[rng][-1])
            correction_fact_unc = davg[rng]['v+unc'] / P_corrected_unc
            ax1.errorbar(davg[rng]['v']*1e6,
                         unp.nominal_values(correction_fact_unc),
                         unp.std_devs(correction_fact_unc),
                         linestyle='-', marker='.',
                         label=f'{rng} dBm')

        print(f' {rng}: {correction_fact_unc[-1]}')
        ax1.set_xscale("log", nonpositive='clip')
        ax1.legend(loc='upper left',
                   prop={'weight': 'normal', 'size': fsize},
                   shadow=True, edgecolor='black')
        ax1.set_xlabel('Power reading ($\mu$W)')
        ax1.set_ylabel('Nonlinearity correction factor')
        ax1.grid()

    ax0 = alphalabel(ax=ax0, xx=1470, yy=1.014, fsize=20,
                     alph='(a)')
    ax1 = alphalabel(ax=ax1, xx=90, yy=0.999, fsize=20,
                     alph='(b)')

    fig.tight_layout()
    return fig


def nonlin_figs(fnonlinpaths):

    fsize = 16
    msize = 20
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(16, 10))

    ax0 = fig.add_subplot(1, 2, 1, aspect=5)
    for fnum in range(1):
        fname = glob.glob(fnonlinpaths[fnum])[0]
        fp = open(fname, 'r')
        gen = yaml.load_all(fp, Loader=yaml.Loader)
        yaml_data = {}
        for i, g in enumerate(gen):
            yaml_data[i] = g
        fp.close()
        # rng_disc = yaml_data[1]
        fit_json = yaml_data[2]['fit_params']
        fit_params = lmfit.Parameters()
        fit_params.loads(fit_json)
        [d, davg, ranges] = extract_nonlinearity_data(
            fname.split('.analysis')[0])
        mscales = {-10: 1, -20: 0.7, -30: 0.4}
        for rng in ranges[:-4]:  # [-10, -20] dBm
            ax0.plot(davg[rng]['att'], davg[rng]['v'],
                     fillstyle='none',
                     marker='o', linestyle='none',
                     markersize=msize*mscales[rng],
                     label=f'$V$,  {rng} dBm')
            ax0.plot(davg[rng]['att'], davg[rng]['vt'],
                     fillstyle='none',
                     color=plt.gca().lines[-1].get_color(),
                     marker='^', linestyle='none',
                     markersize=msize*mscales[rng],
                     label=r'$V_{\tau}$, ' + f'{rng} dBm')
    
    ax1 = fig.add_subplot(1, 2, 2, aspect=6.7e2)
    for fnum in range(1):
        fname = glob.glob(fnonlinpaths[fnum])[0]
        fp = open(fname, 'r')
        gen = yaml.load_all(fp, Loader=yaml.Loader)
        yaml_data = {}
        for i, g in enumerate(gen):
            yaml_data[i] = g
        fp.close()
        # rng_disc = yaml_data[1]
        fit_json = yaml_data[2]['fit_params']
        fit_params = lmfit.Parameters()
        fit_params.loads(fit_json)
        [d, davg, ranges] = extract_nonlinearity_data(
            fname.split('.analysis')[0])
        for rng in ranges[:-4]:  # [-10, -20] dBm
            vmax = max(davg[rng]['v'])
            vmin = min(davg[rng]['vt'])
            vspan = np.linspace(vmin, vmax, 100)
            ax1.plot(vspan, vspan/P_range(fit_params, rng, vspan),
                     linewidth=2,
                     label=f'r = {rng} dBm')

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_yticks([1, 1.005])
    plt.minorticks_off()
    ax1.get_yaxis().set_major_formatter(mtic.ScalarFormatter())
    ax1.grid()
    ax1.legend(loc="upper center",
               prop={'weight': 'normal', 'size': fsize},
               shadow=True, edgecolor='black')
    ax1.set_xlabel('Power meter reading, $V$ (W)')
    ax1.set_ylabel('$V / P_r(V)$', labelpad=-20, fontsize=fsize*1.2)
    ax1 = alphalabel(ax=ax1, xx=3e-7, yy=1.0047, fsize=20,
                     alph='(b)')

    ax0.grid()
    ax0.set_yscale("log")
    ax0.legend(loc="lower left",
               prop={'weight': 'normal', 'size': fsize},
               shadow=True, edgecolor='black')
    ax0.set_xlabel('att1 setting (dB)')
    ax0.set_ylabel('Power meter reading (W)')
    ax0 = alphalabel(ax=ax0, xx=22, yy=7e-5, fsize=20,
                     alph='(a)')

    return fig


def fig_switchratio(fswitchlist, foldswitchlist):

    wl_list = []
    sw_corr = []
    sw_err = []
    maxcf_err_rel = 0.0014
    for fname in fswitchlist:
        wl_list.append(int(re.search('_(.{4})nm', fname).group(1)))
        switchdata = np.loadtxt(fname)
        n_avg = np.where(np.diff(switchdata[:, 1]))[0][0]+1
        pm1_power = switchdata[:, -3]
        pmcal_power = switchdata[:, -2]
        pm1_power.shape = (-1, n_avg)
        pmcal_power.shape = (-1, n_avg)
        pm1_power_mean = pm1_power.mean(axis=1)
        pmcal_power_mean = pmcal_power.mean(axis=1)
        pm1_power_mean = pm1_power_mean[0::2]
        # pm1_power_std = pm1_power.std(axis=1)
        # pm1_power_std = pm1_power_std[0::2]
        # pmcal_power_std = pmcal_power.std(axis=1)
        try:
            ratio = pm1_power_mean/pmcal_power_mean[1::2]
            #  print(ratio)
        except ValueError:
            pm1_power_mean = pm1_power_mean[:-1]
            ratio = pm1_power_mean/pmcal_power_mean[1::2]
        # ratio_std = pm1_power_std/pm1_power_mean
        # ratio_std += pmcal_power_std[1::2]/pmcal_power_mean[1::2]
        # ratio_std = ratio*ratio_std
        sw_corr.append(ratio.mean())
        sw_err.append(np.sqrt(ratio.var()+(maxcf_err_rel)**2))
        # sw_err.append(ratio_std.mean())
    wlold_list = []
    swold_corr = []
    swold_err = []
    for fname in foldswitchlist:
        wlold_list.append(int(re.search('_(.{4})nm', fname).group(1)))
        switchdata = np.loadtxt(fname)
        n_avg = np.where(np.diff(switchdata[:, 1]))[0][0]+1
        pm1_power = switchdata[:, -3]
        pmcal_power = switchdata[:, -2]
        pm1_power.shape = (-1, n_avg)
        pmcal_power.shape = (-1, n_avg)
        pm1_power_mean = pm1_power.mean(axis=1)
        pmcal_power_mean = pmcal_power.mean(axis=1)
        pm1_power_mean = pm1_power_mean[0::2]
        # pm1_power_std = pm1_power.std(axis=1)
        # pm1_power_std = pm1_power_std[0::2]
        # pmcal_power_std = pmcal_power.std(axis=1)
        try:
            ratio = pm1_power_mean/pmcal_power_mean[1::2]
            #  print(ratio)
        except ValueError:
            pm1_power_mean = pm1_power_mean[:-1]
            ratio = pm1_power_mean/pmcal_power_mean[1::2]
        # ratio_std = pm1_power_std/pm1_power_mean
        # ratio_std += pmcal_power_std[1::2]/pmcal_power_mean[1::2]
        # ratio_std = ratio*ratio_std
        swold_corr.append(ratio.mean())
        swold_err.append(ratio.std()+maxcf_err_rel*ratio.mean())
        # sw_err.append(ratio_std.mean())
    fsize = 14
    msize = 8
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(8, 5))
    plt.xlabel('wavelength (nm)', fontweight='bold')
    plt.ylabel('Optical switching ratio', fontweight='bold')
    plt.errorbar(wl_list, sw_corr, sw_err,
                 markersize=msize, marker='o', capsize=10,
                 label='Oct. 31, 2021')
    plt.errorbar(wlold_list, swold_corr, swold_err, linestyle='',
                 markersize=msize, marker='^', capsize=10,
                 label='Sept. 19, 2021')
    print(sw_err)

    plt.legend(loc='lower right',
               prop={'weight': 'normal', 'size': fsize},
               shadow=True, edgecolor='black')

    plt.grid()
    return fig


def fig_pwrallan(pwrallanfile):

    pwrdata = np.loadtxt(pwrallanfile, delimiter='\t', skiprows=6)
    timediffs = []
    times = [0.0]
    for ii in range(1, pwrdata.shape[0]):
        timediffs.append(pwrdata[ii, 0]-pwrdata[ii-1, 0])
        times.append(times[-1]+timediffs[-1])

    # print([np.mean(timediffs), np.std(timediffs)])
    sampling_rate = 1.0/np.mean(timediffs)

    # Equilibration time 17 hrs.
    eqindex = int(17*3600*sampling_rate)
    # print(eqindex)

    fsize = 18
    plt.rcParams.update({'font.size': fsize, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(16, 10))
    as_ratio = times[-1]/(max(pwrdata[:, 1])-min(pwrdata[:, 1]))
    ax0 = fig.add_subplot(1, 2, 1, aspect=0.5*as_ratio/3600)
    ax0.plot((times[eqindex:]-times[eqindex])/3600,
             pwrdata[eqindex:, 1])
    ax0.set_xlabel('time (hours)')
    ax0.set_ylabel('Power ($\mu$W)')
    ax0.grid()

    ax0 = alphalabel(ax=ax0, xx=0, yy=101.65, fsize=20,
                     alph='(a)')

    (taus, adevs, aderrs, adn) = allantools.oadev(pwrdata[eqindex:, 1],
                                                  rate=sampling_rate,
                                                  taus=np.logspace(
                                                      0, 4, 50))
    # print(adevs, taus)
    as_ratio = max(taus)/(max(adevs)-min(adevs))
    ax1 = fig.add_subplot(1, 2, 2, aspect=2.8e-7*as_ratio)
    ax1.loglog(taus, adevs, linewidth=2)
    ax1.set_xlabel('averaging time (seconds)')
    ax1.set_ylabel('Allan deviation')
    ax1.grid()

    ax1 = alphalabel(ax=ax1, xx=1e4, yy=4e-3, fsize=20,
                     alph='(b)')

    fig.tight_layout()
    return fig


def compute_nonlin_errors():

    nonlin_pattern = 'data/neo_nonlinear_last/*nm.dat'
    raw_nonlin_files = glob.glob(nonlin_pattern)
    post_nonlin_files = [
        fname + '.analysis.yaml' for fname in raw_nonlin_files]

    nonlin_errors = {}
    for fnum in range(len(raw_nonlin_files)):
        rawfile = raw_nonlin_files[fnum]
        postfile = post_nonlin_files[fnum]
        wl = int(re.search('(.{4})nm', rawfile).group(1))

        fp = open(postfile, 'r')
        gen = yaml.load_all(fp, Loader=yaml.Loader)
        yaml_data = {}
        for i, g in enumerate(gen):
            yaml_data[i] = g
        fp.close()
        rng_yaml = yaml_data[0]
        rawdata = np.loadtxt(rawfile)
        N = (rawdata[:, -3] > 0).nonzero()[0][0]
        badv = (rawdata[:, -2] == 1000)
        rawdata[badv, -2] = -10
        rng = rawdata[[0, -(N+1)], -2]  # list of ranges used
        rng_list = np.arange(rng[0], rng[1]-10, -10).astype(int)
        extra = rawdata.shape[0] % (2*N)
        if extra > 0:
            rawdata = rawdata[:-extra, :]
        d = {}  # reorganize data into dictionary
        for r in rng_list:
            rngdata = {}
            v = rawdata[np.logical_and(
                rawdata[:, -2] == r, rawdata[:, -3] == 0), -1]
            vt = rawdata[np.logical_and(
                rawdata[:, -2] == r, rawdata[:, -3] == 3), -1]
            att = rawdata[np.logical_and(
                rawdata[:, -2] == r, rawdata[:, -3] == 3), -4]
            rngdata['v'] = v.reshape((-1, N)).mean(axis=1)
            rngdata['vt'] = vt.reshape((-1, N)).mean(axis=1)
            rngdata['att'] = att[::N]
            rngdata['v_std'] = v.reshape((-1, N)).std(axis=1)
            rngdata['vt_std'] = vt.reshape((-1, N)).std(axis=1)
            rngdata['v_relerr'] = rngdata['v_std']/rngdata['v']
            rngdata['vt_relerr'] = rngdata['vt_std']/rngdata['vt']
            d[r] = rngdata
            # print([r, d[r]['v'].shape, len(rng_yaml[r])])
            # xlen = min(len(rng_yaml[r]), d[r]['v'].shape[0])

        overlap_N = 12  # Magic number (FIXME?, derive from files)
        wl_relerr = {}
        for r in rng_list[1:]:  # skip range (-10 dB)
            rel_errors = np.zeros(overlap_N)
            abs_var = 0
            for ii in range(overlap_N):
                ratio = d[r+10]['vt'][ii-overlap_N]/d[r]['v'][ii]
                rel_errors[ii] = np.sqrt(
                    d[r+10]['vt_relerr'][ii-overlap_N]**2
                    + d[r]['v_relerr'][ii]**2)
                abs_var = abs_var + (rel_errors[
                    ii]*ratio)**2
            abs_var = abs_var/overlap_N
            abs_std = np.sqrt(abs_var)
            wl_relerr[r] = abs_std/np.mean(rng_yaml[r])

        wl_cumulative = copy.deepcopy(wl_relerr)
        wl_cumulative[-10] = 0  # range (-10 dB) is reference
        for r in rng_list[1:]:
            wl_cumulative[r] = wl_cumulative[r]**2 + wl_cumulative[r+10]
        for r in rng_list[1:]:
            wl_cumulative[r] = np.sqrt(wl_cumulative[r])

        nonlin_errors[wl] = wl_cumulative

        print(wl, nonlin_errors[wl][-30])  # -30 dB is used in experiment

    return nonlin_errors


if __name__ == '__main__':

    # fig = fig02(21)

    fwlpath = ['data/counts_last_2223/*nm_de_analysis.neo.dat',
               'data/counts_last_2226/*nm_de_analysis.neo.dat',
               'data/counts_last_2234/*nm_de_analysis.neo.dat']
    # fig = fig03(fwlpath)

    fpolscan = ['data/pol_scan_2223/15_09_1550nm_polscan.dat',
                'data/pol_scan_2226/16_25_1550nm_polscan.dat',
                'data/pol_scan_2234/18_10_1550nm_polscan.dat']

    fpolcalib = ['data/pol_scan_2223/14_39_1550nm_polcalib.dat',
                 'data/pol_scan_2226/15_55_1550nm_polcalib.dat',
                 'data/pol_scan_2234/16_54_1550nm_polcalib.dat']
    # fig = newfigtest2d(fpolscan, fpolcalib, 2)  # [0, 1, 2]
    # fig = fig04(fpolscan, fpolcalib)
    # fig = figpolgrid(fpolscan, fpolcalib)

    fWSink = 'data/WSi_nk/WSi_midIR_2.2nm_b.dat'
    # fig = figS2(fWSink)

    fRefl = ['data/1550_WSi3_R_spectra/6_0.csv',
             'data/1550_WSi3_R_spectra/4_0.csv',
             'data/1550_WSi3_R_spectra/2_0.csv',
             'data/1550_WSi3_R_spectra/0_4.csv',
             'data/1550_WSi3_R_spectra/0_2.csv',
             'data/1550_WSi3_R_spectra/0_0.csv',
             'data/1550_WSi3_R_spectra/m2_0.csv',
             'data/1550_WSi3_R_spectra/m4_0.csv',
             'data/1550_WSi3_R_spectra/m6_0.csv',
             'data/1550_WSi3_R_spectra/0_m2.csv',
             'data/1550_WSi3_R_spectra/0_m4.csv']
    Refl_pos = [[6, 0], [4, 0], [2, 0], [0, 4], [0, 2], [0, 0],
                [-2, 0], [-4, 0], [-6, 0], [0, -2], [0, -4]]
    arrowy = [1460, 1470, 1590, 1450, 1585, 1520, 1515,
              1510, 1465, 1580, 1475]
    # fig = figSRefl(fRefl, Refl_pos, arrowy)

    # fnonlinpaths = ['data/neo_nonlinear_last/*_1525*yaml',
    #                 'data/neo_nonlinear_last/*_1550*yaml',
    #                 'data/neo_nonlinear_last/*_1575*yaml']
    fnonlinpaths = ['data/neo_nonlinear_last/*_1550*yaml']
    # fig = powermeter_figs(fnonlinpaths)
    # fig = nonlin_figs(fnonlinpaths)

    pwrallanfile = 'data/10_03_1550nm_pwrrecord.dat'
    # fig = fig_pwrallan(pwrallanfile)

    fswitchpath = 'data/neo_switch_calib_run14/*switch_cal*nm.dat'
    foldswitchpath = 'data/neo_run12/*switch_cal*nm.dat'
    fswitchlist = np.sort(glob.glob(fswitchpath))
    foldswitchlist = np.sort(glob.glob(foldswitchpath))
    # fig = fig_switchratio(fswitchlist, foldswitchlist)

    # nonlin_errors = compute_nonlin_errors()

    plt.show()
