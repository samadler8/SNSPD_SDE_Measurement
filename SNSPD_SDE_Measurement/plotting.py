# %% imports
import os
import logging

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go

from uncertainties import unumpy as unp

from helpers import *

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("plotting.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

## Plotting Functions
# IV Curve
def plot_IV_curve(IV_pickle_filepath='', save_pdf=False):
    with open(IV_pickle_filepath, 'rb') as file:
        IV_data_dict = pickle.load(file)
    print(IV_data_dict)
    ic = get_ic(IV_pickle_filepath)

    # Plot the IV curve
    plt.figure(figsize=(8, 6))
    plt.plot(IV_data_dict['Cur_Array'], IV_data_dict['Volt_Meas_Array'], label="IV Curve", color="blue", linewidth=2)

    # Add labels and title
    plt.xlabel("Current (A)", fontsize=14)
    plt.ylabel("Voltage (V)", fontsize=14)
    plt.title(f"SNSPD IV Curve - Critical Current = {round(ic*10**6, 3)} \u03bcA", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Show the plot
    plt.tight_layout()

    output_dir = os.path.join(current_file_dir, 'figs')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(IV_pickle_filepath)[0])
    figname = f'plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)

    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()
    return

# Polarization Sweeps
def plot_polarization_sweep(pol_counts_filepath='', save_pdf=False):
    with open(pol_counts_filepath, 'rb') as file:
        pol_counts = pickle.load(file)

    coords = np.array(list(pol_counts.keys()))  # Array of (x, y, z)
    counts = np.array(list(pol_counts.values()))  # Array of photon counts
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Create the 3D volumetric plot
    fig = go.Figure(data=go.Volume(
        x=X, 
        y=Y, 
        z=Z,
        value=counts,
        isomin=counts.min(),
        isomax=counts.max(),
        opacity=0.1,  # Adjust opacity for better visualization
        surface_count=25  # Number of isosurfaces
    ))

    # Customize layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="3D Heatmap of Photon Counts"
    )

    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(pol_counts_filepath)[0])
    figname = f'plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)
    fig.show()
    # These final lines take forever
    # fig.write_image(f'{figpath}.png')
    # if save_pdf:
    #     fig.write_image(f'{figpath}.pdf')
    return

# Counts vs Current
def plot_raw_counts_unc(data_filepath='', save_pdf=False):
    with open(data_filepath, 'rb') as file:
        data_dict = pickle.load(file)

    Cur_Array = np.array(data_dict['Cur_Array'])
    Dark_Count_Array = np.array(data_dict['Dark_Count_Array'])
    Maxpol_Count_Array = np.array(data_dict['Maxpol_Count_Array'])
    Minpol_Count_Array = np.array(data_dict['Minpol_Count_Array'])
    Maxpol_Settings = data_dict['Maxpol_Settings']
    Minpol_Settings = data_dict['Minpol_Settings']

    Cur_Array_uA = Cur_Array * 1e6

    Dark_Counts = get_uncertainty(Dark_Count_Array)
    Maxpol_Counts = get_uncertainty(Maxpol_Count_Array)
    Minpol_Counts = get_uncertainty(Minpol_Count_Array)

    Avg_Counts = (Maxpol_Counts + Minpol_Counts) / 2

    plt.figure(figsize = [20,10])
    plt.errorbar(Cur_Array_uA, unp.nominal_values(Maxpol_Counts), 
                 yerr=unp.std_devs(Maxpol_Counts), fmt='--*', color='cyan', 
                 label=f'Max Polarization {Maxpol_Settings}', linewidth=0.5)
    plt.errorbar(Cur_Array_uA, unp.nominal_values(Minpol_Counts), 
                 yerr=unp.std_devs(Minpol_Counts), fmt='--*', color='red', 
                 label=f'Min Polarization {Minpol_Settings}', linewidth=0.5)
    plt.errorbar(Cur_Array_uA, np.maximum(unp.nominal_values(Maxpol_Counts - Dark_Counts), 0), 
                 yerr=unp.std_devs(Maxpol_Counts - Dark_Counts), fmt='-*', color='cyan', 
                 label='Max Polarization - Dark Counts')
    plt.errorbar(Cur_Array_uA, np.maximum(unp.nominal_values(Minpol_Counts - Dark_Counts), 0), 
                 yerr=unp.std_devs(Minpol_Counts - Dark_Counts), fmt='-*', color='red', 
                 label='Min Polarization - Dark Counts')
    plt.errorbar(Cur_Array_uA, unp.nominal_values(Dark_Counts), 
                 yerr=unp.std_devs(Dark_Counts), fmt='-*', color='black', 
                 label='Dark Counts')
    plt.errorbar(Cur_Array_uA, np.maximum(unp.nominal_values(Avg_Counts - Dark_Counts), 0), 
                 yerr=unp.std_devs(Avg_Counts - Dark_Counts), fmt='-*', color='green', 
                 label='Average Counts - Dark Counts')

    plt.title('Raw Counts per Second Plot')
    plt.xlabel('Bias current [uA]')
    plt.ylabel('Counts [per sec]')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
    plt.tight_layout()

    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(data_filepath)[0])
    figname = f'raw_plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)

    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close('all')
    return

def plot_temperature_dependence(data_filepath='', save_pdf=False):
    df = pd.read_pickle(data_filepath)

    plt.figure(figsize=[20, 10])
    plt.plot(df['temperature'], df['plateau_widths'], '-*', color='k')
    plt.title('Plateau Width Temperature Dependence')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Plateau Width [uA]')
    plt.ylim(bottom=0)  # Set y-axis lower limit to 0
    plt.tight_layout()

    output_dir = os.path.join(current_file_dir, 'figs_td')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(data_filepath)[0])
    figname = f'raw_plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)

    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close('all')
    return

def plot_raw_nonlinearity_data(nonlinearity_data_filepath, filtered=True, save_pdf=False, tau=None):
    """
    Plot nonlinearity data with log-scaled y-axis, different colors for each range,
    and different markers for 'v' and 'vt'.

    Parameters:
        nonlinearity_data_filepath (str): Path to the pickle file containing nonlinearity data.
    """

    processed_data = extract_nonlinearity_data(nonlinearity_data_filepath, filtered=filtered, tau=tau)
    # Define markers and a colormap
    markers = {'v': 'o', 'vt': '^'}  # Circles for 'v' and triangles for 'vt'
    colormap = cm.get_cmap('tab10')  # Use a qualitative colormap
    num_ranges = len(processed_data)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop through each range
    for i, (rng, values) in enumerate(processed_data.items()):
        color = colormap(i / num_ranges)  # Assign a unique color for each range
        
        # Plot 'v' data
        v_data = unp.nominal_values(values['v'])  # Get nominal values for 'v'
        ax.plot(values['att'], v_data, marker=markers['v'], color=color, label=f'Range {rng} - v')
        
        # Plot 'vt' data
        vt_data = unp.nominal_values(values['vt'])  # Get nominal values for 'vt'
        ax.plot(values['att'], vt_data, marker=markers['vt'], color=color, label=f'Range {rng} - vt')
    
    # Customize the plot
    ax.set_title('Nonlinearity Data Visualization', fontsize=16)
    ax.set_xlabel('Attenuator1 Setting (-dBmW)', fontsize=14)
    ax.set_ylabel('Power (W)', fontsize=14)
    ax.set_yscale('log')  # Set log scale for the y-axis
    ax.grid(True, linestyle='--', alpha=0.6, which="both")  # Show grid for both major and minor ticks
    ax.legend(fontsize=12)
    
    # Save the plot
    plt.tight_layout()

    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(nonlinearity_data_filepath)[0])
    figname = f'raw_plot_filtered{int(filtered)}_{data_filename}'
    figpath = os.path.join(output_dir, figname)

    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()


def plot_fitted_nonlinearity(nonlinearity_data_filepath, nonlinearity_calculation_filepath, save_pdf=False, tau=None):
    # Load data
    processed_data = extract_nonlinearity_data(nonlinearity_data_filepath, tau=tau)
    data = pd.read_pickle(nonlinearity_calculation_filepath)
    fit_params = data.get("fit_params", {})
    rng_disc = data.get("rng_disc", {})
    
    if not processed_data or not fit_params:
        raise ValueError("Processed data or fit parameters are missing.")

    # Define markers and colormap
    markers = {'v': 'o', 'vt': '^'}  # Circles for 'v' and triangles for 'vt'
    colormap = cm.get_cmap('tab10')  # Use a qualitative colormap
    num_ranges = len(processed_data)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (rng, values) in enumerate(processed_data.items()):
        color = colormap(i / num_ranges)  # Assign a unique color for each range
        
        # Process 'vt' data
        vt_data = unp.nominal_values(values['vt'])
        vt_unc = unp.std_devs(values['vt'])
        vt_data_dbm = 10 * np.log10(vt_data / 1e-3)
        vt_unc_dbm = (vt_unc / vt_data) * np.log(10) * 10

        ax.plot(values['att'], vt_data_dbm, marker=markers['vt'], color=color, label=f'Range {rng} - vt', linestyle='None')

        # Fit curve
        v_data = unp.nominal_values(values['v'])
        if 'tau' in fit_params:
            tv_fit = fit_params['tau'] * nonlinear_power_corrections(fit_params, rng, v_data)
            tv_fit_dbm = 10 * np.log10(tv_fit / 1e-3)
            ax.plot(values['att'], tv_fit_dbm, color=color, linestyle='-', label=f'Fit tau*v {rng}')

    # Customize plot
    ax.set_title('Fitted Nonlinearity', fontsize=16)
    ax.set_xlabel('Attenuator1 Setting (-dBmW)', fontsize=14)
    ax.set_ylabel('Power (dBmW)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, which="both")
    ax.legend(fontsize=12)

    plt.tight_layout()
    # Save the plot
    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(nonlinearity_data_filepath)[0])
    figname = f'fitted_{data_filename}'
    figpath = os.path.join(output_dir, figname)
    
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()
    return



def plot_v_vs_fit_ratio(nonlinearity_data_filepath, nonlinearity_calculation_filepath, save_pdf=False, tau=None):
    plt.figure(figsize=(12, 6))

    processed_data = extract_nonlinearity_data(nonlinearity_data_filepath, tau=tau)
    ranges = processed_data.keys()

    data = pd.read_pickle(nonlinearity_calculation_filepath)
    fit_params = data["fit_params"]
    covar = data["covar"]
    rng_disc = data["rng_disc"]


    for rng in ranges:
        v_data_unc = processed_data[rng]['v']
        v_data = unp.nominal_values(v_data_unc)

        v_data_fit = nonlinear_power_corrections(fit_params, rng, v_data)
        v_data_fit_unc = nonlinear_power_corrections_unc_plotting(fit_params, covar, rng, v_data_unc)

        fit_ratio = v_data / unp.nominal_values(v_data_fit_unc)
        logger.info(f"fit_ratio: {fit_ratio}")

        v_data_dBm = 10 * np.log10(v_data / 1e-3)
        plt.plot(v_data_dBm, fit_ratio, 'o-', label=f'Fit Ratio Range {rng}')

        

    plt.xlabel("v (Input Power, dBmW)")
    plt.ylabel("Fit / v")
    plt.title("v vs. Polynomial Fit Ratio")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(nonlinearity_data_filepath)[0])
    figname = f'fit_ratio_{data_filename}'
    figpath = os.path.join(output_dir, figname)
    
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()


def plot_switch(optical_switch_filepath, save_pdf=False):
    """
    Plot the calibration data from the optical switch and the ratio of CPM to MPM with uncertainty.

    Parameters:
        optical_switch_filepath (str): Path to the calibration data pickle file.
        save_pdf (bool): Whether to save the plot as a PDF in addition to PNG.
    """
    # Load calibration data from the pickle file
    switchdata = pd.read_pickle(optical_switch_filepath)

    # Extract measurement data from the DataFrame
    power_mpm = np.array(switchdata['power_mpm'])
    power_cpm = np.array(switchdata['power_cpm'])

    # Reshape for consistency (if necessary)
    power_mpm = power_mpm.reshape(1, -1)
    power_cpm = power_cpm.reshape(1, -1)

    # Compute mean and standard deviation for each measurement
    power_mpm_unc = get_uncertainty(power_mpm)[0]
    power_cpm_unc = get_uncertainty(power_cpm)[0]

    # Extract mean and standard deviation
    power_mpm_mean = power_mpm_unc.nominal_value
    power_mpm_std = power_mpm_unc.std_dev
    power_cpm_mean = power_cpm_unc.nominal_value
    power_cpm_std = power_cpm_unc.std_dev

    # Calculate the CPM/MPM ratio and uncertainty
    ratio = power_cpm / power_mpm
    ratio_unc = get_uncertainty(ratio)[0]
    ratio_mean = ratio_unc.nominal_value
    ratio_std = ratio_unc.std_dev

    # Prepare data for the box plot
    data = [[power_mpm_mean], [power_cpm_mean]]  # Wrap in lists to make them sequences
    labels = ['MPM', 'CPM']

    # Create the plot with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Subplot 1: Power readings (box plot with error bars)
    ax = axs[0]
    ax.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))
    ax.errorbar(
        [1, 2], 
        [power_mpm_mean, power_cpm_mean], 
        yerr=[power_mpm_std, power_cpm_std], 
        fmt='o', 
        color='red', 
        label='Mean ± Std Dev'
    )
    ax.set_title('Power Meter Readings (MPM vs CPM)', fontsize=14)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    # Subplot 2: CPM/MPM ratio
    ax = axs[1]
    x = np.arange(1)  # X-axis values (indices)
    ax.errorbar(
        x, 
        ratio_mean, 
        yerr=ratio_std, 
        fmt='o', 
        color='green', 
        ecolor='black', 
        capsize=3, 
        label='Ratio ± Std Dev'
    )
    ax.set_title('CPM/MPM Ratio with Uncertainty', fontsize=14)
    ax.set_xlabel('Measurement Index', fontsize=12)
    ax.set_ylabel('Ratio (CPM/MPM)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(optical_switch_filepath)[0])
    figname = f'plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)

    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()
    return

# Counts vs Current
def plot_processed_counts_unc(data_filepath='', sde_processed_filepath="", save_pdf=False):
    with open(data_filepath, 'rb') as file:
        data_dict = pickle.load(file)
    with open(sde_processed_filepath, 'rb') as file:
        sde_processed = pickle.load(file)

    Bias = sde_processed["Bias"]
    Efficiency_nominal = sde_processed["Efficiency_nominal"]
    Efficiency_stddev = sde_processed["Efficiency_stddev"]
    Counts_Expected_nominal = sde_processed["Counts_Expected_nominal"]
    Counts_Expected_stddev = sde_processed["Counts_Expected_stddev"]

    Cur_Array = np.array(data_dict['Cur_Array'])
    Dark_Count_Array = np.array(data_dict['Dark_Count_Array'])
    Maxpol_Count_Array = np.array(data_dict['Maxpol_Count_Array'])
    Minpol_Count_Array = np.array(data_dict['Minpol_Count_Array'])
    Maxpol_Settings = data_dict['Maxpol_Settings']
    Minpol_Settings = data_dict['Minpol_Settings']

    Cur_Array_uA = Cur_Array * 1e6

    Dark_Counts = get_uncertainty(Dark_Count_Array)
    Maxpol_Counts = get_uncertainty(Maxpol_Count_Array)
    Minpol_Counts = get_uncertainty(Minpol_Count_Array)
    array_length = len(Cur_Array_uA)
    counts_expected_array = unp.uarray(
        np.full(array_length, Counts_Expected_nominal),
        np.full(array_length, Counts_Expected_stddev)
    )

    plt.figure(figsize = [20, 10])
    
    y_data = (Maxpol_Counts - Dark_Counts) / counts_expected_array
    plt.errorbar(
        Cur_Array_uA, 
        unp.nominal_values(y_data),
        yerr=unp.std_devs(y_data), 
        fmt='-*', 
        color='cyan', 
        label=f'Max Polarization - Dark Counts {Maxpol_Settings}'
    )

    y_data = (Minpol_Counts - Dark_Counts) / counts_expected_array
    plt.errorbar(
        Cur_Array_uA, 
        unp.nominal_values(y_data),
        yerr=unp.std_devs(y_data), 
        fmt='-*',
        color='red', 
        label=f'Min Polarization - Dark Counts {Minpol_Settings}'
    )

    # Errorbar for Dark Counts
    y_data = (Dark_Counts) / counts_expected_array
    plt.errorbar(
        Cur_Array_uA, 
        unp.nominal_values(y_data),
        yerr=unp.std_devs(y_data), 
        fmt='-*',
        color='black', 
        label='Dark Counts'
    )

    # Errorbar for Average Efficiency
    plt.errorbar(
        Bias*1e6, 
        Efficiency_nominal, 
        yerr=Efficiency_stddev, 
        fmt='-*', 
        color='green', 
        label='Average Efficiency'
    )
    

    plt.title(f'System Detection Efficiency')
    plt.xlabel('Bias current [uA]')
    plt.ylabel('Efficiency')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
    plt.tight_layout()

    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, sde_processed_filename = os.path.split(os.path.splitext(sde_processed_filepath)[0])
    figname = f'efficiency_plot_{sde_processed_filename}'
    figpath = os.path.join(output_dir, figname)

    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close('all')
    return



# %% Main Code Block
if __name__ == '__main__':
    save_pdf = False
    # optical_switch_filepath = os.path.join(current_file_dir, 'data_sde', 'optical_switch_calibration_data_cpm_splice__20250307-195020.pkl')
    # plot_switch(optical_switch_filepath, save_pdf=save_pdf)

    # tau = None
    # nonlinearity_data_filepath = os.path.join(current_file_dir, 'data_sde', 'nonlinear_calibration_data_tau2.5__20250110-210258.pkl')
    # data_dir = os.path.join(current_file_dir, 'data_sde')
    # taus = [1.5, 2, 2, 2.5, 2.5, 3, 3]
    # nonlinearity_data_filenames = ['nonlinear_calibration_data_tau1.5__20250110-210258.pkl', 'nonlinear_calibration_data_tau2__20250110-210258.pkl', 'nonlinear_calibration_data_tau2__20250114-225648.pkl', 'nonlinear_calibration_data_tau2.5__20250110-210258.pkl', 'nonlinear_calibration_data_tau2.5__20250114-225648.pkl', 'nonlinear_calibration_data_tau3__20250110-210258.pkl', 'nonlinear_calibration_data_tau3__20250114-225648.pkl']
    # nonlinearity_calculation_filenames = ['calculation_nonlinear_calibration_data_tau1.5__20250110-210258.pkl', 'calculation_nonlinear_calibration_data_tau2__20250110-210258.pkl', 'calculation_nonlinear_calibration_data_tau2.5__20250110-210258.pkl', 'calculation_nonlinear_calibration_data_tau3__20250110-210258.pkl']
    # for nonlinearity_data_filename, tau in zip(nonlinearity_data_filenames, taus):
    #     nonlinearity_data_filepath = os.path.join(data_dir, nonlinearity_data_filename)
    #     plot_raw_nonlinearity_data(nonlinearity_data_filepath, filtered=True, save_pdf=save_pdf, tau=tau)

    #     for nonlinearity_calculation_filename in nonlinearity_calculation_filenames:
    #         if nonlinearity_data_filename in nonlinearity_calculation_filename:
    #             nonlinearity_calculation_filepath = os.path.join(data_dir, nonlinearity_calculation_filename)
    #             plot_fitted_nonlinearity(nonlinearity_data_filepath, nonlinearity_calculation_filepath, save_pdf=save_pdf, tau=tau)
    #             plot_v_vs_fit_ratio(nonlinearity_data_filepath, nonlinearity_calculation_filepath, save_pdf=save_pdf, tau=tau)

    # nonlinearity_calculation_filepath = os.path.join(current_file_dir, 'data_sde', 'calculation_v1_nonlinear_calibration_data_tau2.5__20250110-210258.pkl')
    
    

    IV_pickle_filepath = os.path.join(current_file_dir, 'data_sde', 'saaed2um_IV_curve_data__20250701-171155.pkl')
    plot_IV_curve(IV_pickle_filepath=IV_pickle_filepath, save_pdf=save_pdf)
    
    # pol_counts_filepath = os.path.join(current_file_dir, "data_sde", "SK3_pol_data_snspd_splice1connectors__20250116-130752.pkl")
    # plot_polarization_sweep(pol_counts_filepath=pol_counts_filepath, save_pdf=save_pdf)

    # data_filepath = os.path.join(current_file_dir, "data_sde", "SK3_counts_data_snspd_splice1__20250110-155421.pkl")
    # sde_processed_filepath = os.path.join(current_file_dir, 'data_sde', 'final_results_nonlinear_correctionFalse__20250114-095804.pkl')
    data_dir = os.path.join(current_file_dir, 'data_sde')
    data_filenames = [f for f in os.listdir(data_dir) if f.startswith('saaed2um_InGaAs_')]
    data_filenames = [f for f in data_filenames if "counts" in f]
    sde_processed_filenames = [f for f in os.listdir(data_dir) if f.startswith('final_results')]
    for data_filename in data_filenames:
        data_filepath = os.path.join(data_dir, data_filename)
        plot_raw_counts_unc(data_filepath=data_filepath, save_pdf=save_pdf)

        sde_processed_filename_arr = [
            sde_processed_filename
            for sde_processed_filename in sde_processed_filenames
            if data_filename in sde_processed_filename
        ]
        for sde_processed_filename in sde_processed_filename_arr:
            sde_processed_filepath = os.path.join(data_dir, sde_processed_filename)
            plot_processed_counts_unc(data_filepath=data_filepath, sde_processed_filepath=sde_processed_filepath, save_pdf=save_pdf)



