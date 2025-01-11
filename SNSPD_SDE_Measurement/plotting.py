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
from datetime import datetime

from helpers import *

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

## Plotting Functions
# IV Curve
def plot_IV_curve(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', save_pdf=False):
    df = pd.read_pickle(IV_pickle_filepath)

    ic = get_ic(IV_pickle_filepath)

    # Plot the IV curve
    output_dir = os.path.join(current_file_dir, 'figs')
    os.makedirs(output_dir, exist_ok=True)
    figname = f'SNSPD_IV_Curve__{now_str}'
    figpath = os.path.join(output_dir, figname)
    plt.figure(figsize=(8, 6))
    plt.plot(df['Current'], df['Voltage'], label="IV Curve", color="blue", linewidth=2)

    # Add labels and title
    plt.xlabel("Current (A)", fontsize=14)
    plt.ylabel("Voltage (V)", fontsize=14)
    plt.title(f"SNSPD IV Curve - Critical Current = {round(ic*10**6, 3)} \u03bcA", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    return

# Polarization Sweeps
def plot_polarization_sweep(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), pol_counts_filepath='',):
    with open(pol_counts_filepath, 'rb') as file:
        pol_counts = pickle.load(file)
    coords = np.array([item[0] for item in pol_counts])  # Array of (x, y, z)
    counts = np.array([item[1] for item in pol_counts])  # Array of photon counts
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

    # Save the plot as an image
    os.makedirs('figs', exist_ok=True)
    figpath = os.path.join("figs", f"polarization_sweep__{now_str}.png")
    fig.write_image(figpath)

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

    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(data_filepath)[0])
    figname = f'raw_plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)
    plt.close('all')
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

    plt.title(f'{figname}')
    plt.xlabel('Bias current [uA]')
    plt.ylabel('Counts [per sec]')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close('all')

    return

def plot_temperature_dependence(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), data_filepath='', save_pdf=False):
    df = pd.read_pickle(data_filepath)

    output_dir = os.path.join(current_file_dir, 'pics_temperatureDependence')
    os.makedirs(output_dir, exist_ok=True)
    figname = f'plateau_width_temperature_dependence__{now_str}'
    figpath = os.path.join(output_dir, figname)
    plt.figure(figsize=[20, 10])
    plt.plot(df['temperature'], df['plateau_widths'], '-*', color='k')
    plt.title('Plateau Width Temperature Dependence')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Plateau Width [uA]')
    plt.ylim(bottom=0)  # Set y-axis lower limit to 0
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()

def plot_raw_nonlinearity_data(nonlinearity_data_filepath, save_pdf=False):
    """
    Plot nonlinearity data with log-scaled y-axis, different colors for each range,
    and different markers for 'v' and 'vt'.

    Parameters:
        nonlinearity_data_filepath (str): Path to the pickle file containing nonlinearity data.
    """

    processed_data = extract_nonlinearity_data(nonlinearity_data_filepath)
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
    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(nonlinearity_data_filepath)[0])
    figname = f'raw_plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)

    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()


def plot_fitted_nonlinearity(nonlinearity_data_filepath, nonlinearity_calculation_filepath, save_pdf=False):
    # Load data
    processed_data = extract_nonlinearity_data(nonlinearity_data_filepath)
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

    # Save the plot
    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(nonlinearity_data_filepath)[0])
    figname = f'fitted_{data_filename}'
    figpath = os.path.join(output_dir, figname)
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()



def plot_v_vs_fit_ratio(nonlinearity_data_filepath, nonlinearity_calculation_filepath, save_pdf=False):
    plt.figure(figsize=(12, 6))

    processed_data = extract_nonlinearity_data(nonlinearity_data_filepath)
    ranges = processed_data.keys()

    data = pd.read_pickle(nonlinearity_calculation_filepath)
    fit_params = data["fit_params"]
    covar = data["covar"]
    rng_disc = data["rng_disc"]


    for rng in ranges:
        v_data = unp.nominal_values(processed_data[rng]['v'])

        v_data_fit = nonlinear_power_corrections(fit_params, rng, v_data)
        fit_ratio = v_data / v_data_fit

        plt.plot(v_data, fit_ratio, 'o-', label=f'Fit Ratio Range {rng}')

    plt.xlabel("v (Input Power, dBmW)")
    plt.ylabel("Fit / v")
    plt.title("v vs. Polynomial Fit Ratio")
    plt.legend()
    plt.grid()
    # Save the plot
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(nonlinearity_data_filepath)[0])
    figname = f'fit_ratio_{data_filename}'
    figpath = os.path.join(output_dir, figname)
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()


def plot_switch(optical_switch_filepath, cpm_splice="", save_pdf=False):
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

    # combined = ''.join(power_cpm)
    # numbers = [f'+{num}' for num in combined.replace('\n', '').split('+') if num]
    # power_cpm = np.array([float(num) for num in numbers])

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

    # Save the plot
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(optical_switch_filepath)[0])
    figname = f'plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)

    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close()

    return

# Counts vs Current
def plot_processed_counts_unc(data_filepath='', sde_processed_filepath="", save_pdf=False):
    with open(data_filepath, 'rb') as file:
        data_dict = pickle.load(file)
    sde_processed_df = pd.read_csv(sde_processed_filepath)
    counts_expected = sde_processed_df["Counts_Expected"].iloc[0]
    Processed_Cur_Array = sde_processed_df["Bias"]
    Processed_Eff = sde_processed_df["Efficiency"]
    

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

    output_dir = os.path.join(current_file_dir, 'figs_sde')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(data_filepath)[0])
    figname = f'efficiency_plot_{data_filename}'
    figpath = os.path.join(output_dir, figname)
    plt.close('all')
    plt.figure(figsize = [20,10])
    plt.plot(Cur_Array_uA, counts_expected/np.maximum(unp.nominal_values(Maxpol_Counts - Dark_Counts), 0), 
                 fmt='-*', color='cyan', label='Max Polarization - Dark Counts')
    plt.plot(Cur_Array_uA, counts_expected/np.maximum(unp.nominal_values(Minpol_Counts - Dark_Counts), 0), 
                 fmt='-*', color='red', label='Min Polarization - Dark Counts')
    plt.errorbar(Cur_Array_uA, counts_expected/unp.nominal_values(Dark_Counts), 
                 yerr=unp.std_devs(Dark_Counts), fmt='-*', color='black', 
                 label='Dark Counts')
    plt.errorbar(Processed_Cur_Array, unp.nominal_values(Processed_Eff), 
                 yerr=unp.std_devs(Processed_Eff), fmt='-*', color='green', 
                 label='Average Efficiency')

    plt.title(f'{figname}')
    plt.xlabel('Bias current [uA]')
    plt.ylabel('Counts [per sec]')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close('all')

    return



# %% Main Code Block
if __name__ == '__main__':
    # optical_switch_filepath = os.path.join(current_file_dir, 'data_sde', 'optical_switch_calibration_data_cpm_splice2__20250109-180754.pkl')
    # plot_switch(optical_switch_filepath, cpm_splice=2)

    nonlinearity_data_filepath = os.path.join(current_file_dir, 'data_sde', 'nonlinear_calibration_data__20250109-182606.pkl')
    plot_raw_nonlinearity_data(nonlinearity_data_filepath)
    # nonlinearity_calculation_filepath = os.path.join(current_file_dir, 'data_sde', '.pkl')
    # plot_fitted_nonlinearity(nonlinearity_data_filepath, nonlinearity_calculation_filepath)
    # plot_v_vs_fit_ratio(nonlinearity_data_filepath, nonlinearity_calculation_filepath, )

    # now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    # IV_pickle_filepath = os.path.join(current_file_dir, 'data_sde', 'SK3_IV_curve_data__20250110-122541.pkl')
    # plot_IV_curve(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, save_pdf=False)

    # data_filepath = os.path.join(current_file_dir, "data_sde", "SK3_counts_data_snspd_splice1__20250110-155421.pkl")
    # plot_raw_counts_unc(data_filepath=data_filepath)

    # now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    # plot_polarization_sweep(now_str=now_str, pol_counts_filepath=pol_counts_filepath, save_pdf=False)

    # now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    # data_filepath = os.path.join(current_file_dir, 'data_sde', 'SK3_data_dict__20241212-142132.pkl')
    # plot_min_max_avg_counts_vs_current(now_str=now_str, data_filepath=data_filepath, save_pdf=False)




