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
def plot_IV_curve(now_str="{:%Y:%m:%d-%H:%M:%S}".format(datetime.now()), IV_pickle_filepath='', save_pdf=False):
    df = pd.read_pickle(IV_pickle_filepath)

    ic = get_ic(IV_pickle_filepath)

    # Plot the IV curve
    figname = f'SNSPD_IV_Curve__{now_str}'
    os.makedirs('figs', exist_ok=True)
    figpath = os.path.join("figs", figname)
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
def plot_polarization_sweep(now_str="{:%Y:%m:%d-%H:%M:%S}".format(datetime.now()), pol_counts_filepath='', save_pdf=False):
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
def plot_min_max_avg_counts_vs_current(now_str="{:%Y:%m:%d-%H:%M:%S}".format(datetime.now()), data_filepath='', save_pdf=False):
    with open(data_filepath, 'rb') as file:
        data_dict = pickle.load(file)

    Cur_Array = np.array(data_dict['Cur_Array'])
    Dark_Count_Array = np.array(data_dict['Dark_Count_Array'])
    Maxpol_Count_Array = np.array(data_dict['Maxpol_Count_Array'])
    Minpol_Count_Array = np.array(data_dict['Minpol_Count_Array'])
    Maxpol_Settings = data_dict['Maxpol_Settings']
    Minpol_Settings = data_dict['Minpol_Settings']


    os.makedirs('figs', exist_ok=True)

    filename = f'counts_vs_current_curves__{now_str}'
    figpath = os.path.join(current_file_dir, 'figs_sde', filename)
    plt.close('all')
    plt.figure(figsize = [20,10])
    plt.plot(Cur_Array, Maxpol_Count_Array, '--*', color = 'cyan', label = f'Max Polarization {Maxpol_Settings}', linewidth=0.5)
    plt.plot(Cur_Array, Minpol_Count_Array, '--*', color = 'red', label = f'Min Polarization {Minpol_Settings}', linewidth=0.5)
    plt.plot(Cur_Array, np.maximum(Maxpol_Count_Array - Dark_Count_Array, 0), '-*', color = 'cyan', label = f'Max Polarization - Dark Counts')
    plt.plot(Cur_Array, np.maximum(Minpol_Count_Array - Dark_Count_Array, 0), '-*', color = 'red', label = f'Min Polarization - Dark Counts')
    plt.plot(Cur_Array, Dark_Count_Array, '-*', color = 'black', label = 'Dark Counts')
    plt.plot(Cur_Array, np.maximum((Maxpol_Count_Array + Minpol_Count_Array)/2 - Dark_Count_Array, 0), '-*', color = 'green', label = 'Average Counts - Dark Counts')
    plt.title(f'{filename}')
    plt.xlabel('Bias current [uA]')
    plt.ylabel('Counts [per sec]')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    if save_pdf:
        plt.savefig(f'{figpath}.pdf')
    plt.close('all')

    return

def plot_temperature_dependence(now_str="{:%Y:%m:%d-%H:%M:%S}".format(datetime.now()), data_filepath=''):
    df = pd.read_pickle(data_filepath)

    figpath = 'pics_temperatureDependence/plateau_width_temperature_dependence'
    plt.figure(figsize=[20, 10])
    plt.plot(df['temperature'], df['plateau_widths'], '-*', color='k')
    plt.title('Plateau Width Temperature Dependence')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Plateau Width [uA]')
    plt.ylim(bottom=0)  # Set y-axis lower limit to 0
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    plt.savefig(f'{figpath}.pdf')
    plt.close()

def plot_nonlinearity_data(nonlinearity_data_filepath):
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
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'nonlinearity_processed_data_{now_str}'
    figpath = os.path.join(current_file_dir, 'figs_sde', filename)
    os.makedirs(os.path.join(current_file_dir, 'figs_sde'), exist_ok=True)

    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    plt.savefig(f'{figpath}.pdf')
    plt.close()


def plot_raw_nonlinearity_data_with_fits(nonlinearity_data_filepath, nonlinearity_calculation_filepath):
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
            tv_fit = fit_params['tau'] * P_range(fit_params, rng, v_data)
            tv_fit_dbm = 10 * np.log10(tv_fit / 1e-3)
            ax.plot(values['att'], tv_fit_dbm, color=color, linestyle='-', label=f'Fit tau*v {rng}')

    # Customize plot
    ax.set_title('Nonlinearity Data Fits', fontsize=16)
    ax.set_xlabel('Attenuator1 Setting (-dBmW)', fontsize=14)
    ax.set_ylabel('Power (dBmW)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, which="both")
    ax.legend(fontsize=12)

    # Save the plot
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    figpath = os.path.join(current_file_dir, 'figs_sde', f'nonlinearity_data_with_fits_{now_str}')
    os.makedirs('figs_sde', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    plt.savefig(f'{figpath}.pdf')
    plt.close()



def plot_v_vs_fit_ratio(nonlinearity_data_filepath, nonlinearity_calculation_filepath):
    plt.figure(figsize=(12, 6))

    processed_data = extract_nonlinearity_data(nonlinearity_data_filepath)
    ranges = processed_data.keys()

    data = pd.read_pickle(nonlinearity_calculation_filepath)
    fit_params = data["fit_params"]
    covar = data["covar"]
    rng_disc = data["rng_disc"]

    convert_to_dBmW_before_fitting = 0

    for rng in ranges:
        v_data = unp.nominal_values(processed_data[rng]['v'])
        v_data_dBmW = 10 * np.log10(v_data/1e-3)
        if convert_to_dBmW_before_fitting:
            v_data_fit = P_range(fit_params, rng, v_data_dBmW)
        else:
            v_data_fit = P_range(fit_params, rng, v_data)
            v_data_fit = 10 * np.log10(v_data_fit/1e-3)
        logging.info(f"v_data_fit: {v_data_fit}")
        logging.info(f"v_data_dBmW: {v_data_dBmW}")
        fit_ratio = v_data_fit / v_data_dBmW

        plt.plot(v_data_dBmW, fit_ratio, 'o-', label=f'Fit Ratio Range {rng}')

    plt.xlabel("v (Input Power, dBmW)")
    plt.ylabel("Fit / v")
    plt.title("v vs. Polynomial Fit Ratio")
    plt.legend()
    plt.grid()
    # Save the plot
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'v_vs_fit_ratio_{now_str}'
    figpath = os.path.join(current_file_dir, 'figs_sde', filename)
    os.makedirs(os.path.join(current_file_dir, 'figs_sde'), exist_ok=True)

    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    plt.savefig(f'{figpath}.pdf')
    plt.close()


def plot_switch(optical_switch_filepath):
    # Load calibration data from the pickle file
    switchdata = pd.read_pickle(optical_switch_filepath)

    # Extract measurement data from the DataFrame
    power_mpm = np.array(switchdata['power_mpm'])
    power_cpm = np.array(switchdata['power_cpm'])  # Convert to NumPy array

    power_mpm = power_mpm.reshape(1, -1)  # Ensure 1 row and the necessary number of columns
    power_cpm = power_cpm.reshape(1, -1)

    # Compute mean and standard deviation for each measurement
    power_mpm_unc = get_uncertainty(power_mpm)
    power_cpm_unc = get_uncertainty(power_cpm)

    # Prepare data for plotting
    power_mpm_mean = power_mpm_unc.nominal_value
    power_mpm_std = power_mpm_unc.std_dev
    power_cpm_mean = power_cpm_unc.nominal_value
    power_cpm_std = power_cpm_unc.std_dev
    data = [power_mpm_mean.flatten(), power_cpm_mean.flatten()]
    labels = ['MPM', 'CPM']

    # Create the box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))

    # Add mean and standard deviation as scatter points
    ax.errorbar(
        [1, 2], 
        [power_mpm_mean, power_cpm_mean], 
        yerr=[power_mpm_std, power_cpm_std], 
        fmt='o', 
        color='red', 
        label='Mean ± Std Dev'
    )

    # Customize plot
    ax.set_title('Power Meter Readings (MPM vs CPM)', fontsize=14)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    # Save the plot
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'optical_switch__{now_str}'
    figpath = os.path.join(current_file_dir, 'figs_sde', filename)
    os.makedirs(os.path.join(current_file_dir, 'figs_sde'), exist_ok=True)

    plt.tight_layout()
    plt.savefig(f'{figpath}.png')
    plt.savefig(f'{figpath}.pdf')
    plt.close()

    return




# %% Main Code Block
if __name__ == '__main__':

    # now_str = "{:%Y:%m:%d-%H:%M:%S}".format(datetime.now())
    # plot_IV_curve(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, save_pdf=False)

    # now_str = "{:%Y:%m:%d-%H:%M:%S}".format(datetime.now())
    # plot_polarization_sweep(now_str=now_str, pol_counts_filepath=pol_counts_filepath, save_pdf=False)

    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    data_filepath = os.path.join(current_file_dir, 'data_sde', 'SK3_data_dict__20241212-142132.pkl')
    plot_min_max_avg_counts_vs_current(now_str=now_str, data_filepath=data_filepath, save_pdf=False)

    # nonlinearity_data_filepath = os.path.join(current_file_dir, 'data_sde', 'nonlinearity_factor_raw_power_meaurements_data_20241210-174441.pkl')
    # plot_nonlinearity_data(nonlinearity_data_filepath)
    # nonlinearity_calculation_filepath = os.path.join(current_file_dir, 'data_sde', 'nonlinear_calibration_data__20250102-045010.pkl')
    # plot_raw_nonlinearity_data_with_fits(nonlinearity_data_filepath, nonlinearity_calculation_filepath)
    # plot_v_vs_fit_ratio(nonlinearity_data_filepath, nonlinearity_calculation_filepath, )

    # optical_switch_filepath = os.path.join(current_file_dir, 'data_sde', 'optical_switch_calibration_data.pkl')
    # plot_switch(optical_switch_filepath)
