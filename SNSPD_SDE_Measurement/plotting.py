# %% imports
import os

import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from datetime import datetime

from helper_functions import *

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
    figpath = os.path.join('figs', filename)
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

# %% Main Code Block
if __name__ == '__main__':
    # now_str = "{:%Y:%m:%d-%H:%M:%S}".format(datetime.now())
    # print("STARTING: plot_IV_curve")
    # plot_IV_curve(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, save_pdf=False)
    # print("COMPLETED: plot_IV_curve")

    # now_str = "{:%Y:%m:%d-%H:%M:%S}".format(datetime.now())
    # print("STARTING: plot_polarization_sweep")
    # plot_polarization_sweep(now_str=now_str, pol_counts_filepath=pol_counts_filepath, save_pdf=False)
    # print("COMPLETED: plot_polarization_sweep")

    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    print("STARTING: plot_min_max_avg_counts_vs_current")
    plot_min_max_avg_counts_vs_current(now_str=now_str, data_filepath='data/SK3_data_dict__20241212-155949.pkl', save_pdf=False)
    print("COMPLETED: plot_min_max_avg_counts_vs_current")
# %%
