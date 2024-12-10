#%% imports
import os

import pickle
import time

import numpy as np
import pandas as pd
import matplotlib as plt
import plotly.graph_objects as go

#%% IV Curve
time_str = time.strftime("%Y%m%d-%H%M%S")
pickle_filepath = os.path.join("data", "IV_curve_data.pkl")
df = pd.read_pickle(pickle_filepath)

# Plot the IV curve
figname = f'SNSPD_IV_Curve_{time_str}'
os.makedirs('figs', exist_ok=True)
figpath = os.path.join("figs", figname)
plt.figure(figsize=(8, 6))
plt.plot(df['Current'], df['Voltage'], label="IV Curve", color="blue", linewidth=2)

# Add labels and title
plt.xlabel("Current (A)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("SNSPD IV Curve", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.savefig(f'{figpath}.png')


#%% Polarization Sweeps
time_str = time.strftime("%Y%m%d-%H%M%S")
pol_counts_filename = "pol_counts.pkl"
pol_counts_filepath = os.path.join("data", pol_counts_filename)
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
output_filepath = os.path.join("figs", f"polarization_sweep_{time_str}.png")
fig.write_image(output_filepath)
print(f"Figure saved to {output_filepath}")


#%% Counts vs Current
time_str = time.strftime("%Y%m%d-%H%M%S")
data_filename = "data_dict.pkl"
data_filepath = os.path.join("data", data_filename)
with open(data_filepath, 'rb') as file:
    data_dict = pickle.load(file)

Cur_Array = data_dict['Cur_Array']
Dark_Count_Array = data_dict['Dark_Count_Array']
Maxpol_Count_Array = data_dict['Maxpol_Count_Array']
Minpol_Count_Array = data_dict['Minpol_Count_Array']
Maxpol_Settings = data_dict['Maxpol_Settings']
Minpol_Settings = data_dict['Minpol_Settings']


os.makedirs('figs', exist_ok=True)

filename = 'counts_vs_current_curves'
figpath = os.path.join('figs', filename)
plt.close('all')
plt.figure(figsize = [20,10])
plt.plot(Cur_Array, Maxpol_Count_Array, '-*', color = 'cyan', label = f'Max Polarization {Maxpol_Settings} Counts')
plt.plot(Cur_Array, Minpol_Count_Array, '-*', color = 'red', label = f'Min Polarization {Minpol_Settings} Counts')
plt.plot(Cur_Array, Dark_Count_Array, '-*', color = 'black', label = 'Dark Counts')
plt.title(f'{filename}')
plt.xlabel('Bias current [uA]')
plt.ylabel('Counts [per sec]')
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
plt.tight_layout()
plt.savefig(f'{figpath}_{time_str}.png')
plt.savefig(f'{figpath}_{time_str}.pdf')

filename = 'final_counts_vs_current'
figpath = os.path.join('figs', filename)
plt.close('all')
plt.figure(figsize = [20,10])
plt.plot(Cur_Array, np.maximum((Maxpol_Count_Array - Minpol_Count_Array)/2 - Dark_Count_Array, 0), '-*', color = 'green', label = 'Counts - Dark Counts')
plt.title(f'{filename}')
plt.xlabel('Bias current [uA]')
plt.ylabel('Counts [per sec]')
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
plt.tight_layout()
plt.savefig(f'{figpath}_{time_str}.png')
plt.savefig(f'{figpath}_{time_str}.pdf')
