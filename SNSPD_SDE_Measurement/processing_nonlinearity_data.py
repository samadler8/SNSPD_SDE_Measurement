import os

import pickle

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from uncertainties import unumpy as unp

from processing_helpers import *

current_file_dir = Path(__file__).parent


def extract_nonlinearity_data(filepath):
    """
    Extract nonlinearity data from a file and organize it by ranges and attenuation settings.

    Parameters:
        filepath (str): Path to the pickled dataframe file containing the data.

    Returns:
        dict: A dictionary with data organized by range, containing uncertainty arrays.
    """

    # Load the dataframe from the pickle file
    df = pd.read_pickle(filepath)

    # Get unique and sorted ranges from the dataframe
    ranges = df['Range'].unique()

    # Initialize a dictionary to store the results
    d = {}

    # Loop through each range
    for rng in ranges:
        # Filter out data where Attenuation Setting < (-1 * Range + 10)
        filtered_df = df[(df['Range'] == rng) & (abs(df['Attenuation Setting']) >= (abs(rng) - (10 - 2)))]
        d[rng] = {}

        # Get unique and sorted attenuation steps
        taus = filtered_df['Attenuation Step'].unique()
        taus.sort()

        # Filter data for the two attenuation steps
        filtered_df_tau0 = filtered_df[filtered_df['Attenuation Step'] == taus[0]]
        filtered_df_tau1 = filtered_df[filtered_df['Attenuation Step'] == taus[1]]

        # Get unique and sorted attenuation settings
        atts = filtered_df['Attenuation Setting'].unique()
        atts.sort()
        d[rng]['att'] = atts

        # Determine the number of iterations
        N = len(filtered_df['Iteration'].unique())

        # Initialize temporary arrays for storing data
        v_temp = np.empty((len(atts), N), dtype=float)
        vt_temp = np.empty((len(atts), N), dtype=float)

        # Loop through each attenuation setting and populate data
        for i, att in enumerate(atts):
            v_temp[i] = filtered_df_tau0[filtered_df_tau0['Attenuation Setting'] == att]['Power'].values
            vt_temp[i] = filtered_df_tau1[filtered_df_tau1['Attenuation Setting'] == att]['Power'].values

        # Calculate mean and uncertainty for v and vt
        v_avg_temp, v_unc_temp = get_mean_uncertainty(v_temp)
        vt_avg_temp, vt_unc_temp = get_mean_uncertainty(vt_temp)

        # Store the data with uncertainties
        d[rng]['v'] = unp.uarray(v_avg_temp, v_unc_temp)
        d[rng]['vt'] = unp.uarray(vt_avg_temp, vt_unc_temp)

    # Save the processed data to a file
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    nonlinearity_processed_filename = f'nonlinearity_processed__{now_str}.pkl'
    os.makedirs("data_sde", exist_ok=True)
    nonlinearity_processed_filepath = os.path.join(current_file_dir, "data_sde", nonlinearity_processed_filename)
    with open(nonlinearity_processed_filepath, 'wb') as f:
        pickle.dump(d, f)

    return


if __name__ == '__main__':

    nonlinearity_filepath = os.path.join(current_file_dir, 'data_sde', 'nonlinearity_factor_raw_power_meaurements_data_20241210-174441.pkl')
    extract_nonlinearity_data(nonlinearity_filepath)