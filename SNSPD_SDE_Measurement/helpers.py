import pickle

import pandas as pd
         
def get_ic(pickle_filepath, ic_threshold=1e-4):
    df = pd.read_pickle(pickle_filepath)
    filtered_df = df[df['Voltage'] > ic_threshold]  # Filter rows where Voltage > threshold
    if not filtered_df.empty:
        ic = filtered_df['Current'].iloc[0]  # Get the first current value
    else:
        ic = None
    return ic