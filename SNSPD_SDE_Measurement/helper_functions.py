import pandas as pd

def get_ic(pickle_filepath, threshold=1e-5):
    df = pd.read_pickle(pickle_filepath)
    filtered_df = df[df['Voltage'] > threshold]  # Filter rows where Voltage > threshold
    if not filtered_df.empty:
        ic = filtered_df['Current'].iloc[0]  # Get the first current value
    else:
        ic = None
    return ic