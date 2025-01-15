import os
import pickle
import json

from pathlib import Path
current_file_dir = Path(__file__).parent


pol_counts_filepath = os.path.join(current_file_dir, "data_sde", "SK3_pol_data_snspd_splice1__20250110-125128.pkl")

with open(pol_counts_filepath, 'rb') as file:
    pol_data = pickle.load(file)

# Find the tuple with the highest count
maxpol_settings = max(pol_data, key=pol_data.get)
minpol_settings = min(pol_data, key=pol_data.get)

readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
os.makedirs(readable_output_dir, exist_ok=True)
json_filepath = f'{os.path.splitext(pol_counts_filepath)[0]}.json'
with open(json_filepath, 'w') as f:
    json.dump(pol_data, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))