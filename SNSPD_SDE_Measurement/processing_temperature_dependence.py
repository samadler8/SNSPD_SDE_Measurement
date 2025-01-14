import os
import pickle
from pathlib import Path
import logging
import json

import numpy as np
from datetime import datetime

from processing_helpers import *

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("processing_temperature_dependence.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

def get_temperature_at_time(timestamps, temperature_filepath):
    temperatures = np.empty(len(timestamps), dtype=float)

    temperature_dict = {}
    with open(temperature_filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            time, temperature = parts[0].strip(), parts[-3].strip()
            temperature_dict[time] = float(temperature) if temperature != 'nan' else None
    sorted_times = sorted(temperature_dict.keys())

    for i, timestamp in enumerate(timestamps):
        dt = datetime.strptime(timestamp, '%Y%m%d-%H%M%S')

        prev_time, next_time = None, None

        for t in sorted_times:
            t_dt = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
            if t_dt <= dt:
                prev_time = t
            elif t_dt > dt and prev_time:
                next_time = t
                break

        if not prev_time or not next_time:
            temperatures[i] = None
            break

        prev_temp, next_temp = temperature_dict[prev_time], temperature_dict[next_time]
        if prev_temp is None or next_temp is None:
            temperatures[i] = None
            break

        prev_dt = datetime.strptime(prev_time, '%Y-%m-%d %H:%M:%S.%f')
        next_dt = datetime.strptime(next_time, '%Y-%m-%d %H:%M:%S.%f')
        total_seconds = (next_dt - prev_dt).total_seconds()
        elapsed_seconds = (dt - prev_dt).total_seconds()
        temperature = prev_temp + (next_temp - prev_temp) * (elapsed_seconds / total_seconds)

    return temperatures


if __name__ == "__main__":
    temperature_filepath = os.path.join(current_file_dir, 'data_temperatureDependence', 'CTCLog 102424_15-09.txt')
    time_counts_data_filepath = os.path.join(current_file_dir, 'data_temperatureDependence', ',pkl')

    with open(time_counts_data_filepath, "wb") as file:
        time_counts_data_dict = pickle.load(file)

    temperatures = get_temperature_at_time(time_counts_data_dict.keys(), temperature_filepath)
    all_data = time_counts_data_dict.values()

    data_dict = {}
    for temperature, data in zip(temperatures, all_data):
        if temperature is not None:
            Cur_Array = data['Cur_array']
            Light_Array = data['Count_Array'] - data['Dark_Count_Array']
            plateau_cur, _ = get_plateau(Cur_Array, Light_Array)
            plateau_width = plateau_cur[-1] - plateau_cur[0]
            data_dict[temperature] = plateau_width
    
    output_dir = os.path.join(current_file_dir, 'data_temperatureDependence')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(time_counts_data_filepath)[0])
    filename = f'processed_{data_filename}.pkl'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    readable_output_dir = os.path.join(current_file_dir, 'readable_data_temperatureDependence')
    os.makedirs(readable_output_dir, exist_ok=True)
    json_filepath = f'{os.path.splitext(filepath)[0]}.json'
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))


