import os
import re
import pickle
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

def interpolate_temperature(timestamp, temperature_dict):
    adjusted_dt = datetime.strptime(timestamp, '%Y%m%d-%H%M%S') + timedelta(minutes=1)
    
    sorted_times = sorted(temperature_dict.keys())
    prev_time, next_time = None, None

    for t in sorted_times:
        t_dt = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
        if t_dt <= adjusted_dt:
            prev_time = t
        elif t_dt > adjusted_dt and prev_time:
            next_time = t
            break

    if not prev_time or not next_time:
        return None

    prev_temp, next_temp = temperature_dict[prev_time], temperature_dict[next_time]
    if prev_temp is None or next_temp is None:
        return None

    prev_dt = datetime.strptime(prev_time, '%Y-%m-%d %H:%M:%S.%f')
    next_dt = datetime.strptime(next_time, '%Y-%m-%d %H:%M:%S.%f')
    total_seconds = (next_dt - prev_dt).total_seconds()
    elapsed_seconds = (adjusted_dt - prev_dt).total_seconds()

    return prev_temp + (next_temp - prev_temp) * (elapsed_seconds / total_seconds)


if __name__ == "__main__":
    temperature_filepath = os.path.join(current_file_dir, 'data_temperatureDependence', 'CTCLog 102424_15-09.txt')
    data_filepath = os.path.join(current_file_dir, 'data_temperatureDependence', ',pkl')
    
    temperature_dict = {}
    with open(temperature_filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            time, temperature = parts[0].strip(), parts[-3].strip()
            temperature_dict[time] = float(temperature) if temperature != 'nan' else None

    with open(data_filepath, "wb") as file:
        data_dict = pickle.load(file)

    for timestamp, data in data_dict.items():
        temperature = interpolate_temperature(timestamp, temperature_dict)
        if temperature is not None:
            plateau_cur, _ = get_plateau(data)
            plateau_width = plateau_cur[-1] - plateau_cur[0]
            data_dict[temperature] = plateau_width
    
    output_dir = os.path.join(current_file_dir, 'data_temperatureDependence')
    os.makedirs(output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(data_filepath)[0])
    filename = f'processed_{data_filename}.pkl'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    readable_output_dir = os.path.join(current_file_dir, 'readable_data_temperatureDependence')
    os.makedirs(readable_output_dir, exist_ok=True)
    json_filepath = f'{os.path.splitext(filepath)[0]}.json'
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))


