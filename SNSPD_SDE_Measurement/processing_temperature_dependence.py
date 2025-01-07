import os
import re
import pickle
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from processing_helpers import *

current_file_dir = Path(__file__).parent

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

def process_temperature(now_str="{:%Y:%m:%d-%H:%M:%S}".format(datetime.now()), temperature_filepath='CTCLog 102424_15-09.txt'):
    temperature_dict = {}
    with open(temperature_filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            time, temperature = parts[0].strip(), parts[-3].strip()
            temperature_dict[time] = float(temperature) if temperature != 'nan' else None

    data_dict = {}
    pattern = r'^lollipop_w130_counts_2micronLight__(\d{8}-\d{6})\.pbz2$'

    for file_name in os.listdir('data'):
        match = re.match(pattern, file_name)
        if match:
            timestamp = match.group(1)
            data = pickle.load(os.path.join('data', file_name))
            if data:
                data_dict[timestamp] = data

    temperatures, plateau_widths = [], []

    for timestamp, data in data_dict.items():
        interpolated_temp = interpolate_temperature(timestamp, temperature_dict)
        if interpolated_temp is not None:
            temperatures.append(interpolated_temp)
            plateau_cur, _ = get_plateau(data)
            plateau_width = plateau_cur[-1] - plateau_cur[0]
            plateau_widths.append(plateau_width)

    df = pd.DataFrame(
        {'temperatures': temperatures,
        'plateau_widths': plateau_widths,
        })
    
    temperature_dependence_filename = f'temperature_dependence_data__{now_str}.pkl'
    os.makedirs("data", exist_ok=True)
    temperature_dependence_filepath = os.path.join("data", temperature_dependence_filename)
    df.to_pickle(temperature_dependence_filepath)
    logger.info(f"Processed temperature dependence saved to: {temperature_dependence_filepath}")
    return

