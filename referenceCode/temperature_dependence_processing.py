"""
SNSPD Temperature Dependence Processing
"""

import os
import re
import bz2
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def decompress_pickle(file):
    try:
        with bz2.BZ2File(file, 'rb') as f:
            data = cPickle.load(f)
        return data
    except Exception as e:
        print(f"Error decompressing {file}: {e}")
        return None

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

def calculate_plateau_width(data):
    Cur_Array = np.array(data['Cur_Array'])
    Count_Array = np.array(data['Count_Array'])
    Dark_Count_Array = np.array(data['Dark_Count_Array'])
    Real_Count_Array = np.maximum(Count_Array - Dark_Count_Array, 0)

    max_counts = np.sort(Real_Count_Array)[-2]

    threshold = 0.94 * max_counts
    plateau_indices = np.where(Real_Count_Array >= threshold)[0]

    if len(plateau_indices) > 0:
        return Cur_Array[plateau_indices[-1]] - Cur_Array[plateau_indices[0]]
    return 0

temperature_dict = {}
with open('CTCLog 102424_15-09.txt', 'r') as file:
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
        data = decompress_pickle(os.path.join('data', file_name))
        if data:
            data_dict[timestamp] = data

temperatures, plateau_widths = [], []

for timestamp, data in data_dict.items():
    interpolated_temp = interpolate_temperature(timestamp, temperature_dict)
    if interpolated_temp is not None:
        temperatures.append(interpolated_temp)
        plateau_widths.append(calculate_plateau_width(data))

figpath = 'pics/plateau_width_temperature_dependence'
plt.figure(figsize=[20, 10])
plt.plot(temperatures, plateau_widths, '-*', color='k')
plt.title('Plateau Width Temperature Dependence')
plt.xlabel('Temperature [K]')
plt.ylabel('Plateau Width [uA]')
plt.ylim(bottom=0)  # Set y-axis lower limit to 0
plt.tight_layout()
plt.savefig(f'{figpath}.png')
plt.savefig(f'{figpath}.pdf')
plt.close()