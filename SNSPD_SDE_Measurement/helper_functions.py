import time
import os
import pickle

import pandas as pd
import numpy as np

from datetime import datetime

# Algorithm S3.0.1. SNSPD IV Curve
def SNSPD_IV_Curve(instruments, now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), max_cur=15e-6, bias_resistor=100e3, name=''):
    ando = instruments['ando']
    laser_ch = instruments['laser_ch']
    sw_ch = instruments['sw_ch']
    monitor_port = instruments['monitor_port']
    att_list = instruments['att_list']
    srs = instruments['srs']

    ando.aq82011_disable(laser_ch)
    ando.aq8201418_set_route(sw_ch, monitor_port)
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, 0)
        ando.aq820133_disable(att_ch)

    srs.set_voltage(0)
    srs.set_output(output=True)

    # Generate current and voltage arrays
    num_biases = 100
    Cur_Array = np.linspace(0, max_cur, num_biases)
    Cur_Array = np.concatenate([Cur_Array, Cur_Array[::-1][1:], -Cur_Array[1:], -Cur_Array[::-1][1:]])
    Volt_Array = np.round(Cur_Array * bias_resistor, decimals=3)

    # Initialize array for measured voltages
    Volt_Meas = np.empty(Volt_Array.size, dtype=float)

    # Measure voltage for each set bias
    for i, volt in enumerate(Volt_Array):
        srs.set_voltage(volt)
        time.sleep(0.1)  # Wait for stabilization
        Volt_Meas[i] = multi.read_voltage()
        print(f"Applied Voltage: {volt}, Measured Voltage: {Volt_Meas[i]}")
        print(f"{round(100*i/Volt_Meas.size, 2)}%")
    srs.set_voltage(0)
    srs.set_output(output=False)

    # Combine data into a pandas DataFrame
    IV_curve_data = {
        "Current": Cur_Array,
        "Voltage": Volt_Meas
    }
    df = pd.DataFrame(IV_curve_data)

    # Save the DataFrame as a pickle file
    IV_pickle_file = f"{name}_IV_curve_data__{now_str}.pkl"
    os.makedirs("data", exist_ok=True)  # Ensure the "data" directory exists
    IV_pickle_filepath = os.path.join("data", IV_pickle_file)
    df.to_pickle(IV_pickle_filepath)

    print(f"IV curve data saved to {IV_pickle_filepath}")

    return IV_pickle_filepath

def get_ic(pickle_filepath, ic_threshold=1e-4):
    df = pd.read_pickle(pickle_filepath)
    filtered_df = df[df['Voltage'] > ic_threshold]  # Filter rows where Voltage > threshold
    if not filtered_df.empty:
        ic = filtered_df['Current'].iloc[0]  # Get the first current value
    else:
        ic = None
    return ic

def meas_counts(position, instruments, N=10, counting_time=1):
    pc = instruments['pc']
    counter = instruments['counter']
    srs = instruments['srs']

    pc.set_waveplate_positions(position)
    time.sleep(0.1)  # Wait for the motion to complete
    temp_counts = np.empty(N, dtype=float)
    for l in np.arange(temp_counts.size):
        counts = counter.timed_count(counting_time=counting_time)
        crap = 0
        while counts == 0:
            crap += 1
            if crap == 5:
                pol_counts_filepath = None
                return pol_counts_filepath
            srs.set_output(output=False)
            srs.set_output(output=True)
            counts = counter.timed_count(counting_time=counting_time)
        temp_counts[l] = counts
    avg_counts = np.mean(temp_counts)
    return avg_counts

def get_counts(Cur_Array, instruments, trigger_voltage=0.12, bias_resistor=100e3, counting_time=1, N=10):
    """
    Measures counts for a given array of currents.
    """

    srs = instruments['srs']
    counter = instruments['counter']

    srs.set_voltage(0)
    srs.set_output(output=True)

    counter.basic_setup()
    counter.set_impedance(ohms=50, channel=1)
    counter.setup_timed_count(channel=1)
    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

    Count_Array = np.zeros(len(Cur_Array))

    for i in range(len(Cur_Array)):
        this_volt = round(Cur_Array[i] * bias_resistor, 3)
        srs.set_voltage(this_volt)
        time.sleep(0.1)
        temp_counts = np.empty(N, dtype=float)
        for l in np.arange(temp_counts.size):
            temp_counts[l] = counter.timed_count(counting_time=counting_time)
        Count_Array[i] = np.mean(temp_counts)
        print(f"Voltage: {this_volt} V, Counts: {Count_Array[i]}")
    
    srs.set_voltage(0)
    srs.set_output(output=False)

    return Count_Array