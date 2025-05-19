import os
import pickle
import time
import logging
import json
from helpers import *

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import pyautogui
from PIL import Image

current_file_dir = Path(__file__).parent
logger = logging.getLogger(__name__)

# SNSPD IV Curve
def SNSPD_IV_Curve(instruments, now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), max_cur=15e-6, bias_resistor=100e3, name=''):
    logger.info("STARTING: SNSPD IV Curve")
    sw = instruments['sw']
    monitor_port = instruments['monitor_port']
    att_list = instruments['att_list']
    srs = instruments['srs']
    multi = instruments['multi']

    sw.set_route(monitor_port)
    for att in att_list:
        att.disable()

    srs.set_voltage(0)
    srs.set_output(output=True)

    # Generate current and voltage arrays
    num_biases = 100
    Cur_Array = np.linspace(0, max_cur, num_biases)
    Cur_Array = np.concatenate([Cur_Array, Cur_Array[::-1][1:], -Cur_Array[1:], -Cur_Array[::-1][1:]])
    Volt_Array = np.round(Cur_Array * bias_resistor, decimals=3)
    Volt_Meas_Array = np.empty(Cur_Array.shape, dtype=float)

    total_num_biases = Cur_Array.size

    # Measure voltage for each set bias
    for i in tqdm(total_num_biases):
        set_volt = Volt_Array[i]
        srs.set_voltage(set_volt)
        time.sleep(0.1)  # Wait for stabilization
        Volt_Meas = multi.read_voltage()
        logger.info(f"Applied Current: {Cur_Array[i]}, Voltage: {set_volt}; Measured Voltage: {Volt_Meas}")
        Volt_Meas_Array[i] = Volt_Meas
        logger.info(f"{round(100*i/total_num_biases, 2)}%")
    srs.set_voltage(0)
    srs.set_output(output=False)    

    data_dict = {
        'Cur_Array': Cur_Array,
        'Volt_Meas_Array': Volt_Meas_Array, 
    }

    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}_IV_curve_data__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(data_dict, file)

    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    json_filepath = f'{os.path.splitext(filepath)[0]}.json'
    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    logger.info(f"IV curve data saved to: {filepath}")
    logger.info("COMPLETED: SNSPD IV Curve")
    return filepath

def get_counts(Cur_Array, instruments, trigger_voltage=0.12, bias_resistor=100e3, counting_time=1, N=3):
    """
    Measures counts for a given array of currents.
    """
    # Change this function so that it saves multiple mesaurements at each bias current

    srs = instruments['srs']
    counter = instruments['counter']

    srs.set_voltage(0)
    srs.set_output(output=True)

    counter.basic_setup()
    counter.set_impedance(ohms=50, channel=1)
    counter.setup_timed_count(channel=1)
    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

    Volt_Array = np.round(Cur_Array * bias_resistor, decimals=3)
    Count_Array = np.empty((len(Cur_Array), N), dtype=float)

    for i in tqdm(len(Cur_Array)):
        srs.set_voltage(Volt_Array[i])
        time.sleep(0.1)
        temp_cps = np.empty(N, dtype=float)
        for j in np.arange(temp_cps.size):
            temp_cps[j] = counter.timed_count(counting_time=counting_time)/counting_time
        Count_Array[i] = temp_cps
        logger.info(f"Voltage: {Volt_Array[i]} V, Counts: {np.mean(temp_cps)}")
    
    srs.set_voltage(0)
    srs.set_output(output=False)

    return Count_Array

def find_min_trigger_threshold(
    instruments, 
    max_trigger_voltage=0.2, 
    N=7, 
    channel=1, 
    ohms=50, 
    counting_time=0.5
):
    """
    Finds the minimum trigger threshold voltage for the counter using binary search.
    
    Parameters:
        instruments (dict): Dictionary containing instrument instances ('srs' and 'counter').
        now_str (str): Timestamp for saving data files.
        max_trigger_voltage (float): Maximum voltage to test for triggering.
        N (int): Number of measurements to average for each voltage.
        channel (int): Counter channel to use.
        ohms (int): Impedance value in ohms.
        counting_time (float): Measurement time for each count.
    
    Returns:
        float: Final calculated trigger voltage.
    """
    srs = instruments['srs']
    counter = instruments['counter']

    # Initialize SRS and counter
    srs.set_voltage(0)
    srs.set_output(output=True)

    counter.basic_setup()
    counter.set_impedance(ohms=ohms, channel=channel)
    counter.setup_timed_count(channel=channel)    

    def measure_avg_cps(trigger_voltage):
        """Measure average counts per second (CPS) at a given trigger voltage."""
        counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=channel)
        time.sleep(0.1)  # Allow system to stabilize
        cps_values = [counter.timed_count(counting_time=counting_time) / counting_time for _ in range(N)]
        avg_cps = np.mean(cps_values)
        return avg_cps

    # Binary search for trigger voltage
    low_voltage = 0
    high_voltage = max_trigger_voltage
    tolerance = 0.005

    data_dict = {}
    while abs(high_voltage - low_voltage) > tolerance:
        mid_voltage = (low_voltage + high_voltage) / 2
        avg_cps = measure_avg_cps(mid_voltage)
        
        data_dict[mid_voltage] = avg_cps
        logger.info(f"Voltage: {mid_voltage:.3f}, Avg CPS: {avg_cps:.3f}")

        if avg_cps == 0:
            high_voltage = mid_voltage
        elif avg_cps < 1:
            low_voltage = (mid_voltage + low_voltage) / 2
        else:
            low_voltage = mid_voltage

    # Apply a safety margin to the final trigger voltage
    final_trigger_voltage = high_voltage * 1.1

    # # Save measurement data
    # output_dir = os.path.join(current_file_dir, 'data_sde')
    # os.makedirs(output_dir, exist_ok=True)
    # filename = f'trigger_voltage_data__{now_str}.pkl'
    # filepath = os.path.join(output_dir, filename)
    # with open(filepath, "wb") as file:
    #     pickle.dump(data_dict, file)

    # readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    # os.makedirs(readable_output_dir, exist_ok=True)
    # json_filepath = f'{os.path.splitext(filepath)[0]}.json'
    # with open(json_filepath, 'w') as f:
    #     json.dump(data_dict, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    # logger.info(f"Trigger voltage data saved to: {filepath}")

    logger.info(f"Final trigger voltage: {final_trigger_voltage:.3f}")
    return final_trigger_voltage


def zero_ando_pm(instruments):
    sw = instruments['sw']
    detector_port = instruments['detector_port']
    att_list = instruments['att_list']
    mpm = instruments['mpm']

    sw.set_route(instruments[detector_port])
    for att in att_list:
        att.disable()
    time.sleep(0.3)
    mpm.zero()

def reset_attenuators(instruments):
    att_list = instruments['att_list']
    for att in att_list:
        att.set_att(0)
        att.enable()

def find_mpm_rng(instruments, rng, N_init=3):
    mpm = instruments['mpm']
    while True:
        logger.info(f"rng: {rng}")
        mpm.set_range(rng)
        time.sleep(0.3)
        mpm.get_power()
        powers = [mpm.get_power() for _ in range(N_init)]  # Collect power readings
        logger.info(f"powers: {powers}")
        check_range = [mpm.check_ideal_rng(power=power, rng=rng) for power in powers]
        logger.info(f"check_range: {check_range}")
        sum_check_range = sum(check_range)
        logger.info(f"sum_check_range: {sum_check_range}")
        if all(el == 0 for el in check_range):
            return rng
        elif sum_check_range > 0:
            rng += 10
        elif sum_check_range < 0:
            rng -= 10
        if rng < -60:
            logger.error("Range setting is being set below -60")
            break
        if rng > 30:
            logger.error("Range setting is being set above 30")
            break

def meas_counts(instruments, position, N=3, counting_time=1):
    pc = instruments['pc']
    counter = instruments['counter']
    srs = instruments['srs']
    att_list = instruments['att_list']
    pc.set_waveplate_positions(position)
    time.sleep(0.3)  # Wait for the motion to complete
    cps = []
    for _ in range(N):
        cps.append(counter.timed_count(counting_time=counting_time)/counting_time)
    
    while np.mean(cps) == 0: # if measured 0 counts, it might have latched, reset sres
        logger.warning("Getting 0 counts. Resetting SRS and attenuators and hoping it will correct")
        srs.set_output(output=False)
        for att in att_list:
            att.disable()
        srs.set_output(output=True)
        for att in att_list:
            att.enable()
        cps = []
        for _ in range(N):
            cps.append(counter.timed_count(counting_time=counting_time)/counting_time)

    return cps

# Algorithm S1.1 Missing Algorithm (optical switch calibration)
def optical_switch_calibration(instruments,
                               name="{:%Y%m%d-%H%M%S}".format(datetime.now()),
                               mpm_types=None,
                               wavelength=1550):
    """
    Perform optical switch calibration and save the calibration data.

    Parameters:
        instruments (dict): Dictionary of instrument interfaces.
        name (str): Timestamped name for saving files.
        mpm_types (list): List of power meter types corresponding to mpms list.
        wavelength (float): Wavelength to set for power meters (nm).

    Returns:
        str: Filepath of the saved calibration data.
    """
    if mpm_types is None:
        mpm_types = []

    logger.info("Starting: Algorithm S1.1 Optical Switch Calibration")
    logger.warning("Ensure detector fiber is spliced to calibrated power meter")

    sw = instruments['sw']
    monitor_port = instruments['monitor_port']
    detector_port = instruments['detector_port']
    laser = instruments['laser']
    mpms = instruments['mpms']
    mpm_sw = instruments['mpm_sw']
    att_list = instruments['att_list']
    cpm = instruments['cpm']

    # Initialize power meters
    sw.set_route(monitor_port)
    reset_attenuators(instruments)
    for mpm, mpm_type in zip(mpms, mpm_types):
        if mpm_type == 'ando':
            mpm.set_lambda(wavelength)
            zero_ando_pm()
            init_rng = find_mpm_rng(0)
            mpm.set_range(init_rng)
    cpm.set_pm_wavelength(wavelength)

    # Calibrate CPM to 100 uW
    N = 10
    sw.set_route(detector_port)
    low_att = 0.0
    high_att = 10.0
    tol = 1e-6
    avg_cpm = 0.0
    while abs(avg_cpm - 100e-6) > tol:
        mid = round((low_att + high_att) / 2, 3)
        laser.set_att(mid)
        time.sleep(0.1)
        readings = [float(cpm.read_power()) for _ in range(N)]
        avg_cpm = sum(readings) / N
        logger.info(f"Tuning attenuation={mid}, Avg CPM={avg_cpm}")
        if avg_cpm < 100e-6:
            high_att = mid
        else:
            low_att = mid

    # Prepare storage arrays
    num_mpm = len(mpms)
    total_pts = N * N
    power_mpm = np.empty((total_pts, num_mpm), dtype=float)
    power_cpm = np.empty(total_pts, dtype=float)

    # Collect power data
    for i in range(N):
        sw.set_route(monitor_port)
        time.sleep(0.3)
        for k, (mpm, mpm_type) in enumerate(zip(mpms, mpm_types)):
            if mpm_type == 'ando':
                mpm_sw.set_route(instruments['ando_port'])
                for j in range(N):
                    power_mpm[i*N + j, k] = mpm.get_power()
            elif mpm_type == 'InGaAs':
                mpm_sw.set_route(instruments['ingaas_port'])
                for j in range(N):
                    power_mpm[i*N + j, k] = mpm.get_meter_pow(4)
            elif mpm_type == 'thermal':
                mpm_sw.set_route(instruments['thermal_port'])
                for j in range(N):
                    power_mpm[i*N + j, k] = capture_screen_and_extract_text(100, 200, 100, 200)

        sw.set_route(detector_port)
        time.sleep(0.3)
        for j in range(N):
            power_cpm[i*N + j] = cpm.read_power()

        logger.info(f"Iter {i+1}/{N}: MPM readings={power_mpm[i*N:(i+1)*N, :]}, CPM readings={power_cpm[i*N:(i+1)*N]}")
        logger.info(f"Progress: {100*(i+1)/N:.1f}%")

    # Reset switch and attenuators
    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Build DataFrame
    col_names = [f"power_mpm_{k}" for k in range(num_mpm)]
    df = pd.DataFrame(power_mpm, columns=col_names)
    df['power_cpm'] = power_cpm

    # Save DataFrame
    current_dir = os.path.dirname(__file__)
    out_dir = os.path.join(current_dir, 'data_sde')
    os.makedirs(out_dir, exist_ok=True)
    fname = f"optical_switch_calibration_{name}.pkl"
    out_path = os.path.join(out_dir, fname)
    df.to_pickle(out_path)

    # Also save JSON
    readable_dir = os.path.join(current_dir, 'readable_data_sde')
    os.makedirs(readable_dir, exist_ok=True)
    json_name = f"optical_switch_calibration_{name}.json"
    json_path = os.path.join(readable_dir, json_name)
    with open(json_path, 'w') as jf:
        json.dump(df.to_dict(orient='list'), jf, indent=4)

    logger.info(f"Calibration data saved to: {out_path}")
    logger.info("Completed: Algorithm S1.1 Optical Switch Calibration")

    return out_path

# Algorithm S2. Attenuator Calibration
def attenuator_calibration(instruments,now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), attval=30, mpm_type=''):
    logger.info("Starting: Algorithm S2. Attenuator Calibration")
    
    sw=instruments['sw']
    monitor_port=instruments['monitor_port']
    att1=instruments['att1']
    mpm=instruments['mpm']
    att_list=instruments['att_list']
    
    
    sw.set_route(monitor_port)

    # Parameters
    N = 100
    sw.set_route(monitor_port)
    att_rng=0
    init_rng = 0 
    if mpm_type != 'InGaAs':
        reset_attenuators(instruments)
        att1.set_att(attval)
        att_rng = find_mpm_rng(round(-attval, -1))
        logger.info(f" att_rng: {att_rng}")
        reset_attenuators(instruments)
        init_rng = find_mpm_rng(round(0, -1))
        logger.info(f" init_rng: {init_rng}")

    # Initialize an empty DataFrame to store results
    columns = ['Attenuator', 'Attenuation (dB)', 'Range', 'Power Measurement']
    df = pd.DataFrame(columns=columns)

    # Calibrate each attenuator in att_list
    init_powers = []
    rows = []  # Collect rows to add to the DataFrame
    for i, atti in enumerate(att_list):
        # Step 2: Monitor setup for initial power measurements
        if mpm_type != 'InGaAs':
            mpm.set_range(init_rng)
            zero_pm()
        sw.set_route(monitor_port)
        reset_attenuators(instruments)
        time.sleep(0.3)
       

        if mpm_type != 'InGaAs':
            mpm.get_power()
            for _ in range(N):
                init_powers.append(mpm.get_power())
            mpm.get_power()
        else:
            mpm.get_meter_pow(4)
            for _ in range(N):
                init_powers.append(mpm.get_meter_pow(4))
            mpm.get_meter_pow(4)

        # Step 3: Apply attenuation and measure power
        if mpm_type != 'InGaAs':
            mpm.set_range(att_rng)
            zero_pm()
        sw.set_route(monitor_port)
        reset_attenuators(instruments)
        atti.set_att(attval)
        temp_powers = []
        time.sleep(0.3)
        if mpm_type != 'InGaAs':
            mpm.get_power()
            for _ in range(N):
                temp_powers.append(mpm.get_power())
            mpm.get_power()
        else:
            mpm.get_meter_pow(4)
            for _ in range(N):
                temp_powers.append(mpm.get_meter_pow(4))
            mpm.get_meter_pow(4)
        
        # Collect the row to add later
        data_temp = {
            'Attenuator': i,
            'Attenuation (dB)': attval,
            'Range': att_rng,
            'Power Measurement': temp_powers
        }
        rows.append(data_temp)

        logger.info(data_temp)
        logger.info(f"{round(100 * i / len(att_list), 2)}% completed")

    # Turn off attenuators
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Add initial power measurements row
    data_0 = {
        'Attenuator': None,
        'Attenuation (dB)': 0,
        'Range': init_rng,
        'Power Measurement': init_powers
    }
    rows.append(data_0)
    logger.info(data_0)

    # Add all rows to the DataFrame at once
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # Save the calibration data to a file
    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"attenuator_calibration_data_attval{attval}__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    df.to_pickle(filepath)
    
    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(filepath)[0])
    csv_filename = f'{data_filename}.csv'
    csv_filepath = os.path.join(readable_output_dir, csv_filename)
    df.to_csv(csv_filepath, index=False)

    logger.info(f"attenuator_calibration saved to: {filepath}")
    logger.info("Completed: Algorithm S2. Attenuator Calibration")
    return filepath

# Algorithm S3.1. SDE Counts Measurement - Polarization Sweep
def sweep_polarizations(instruments,now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', attval=30, name='', trigger_voltage=0.01, num_pols=13, counting_time=0.5, N=3,bias_resistor=97e3,snspd_splice="1connectors"):
    """Sweep through all polarizations to find max and min counts.
    Args:
        detector (object): Detector object with a `get_counts()` method.
        step (float): Step size in degrees for the sweep.
    Returns:
        dict: A dictionary with max and min polarization settings and counts.
    """
    logger.info("Starting: Algorithm S3.1. SDE Counts Measurement - Polarization Sweep")
    logger.warning("Ensure detector fiber it spliced to SNSPD")

    counter=instruments['counter']
    sw=instruments['sw']
    detector_port=instruments['detector_port']
    srs=instruments['srs']
    att_list=instruments['att_list']
    monitor_port=instruments['monitor_port']
    att_list=instruments['att_list']
    att_list=instruments['att_list']


    ic = get_ic(IV_pickle_filepath)

    sw.set_route(detector_port)
    for att in att_list:
        att.set_att(attval)
        att.enable()

    counter.basic_setup()
    counter.set_impedance(ohms=50, channel=1)
    counter.setup_timed_count(channel=1)
    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

    srs.set_voltage(0)
    srs.set_output(output=True)
    this_volt = round(ic*0.92 * bias_resistor, 3)
    srs.set_voltage(this_volt)

    num_repeats = 3
    positions = np.linspace(-60.0, 60.0, num_pols) # Max range is -99 to 100 but I want to limit these edge cases
    positions = np.round(positions, 2)
    pol_data = {}
    for n in tqdm(num_repeats):
        for i, x in enumerate(positions):
            for j, y in enumerate(positions):
                for k, z in enumerate(positions):
                    position = (x, y, z)
                    cps = meas_counts(instruments,position, N=N, counting_time=counting_time)
                    logger.info(f"Position: {position}, counts: {np.mean(cps)}")
                    if position not in pol_data:
                        pol_data[position] = []
                    pol_data[position].append(cps)
                    logger.info(f"{round(100*(n*positions.size**3 + i*positions.size**2 + j*positions.size + k)/(num_repeats*positions.size**3), 2)}% Complete")

    data_dict = {key: np.mean(value) for key, value in pol_data.items()}
    srs.set_voltage(0)
    srs.set_output(output=False)

    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    maxpol_settings = max(data_dict, key=data_dict.get)
    minpol_settings = min(data_dict, key=data_dict.get)
    logger.info(f"max pol settings: {maxpol_settings}, cps: {data_dict[maxpol_settings]}")
    logger.info(f"min pol settings: {minpol_settings}, cps: {data_dict[minpol_settings]}")
    
    output_dir = os.path.join(current_file_dir, 'data_sde')
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}_pol_data_snspd_splice{snspd_splice}__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(data_dict, file)

    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(filepath)[0])
    json_filename = f'{data_filename}.json'
    json_filepath = os.path.join(readable_output_dir, json_filename)
    data_dict_json = {str(k): v for k, v in data_dict.items()}
    with open(json_filepath, 'w') as f:
        json.dump(data_dict_json, f, indent=4)
    
    logger.info(f"Polarization data saved to: {filepath}")
    logger.info("Completed: Algorithm S3.1. SDE Counts Measurement - Polarization Sweep")
    return filepath

# Algorithm S3.2. SDE Counts Measurement - True Counting
def SDE_Counts_Measurement(instruments,now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', pol_counts_filepath='', attval=30, name='', trigger_voltage=0.01, bias_resistor=97e3,counting_time=1,snspd_splice="1connectors",):    
    logger.info("Starting: Algorithm S3.2. SDE Counts Measurement - True Counting")
    logger.warning("Ensure detector fiber it spliced to SNSPD")

    monitor_port=instruments['monitor_port']
    sw=instruments['sw']
    att_list=instruments['att_list']
    detector_port=instruments['detector_port']
    pc=instruments['pc']


    with open(pol_counts_filepath, 'rb') as file:
        pol_data = pickle.load(file)

    # Find the tuple with the highest count
    maxpol_settings = max(pol_data, key=pol_data.get)
    minpol_settings = min(pol_data, key=pol_data.get)

    # Perform measurements
    ic = get_ic(IV_pickle_filepath)
    num_biases = 100
    Cur_Array = np.linspace(ic * 0.2, ic * 1.1, num_biases)

    # Dark counts measurement
    sw.set_route(monitor_port)
    for att in att_list:
        att.disable()
    logger.info("Collecting dark counts")
    Dark_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    logger.info("Dark counts collected")

    # Max and min polarization measurements
    sw.set_route(detector_port)
    for att in att_list:
        att.enable()
        att.set_att(attval)
    
    # Measure counts at max polarization
    pc.set_waveplate_positions(maxpol_settings)
    logger.info("Collecting max polarization counts")
    Maxpol_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    logger.info("Max polarization counts collected")

    # Measure counts at min polarization
    pc.set_waveplate_positions(minpol_settings)
    logger.info("Collecting min polarization counts")
    Minpol_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    logger.info("Min polarization counts collected")

    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Save data
    data_dict = {
        'Cur_Array': list(Cur_Array),
        'Dark_Count_Array': list(Dark_Count_Array),
        'Maxpol_Count_Array': list(Maxpol_Count_Array),
        'Minpol_Count_Array': list(Minpol_Count_Array),
        'Maxpol_Settings': maxpol_settings,
        'Minpol_Settings': minpol_settings,
    }    
    
    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}_counts_data_snspd_splice{snspd_splice}_attval{attval}__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(data_dict, file)

    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(filepath)[0])
    json_filename = f'{data_filename}.json'
    json_filepath = os.path.join(readable_output_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    logger.info(f"data_dict saved to: {filepath}")
    logger.info("Completed: Algorithm S3.2. SDE Counts Measurement - True Counting")
    return filepath


# Daniel Sorensen Code
# def optimizePolarization(instruments, biasV = 0.7, biasChannel=1, ttChannel=5, tMeasure=0.5, minimize=False, attValue=28):
#     sc = serverConnection()
#     cr = ttCountRate(channel=ttChannel)
#     pc = instruments.pc

#     def optFunc(v, grad=None):
#         pc.setAll(v)
#         counts=cr.measureFor(tMeasure)/tMeasure
#         logger.info(f"P: {v}, Counts: {counts}")
#         return counts
    
#     instruments.att1.set_att(attValue)
#     instruments.att2.set_att(attValue)
#     instruments.att3.set_att(attValue)
#     instruments.laser.enable()
#     instruments.switch.set_route(2)
#     sc.setBias(biasChannel,biasV)

#     opt = nlopt.opt(nlopt.LN_SBPLX,3)
#     if minimize:
#         opt.set_min_objective(optFunc)
#     else:
#         opt.set_max_objective(optFunc)
#     opt.set_lower_bounds(np.ones(3)*-99.)
#     opt.set_upper_bounds(np.ones(3)*99.)
#     opt.set_xtol_abs(np.ones(3))
#     opt.set_initial_step(np.ones(3)*30.)
#     startPol = np.zeros(3)
#     opt.optimize(startPol)

#     optX = pc.getAxis('X')
#     optY = pc.getAxis('Y')
#     optZ = pc.getAxis('Z')
#     optPol = (optX,optY,optZ)
#     if minimize:
#         logger.info("Found minimum polarization:")
#     else:
#         logger.info("Found maximum polarization:")
#     logger.info(optPol)
#     return optPol


# My attempt
# bounds = [(-99, 100)] * 3
# def neg_meas_counts(position, *args):
#     return -meas_counts(position, *args)
# initial_guess = np.array([0, 0, 0])
# res_min = scipy.optimize.minimize(meas_counts, initial_guess, args=(instruments, N, counting_time), bounds=bounds)
# res_max = scipy.optimize.minimize(neg_meas_counts, initial_guess, args=(instruments, N, counting_time), bounds=bounds)
# pol_counts = [(res_min['x'], res_min['fun']), (res_max['x'], res_max['fun'])]
# logger.info(pol_counts)




# Algorithm S1. Nonlinearity factor raw power measurements
def nonlinearity_factor_raw_power_measurements(instruments, now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), taus=[3]):
    logger.info("Starting: Algorithm S1. Nonlinearity factor raw power measurements")

    mpm = instruments['mpm']
    sw = instruments['sw']
    monitor_port = instruments['monitor_port']
    att1 = instruments['att1']
    att2 = instruments['att2']
    att_list = instruments['att_list']
    # sw.set_route(monitor_port)
    # reset_attenuators(instruments)


    N_init = 10

    

    rng_settings = [0, -10, -20, -30, -40, -50, -60]
    att2_settings = [0] + taus
    tau_min = min(taus)
    N = 25

    # Algorithm to find appropriate attenuation settings for each monitor power meter range setting
    tolerance = 0.1
    att_min_max_settings = {}
    for rng in rng_settings:
        mpm.set_range(rng)
        zero_pm()
        sw.set_route(monitor_port)
        reset_attenuators(instruments)
        middle_a = -rng - 2.5

        high_max_a = min((middle_a + 20), 60)
        low_max_a = max((middle_a - 20), 0)
        while abs(high_max_a - low_max_a) > tolerance:
            middle_max_a = round(((high_max_a + low_max_a) / 2), 3)
            logger.info(f"middle_min_a: {middle_max_a}")
            att1.set_att(middle_max_a)
            time.sleep(0.3)
            mpm.get_power()
            powers = [mpm.get_power() for _ in range(N_init)]  # Collect power readings
            logger.info(f"powers: {powers}")
            check_range = [mpm.check_correct_rng(power=power, rng=rng) for power in powers]
            logger.info(f"check_range: {check_range}")
            sum_check_range = sum(check_range)
            logger.info(f"sum_check_range: {sum_check_range}")
            if sum_check_range < 0:
                high_max_a = middle_max_a
            else:
                low_max_a = middle_max_a
        a_max = round(middle_max_a + 0.1, 1)

        att2.set_att(tau_min)
        high_min_a = min((middle_a + 20), 60)
        low_min_a = max((middle_a - 20), 0)
        while abs(high_min_a - low_min_a) > tolerance:
            middle_min_a = round(((high_min_a + low_min_a) / 2), 3)
            att1.set_att(middle_min_a)
            time.sleep(0.3)
            mpm.get_power()
            powers = [mpm.get_power() for _ in range(N_init)]  # Collect power readings
            check_range = [mpm.check_correct_rng(power=power, rng=rng) for power in powers]
            sum_check_range = sum(check_range)
            if sum_check_range > 0:
                low_min_a = middle_min_a
            else:
                high_min_a = middle_min_a
        a_min = round(middle_min_a - 0.1, 1)

        att_min_max_settings[rng] = {
            'min': a_min, 
            'max': a_max, 
            }
        logger.info(f"att_min_max_settings: {att_min_max_settings}")

    logger.info(f"rng_settings: {rng_settings}")
    logger.info(f"att_min_max_settings: {att_min_max_settings}")

    att_settings = {}
    for i in range(len(rng_settings)):
        current_range = rng_settings[i]
        prev_range = rng_settings[i - 1] if i > 0 else None
        next_range = rng_settings[i + 1] if i < len(rng_settings) - 1 else None

        if prev_range is None:  # First element (use next_range)
            att_settings[current_range] = np.concatenate([
                np.arange(att_min_max_settings[current_range]['min'], att_min_max_settings[next_range]['min'], 1),
                np.arange(att_min_max_settings[next_range]['min'], att_min_max_settings[current_range]['max'], 0.1),
            ])
        elif next_range is None:  # Last element (use prev_range)
            att_settings[current_range] = np.concatenate([
                np.arange(att_min_max_settings[current_range]['min'], att_min_max_settings[prev_range]['max'], 0.1),
                np.arange(att_min_max_settings[prev_range]['max'], att_min_max_settings[current_range]['max'], 1),
            ])
        else:  # Middle elements
            att_settings[current_range] = np.concatenate([
                np.arange(att_min_max_settings[current_range]['min'], att_min_max_settings[prev_range]['max'], 0.1),
                np.arange(att_min_max_settings[prev_range]['max'], att_min_max_settings[next_range]['min'], 1),
                np.arange(att_min_max_settings[next_range]['min'], att_min_max_settings[current_range]['max'], 0.1),
            ])

    total_data = 0
    for value in att_settings.values():
        total_data += len(value)

    att_settings = {
        rng: [round(val, 1) for val in values]
        for rng, values in att_settings.items()
    }
    logger.info(f"att_settings: {att_settings}")

    total_data *= len(att2_settings)

    

    # Now we actually start taking data
    data = []
    i = 0
    for rng in rng_settings:
        mpm.set_range(rng)
        zero_pm()
        sw.set_route(monitor_port)
        reset_attenuators(instruments)
        for a in att_settings[rng]:
            att1.set_att(a)
            for att2_val in att2_settings:
                att2.set_att(att2_val)
                powers = []
                time.sleep(0.3)
                mpm.get_power()
                powers = [mpm.get_power() for _ in range(N)]
                mpm.get_power()
                data_temp = (rng, a, att2_val, powers)
                data.append(data_temp)
                logger.info(f" data_temp: {data_temp}")
                i += 1
                logger.info(f" {round(100*i/total_data, 2)}% complete")
                
    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Convert the data to a pandas DataFrame
    columns = ['Range', 'Attenuator 1', 'Attenuator 2', 'Power']
    df = pd.DataFrame(data, columns=columns)

    output_dir = os.path.join(current_file_dir, "data_sde")
    os.makedirs(output_dir, exist_ok=True)
    filename = f'nonlinear_calibration_data__{now_str}.pkl'
    filepath = os.path.join(output_dir, filename)
    df.to_pickle(filepath)
    
    # This is untested
    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(filepath)[0])
    csv_filename = f'{data_filename}.csv'
    csv_filepath = os.path.join(readable_output_dir, csv_filename)
    df.to_csv(csv_filepath, index=False)

    logger.info(f"nonlinearity_factor saved to: {filepath}")
    logger.info("Completed: Algorithm S1. Nonlinearity factor raw power measurements")
    return filepath



# If Tesseract is not in your PATH, specify the path manually
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_screen_and_extract_text(x, y, width, height):
    logger.info("Take a screenshot of the area you want to capture...")
    
    # # Let the user select a region of the screen
    # screenshot = pyautogui.screenshot()
    # screenshot.save("full_screenshot.png")

    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    
    # Extract text from the screenshot
    text = pytesseract.image_to_string(screenshot)
    
    logger.info("\nExtracted Text:\n")
    logger.info(text)