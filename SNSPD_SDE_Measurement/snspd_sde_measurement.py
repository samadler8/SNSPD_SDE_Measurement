# %% Imports, parameters, initialization
import os
import time
import pickle
import logging
import json

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.ando_aq82011 import AndoAQ82011
from amcc.instruments.ando_aq820133 import AndoAQ820133
from amcc.instruments.ando_aq8201418 import AndoAQ8201418
from amcc.instruments.ando_aq82012 import AndoAQ82012
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.agilent_8163a import Agilent8163A
from amcc.instruments.fiberControl_MPC101 import FiberControlMPC101
from amcc.instruments.agilent_34411a import Agilent34411A

from measure_helpers import *
from helpers import *

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("snspd_sde_measurement.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

srs = SIM928('GPIB0::2::INSTR', 5)
laser = AndoAQ82011('GPIB0::4::INSTR', 1)
att1 = AndoAQ820133('GPIB0::4::INSTR', 2)
att2 = AndoAQ820133('GPIB0::4::INSTR', 3)
att3 = AndoAQ820133('GPIB0::4::INSTR', 4)
sw = AndoAQ8201418('GPIB0::4::INSTR', 5)
mpm = AndoAQ82012('GPIB0::4::INSTR', 6)
counter = Agilent53131a('GPIB0::5::INSTR')
cpm = Agilent8163A('GPIB0::9::INSTR', 1)
pc = FiberControlMPC101('GPIB0::3::INSTR')
multi = Agilent34411A('GPIB0::21::INSTR')

att_list = [att1, att2, att3]

monitor_port = 1
detector_port = 2

instruments = {'srs': srs,
    'counter': counter,
    'cpm': cpm,
    'counter': counter,
    'pc': pc,
    'multi': multi,
    'laser': laser,
    'att1': att1,
    'att2': att2,
    'att3': att3,
    'att_list': att_list,
    'sw': sw,
    'mpm': mpm,
    'monitor_port': monitor_port,
    'detector_port': detector_port,
    }

rng_dict = {'A': 'AUTO',
    'C': +30,
    'D': +20,
    'E': +10,
    'F': 0,
    'G': -10,
    'H': -20,
    'I' : -30,
    'J' : -40,
    'K': -50,
    'L' : -60,
    'Z' : 'HOLD'
}

# setup constants
wavelength = 1566.314  # nm
bias_resistor = 97e3 #Ohms
init_rng = 0 #dBm

counting_time = 0.5 #s
num_pols = 13
N_init = 3

name = 'Saeed2um'
max_cur = 15e-6 # A

cpm_splice = 2
snspd_splice = 1

def zero_pm():
    sw.set_route(detector_port)
    for att in att_list:
        att.disable()
    time.sleep(0.3)
    mpm.zero()

def reset_attenuators():
    for att in att_list:
        att.set_att(0)
        att.enable()

# Initialize and turn on everything
def snspd_sde_setup():
    laser.std_init()
    laser.set_lambda(wavelength)
    laser.enable()

    for att in att_list:
        att.set_att(0)
        att.disable()
        
    mpm.set_lambda(wavelength)
    mpm.set_range('A')
    zero_pm()

    cpm.set_pm_wavelength(wavelength)

    counter.basic_setup()
    counter.set_impedance(ohms=50, channel=1)
    counter.setup_timed_count(channel=1)

    srs.set_voltage(0)
    srs.set_output(output=False)

def find_mpm_rng(rng):
    while True:
        logger.info(f"rng: {rng}")
        mpm.set_range(rng)
        time.sleep(0.3)
        mpm.get_power()
        powers = [mpm.get_power() for _ in range(N_init)]  # Collect power readings
        logger.info(f"powers: {powers}")
        check_range = [mpm.check_correct_rng(power=power, rng=rng) for power in powers]
        logger.info(f"check_range: {check_range}")
        sum_check_range = sum(check_range)
        logger.info(f"sum_check_range: {sum_check_range}")
        if sum_check_range == 0:
            return rng
        elif sum_check_range > 0:
            rng += 10
        elif sum_check_range < 0:
            rng -= 10

def meas_counts(position, N=3, counting_time=1):
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
def optical_switch_calibration(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), ):
    """
    Perform optical switch calibration and save the calibration data.

    Parameters:
        now_str (str): Timestamp for the filename. Defaults to current time.
        num_iterations (int): Number of iterations for data collection. Default is 50.
        output_dir (str): Directory to save calibration data. Default is 'data_sde'.
        init_rng: Initial range for power meter.
        att_list (list): List of attenuator objects to initialize.
        cpm_splice: Splice identifier for CPM.
        monitor_port: Monitor port for the optical switch.
        detector_port: Detector port for the optical switch.

    Returns:
        str: Filepath of the saved calibration data.
    """
    logger.info("Starting: Algorithm S1.1 Missing Algorithm (optical switch calibration)")
    logger.warning("Ensure detector fiber it spliced to calibrated power meter")
    # Initialize power meter
    sw.set_route(monitor_port)
    reset_attenuators()
    init_rng = find_mpm_rng(0)

    mpm.set_range(init_rng)
    mpm.zero()

    # Set optical switch to monitoring port and initialize attenuators
    for att in att_list:
        att.set_att(0)
        att.enable()

    N = 50

    power_mpm = np.empty(N**2, dtype=float)
    power_cpm = np.empty(N**2, dtype=float)

    # Collect power data
    for i in range(N):
        sw.set_route(monitor_port)
        time.sleep(0.3)
        mpm.get_power()
        for j in range(N):
            power_mpm[i*N + j] = mpm.get_power()
        mpm.get_power()

        sw.set_route(detector_port)
        time.sleep(0.3)
        cpm.read_power()
        for j in range(N):
            power_cpm[i*N + j] = cpm.read_power()
        cpm.read_power()

        logger.info(f"Iteration {i+1}/{N}: Monitor Power={power_mpm[i*N:i*N+N]}, Detector Power={power_cpm[i*N:i*N+N]}")
        logger.info(f"Progress: {round(100 * (i + 1) / N, 2)}%")

    # Reset switch and attenuators
    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Save data to a DataFrame
    data_dict = {
        'power_mpm': power_mpm, 
        'power_cpm': power_cpm,
        }

    # Save the DataFrame as a pickle file
    output_dir = os.path.join(current_file_dir, "data_sde")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"optical_switch_calibration_data_cpm_splice{cpm_splice}__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)
    
    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(filepath)[0])
    json_filename = f'{data_filename}.json'
    json_filepath = os.path.join(readable_output_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    logger.info(f"Optical switch calibration data saved to: {filepath}")
    logger.info("Completed: Algorithm S1.1 Optical Switch Calibration")

    return filepath

# Algorithm S1. Nonlinearity factor raw power measurements
def nonlinearity_factor_raw_power_measurements(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), taus=[3]):
    logger.info("Starting: Algorithm S1. Nonlinearity factor raw power measurements")
    # sw.set_route(monitor_port)
    # reset_attenuators()

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
        reset_attenuators()
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
        reset_attenuators()
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
    
    # # This is untested
    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(filepath)[0])
    csv_filename = f'{data_filename}.csv'
    csv_filepath = os.path.join(readable_output_dir, csv_filename)
    df.to_csv(csv_filepath, index=False)

    logger.info(f"nonlinearity_factor saved to: {filepath}")
    logger.info("Completed: Algorithm S1. Nonlinearity factor raw power measurements")
    return filepath

# Algorithm S2. Attenuator Calibration
def attenuator_calibration(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), attval=30):
    logger.info("Starting: Algorithm S2. Attenuator Calibration")
    sw.set_route(monitor_port)

    # Parameters
    N = 50
    sw.set_route(monitor_port)
    reset_attenuators()
    att1.set_att(attval)
    att_rng = find_mpm_rng(round(-attval, -1))
    logger.info(f"att_rng: {att_rng}")
    reset_attenuators()
    init_rng = find_mpm_rng(round(0, -1))
    logger.info(f"init_rng: {init_rng}")


    # Initialize an empty DataFrame to store results
    columns = ['Attenuator', 'Attenuation (dB)', 'Range', 'Power Measurement']
    df = pd.DataFrame(columns=columns)

    # Calibrate each attenuator in att_list
    init_powers = []
    rows = []  # Collect rows to add to the DataFrame
    for i, atti in enumerate(att_list):
        # Step 2: Monitor setup for initial power measurements
        mpm.set_range(init_rng)
        zero_pm()
        sw.set_route(monitor_port)
        reset_attenuators()
        time.sleep(0.3)
        mpm.get_power()
        for _ in range(N):
            init_powers.append(mpm.get_power())
        mpm.get_power()

        # Step 3: Apply attenuation and measure power
        mpm.set_range(att_rng)
        zero_pm()
        sw.set_route(monitor_port)
        reset_attenuators()
        atti.set_att(attval)
        temp_powers = []
        time.sleep(0.3)
        mpm.get_power()
        for _ in range(N):
            temp_powers.append(mpm.get_power())
        mpm.get_power()
        
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
def sweep_polarizations(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', attval=30, name='', trigger_voltage=0.01, num_pols=13, counting_time=0.5, N=3):
    """Sweep through all polarizations to find max and min counts.
    Args:
        detector (object): Detector object with a `get_counts()` method.
        step (float): Step size in degrees for the sweep.
    Returns:
        dict: A dictionary with max and min polarization settings and counts.
    """
    logger.info("Starting: Algorithm S3.1. SDE Counts Measurement - Polarization Sweep")
    logger.warning("Ensure detector fiber it spliced to SNSPD")
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
    positions = np.linspace(-50.0, 50.0, num_pols) # Max range is -99 to 100 but I want to limit these edge cases
    positions = np.round(positions, 2)
    pol_data = {}
    for n in range(num_repeats):
        for i, x in enumerate(positions):
            for j, y in enumerate(positions):
                for k, z in enumerate(positions):
                    position = (x, y, z)
                    cps = meas_counts(position, N=N, counting_time=counting_time)
                    logger.info(f"Position: {position}, counts: {np.mean(cps)}")
                    if position not in pol_data:
                        pol_data[position] = []
                    pol_data[position].append(cps)
                    logger.info(f"{round(100*(n*num_repeats*positions.size**2 + i*positions.size**2 + j*positions.size + k)/(num_repeats*positions.size**3), 2)}% Complete")

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
def SDE_Counts_Measurement(now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', pol_counts_filepath='', attval=30, name='', trigger_voltage=0.01, ):    
    logger.info("Starting: Algorithm S3.2. SDE Counts Measurement - True Counting")
    logger.warning("Ensure detector fiber it spliced to SNSPD")
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

# %%
if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())

    taus = [2.75, 2.25]
    attval = 30

    # snspd_sde_setup()
        
    # optical_switch_calibration_filepath = optical_switch_calibration(now_str=now_str, )
    
    nonlinearity_factor_filepath = nonlinearity_factor_raw_power_measurements(now_str=now_str, taus=taus)

    IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    # IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "SK3_IV_curve_data__20250110-122541.pkl")

    trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    # trigger_voltage = 0.127

    pol_counts_filepath = sweep_polarizations(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, attval=attval, name=name, num_pols=num_pols, trigger_voltage=trigger_voltage, counting_time=0.5, N=3)
    # pol_counts_filepath = os.path.join(current_file_dir, "data_sde", "SK3_pol_data_snspd_splice1__20250110-125128.pkl")

    # data_filepath = SDE_Counts_Measurement(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval, name=name, trigger_voltage=trigger_voltage)
    # attenuator_calibration_filepath = attenuator_calibration(now_str=now_str, attval=attval)
    attvals = [29, 30, 28, 31, 27, 32, 26, 25]
    for attval in attvals:
        data_filepath = SDE_Counts_Measurement(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval, name=name, trigger_voltage=trigger_voltage)
        attenuator_calibration_filepath = attenuator_calibration(now_str=now_str, attval=attval)
        

    