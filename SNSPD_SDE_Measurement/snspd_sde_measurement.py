# %% Imports, parameters, initialization
import os
import time
import pickle
import scipy
import math
import logging

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta

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
        logging.FileHandler("script_log.log", mode="a"),
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
rng_settings = [0, -10, -20, -30, -40, -50, -60]
mpm_min_max_powers = {
    0: [5e-05, 9.75e-04],
    -10: [5e-06, 9.75e-05],
    -20: [5e-07, 9.75e-06],
    -30: [5e-08, 9.75e-07],
    -40: [5e-09, 9.75e-08],
    -50: [5e-10, 9.75e-09],
    -60: [5e-11, 9.75e-10],
}

# setup constants
wavelength = 1566.314  # nm
bias_resistor = 97e3 #Ohms
init_rng = 0 #dBm

counting_time = 0.5 #s
num_pols = 13
N_init = 5

name = 'SK3'
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
    for rng in rng_settings:
        mpm.set_range(init_rng)
        time.sleep(0.3)
        mpm.get_power()
        powers = [mpm.get_power() for _ in range(N_init)]  # Collect power readings
        if all(mpm_min_max_powers[rng][1] > power > mpm_min_max_powers[rng][0] for power in powers):
            break
    init_rng = rng

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
    df = pd.DataFrame(
        {
            'power_mpm': power_mpm, 
            'power_cpm': power_cpm,
            }
            )

    # Save the DataFrame as a pickle file
    output_dir = os.path.join(current_file_dir, "data_sde")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"optical_switch_calibration_data_cpm_splice{cpm_splice}__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    df.to_pickle(filepath)

    logger.info(f"Optical switch calibration data saved to: {filepath}")
    logger.info("Completed: Algorithm S1.1 Optical Switch Calibration")

    return filepath

# Algorithm S1. Nonlinearity factor raw power measurements
def nonlinearity_factor_raw_power_measurements(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), tau=3):
    logger.info("Starting: Algorithm S1. Nonlinearity factor raw power measurements")
    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.enable()

    att2_settings = [0, tau]
    N = 25

    att_min_max_settings = {}
    for rng in rng_settings:
        mpm.set_range(rng)
        zero_pm()
        sw.set_route(monitor_port)
        reset_attenuators()
        a = -rng - 2.5
        while True:
            a -= 1
            att1.set_att(a)
            time.sleep(0.3)
            mpm.get_power()
            for _ in range(N_init):
                if mpm.get_power() > mpm_min_max_powers[rng][1]:
                    break
            else:
                continue  # Only executes if 'for' loop completes without breaking
            break
        while True:
            a += 0.1
            att1.set_att(a)
            time.sleep(0.3)
            mpm.get_power()
            for _ in range(N_init):
                if mpm.get_power() < mpm_min_max_powers[rng][1]:
                    break
            else:
                continue  # Only executes if 'for' loop completes without breaking
            break
        a_min = a - 0.1

        att2.set_att(tau)
        while True:
            a += 1
            att1.set_att(a)
            time.sleep(0.3)
            mpm.get_power()
            for _ in range(N_init):
                if mpm.get_power() < mpm_min_max_powers[rng][0]:
                    break
            else:
                continue  # Only executes if 'for' loop completes without breaking
            break
        while True:
            a -= 0.1
            att1.set_att(a)
            time.sleep(0.3)
            mpm.get_power()
            for _ in range(N_init):
                if mpm.get_power() > mpm_min_max_powers[rng][0]:
                    break
            else:
                continue  # Only executes if 'for' loop completes without breaking
            break
        a_max = a + 0.1

        att_min_max_settings[rng] = [a_min, a_max]

    att_settings = {}
    total_data = 0
    for i in range(len(rng_settings)):
        current_range = rng_settings[i]
        prev_range = rng_settings[i - 1] if i > 0 else None
        next_range = rng_settings[i + 1] if i < len(rng_settings) - 1 else None

        if prev_range is None:  # First element (use next_range)
            att_settings[current_range] = np.concatenate([
                np.arange(att_min_max_settings[current_range][0], att_min_max_settings[next_range][0], 1),
                np.arange(att_min_max_settings[next_range][0], att_min_max_settings[current_range][1], 0.1),
            ])
        elif next_range is None:  # Last element (use prev_range)
            att_settings[current_range] = np.concatenate([
                np.arange(att_min_max_settings[current_range][0], att_min_max_settings[prev_range][1], 0.1),
                np.arange(att_min_max_settings[prev_range][1], att_min_max_settings[current_range][1], 1),
            ])
        else:  # Middle elements
            att_settings[current_range] = np.concatenate([
                np.arange(att_min_max_settings[current_range][0], att_min_max_settings[prev_range][1], 0.1),
                np.arange(att_min_max_settings[prev_range][1], att_min_max_settings[current_range][0], 1),
                np.arange(att_min_max_settings[next_range][0], att_min_max_settings[current_range][1], 0.1),
            ])

        total_data += len(att_settings[rng])

    att_settings = {
        rng: [round(val, 1) for val in values]
        for rng, values in att_settings.items()
    }

    total_data *= len(att2_settings)

    data = []

    # Iterate through the ranges and settings
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
                powers_temp = []
                time.sleep(0.3)
                mpm.get_power()
                powers_temp = [mpm.get_power() for _ in range(N)]
                mpm.get_power()
                data_temp = (rng, a, att2_val, powers_temp)
                data.append(data_temp)
                logging.info(f"data_temp: {data_temp}")
                i += 1
                logging.info(f"{round(100*i/total_data, 2)}%")
                
    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Convert the data to a pandas DataFrame
    columns = ['Range', 'Attenuator 1', 'Attenuator 2', 'Power']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as a pickle file
    nonlinearity_factor_filename = f'nonlinear_calibration_data_tau{tau}__{now_str}.pkl'
    output_dir = os.path.join(current_file_dir, "data_sde")
    os.makedirs(output_dir, exist_ok=True)
    nonlinearity_factor_filepath = os.path.join(output_dir, nonlinearity_factor_filename)
    df.to_pickle(nonlinearity_factor_filepath)
    logger.info(f"nonlinearity_factor saved to: {nonlinearity_factor_filepath}")
    logger.info("Completed: Algorithm S1. Nonlinearity factor raw power measurements")
    return nonlinearity_factor_filepath

# Algorithm S2. Attenuator Calibration
def attenuator_calibration(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), attval=30):
    logger.info("Starting: Algorithm S2. Attenuator Calibration")
    sw.set_route(monitor_port)

    # Parameters
    N = 50
    for rng in rng_settings:
        mpm.set_range(rng)
        time.sleep(0.3)
        mpm.get_power()
        powers = [mpm.get_power() for _ in range(N_init)]  # Collect power readings
        if all(mpm_min_max_powers[rng][1] > power > mpm_min_max_powers[rng][0] for power in powers):
            break
    att_rng = rng

    # Initialize an empty DataFrame to store results
    columns = ['Attenuator', 'Attenuation (dB)', 'Range', 'Power Measurement']
    powers_df = pd.DataFrame(columns=columns)

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
        rows.append({
            'Attenuator': i,
            'Attenuation (dB)': attval,
            'Range': att_rng,
            'Power Measurement': temp_powers
        })

        logging.info(f"{round(100 * i / len(att_list), 2)}% completed")

    # Add initial power measurements row
    rows.append({
        'Attenuator': None,
        'Attenuation (dB)': 0,
        'Range': init_rng,
        'Power Measurement': init_powers
    })

    # Add all rows to the DataFrame at once
    powers_df = pd.concat([powers_df, pd.DataFrame(rows)], ignore_index=True)

    # Reset attenuators
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Save the calibration data to a file
    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"attenuator_calibration_data_attval{attval}__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    powers_df.to_pickle(filepath)
    logger.info(f"attenuator_calibration saved to: {filepath}")
    logger.info("Completed: Algorithm S2. Attenuator Calibration")
    return filepath


# At this point, the "detector" fiber MUST be spliced to the SNSPD
# If you have not done that yet, do so now

def meas_counts(position, N=3, counting_time=1):
    pc.set_waveplate_positions(position)
    time.sleep(0.3)  # Wait for the motion to complete
    cps = []
    for _ in range(N):
        cps.append(counter.timed_count(counting_time=counting_time)/counting_time)
    return cps

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
    this_volt = round(ic*0.80 * bias_resistor, 3)
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
                    logging.info(f"Position: {position}, counts: {np.mean(cps)}")
                    if position not in pol_data:
                        pol_data[position] = []
                    pol_data[position].append(cps)
                    logging.info(f"{round(100*(n*num_repeats*positions.size**2 + i*positions.size**2 + j*positions.size + k)/(num_repeats*positions.size**3), 2)}% Complete")

    pol_data_avg = {key: np.mean(value) for key, value in pol_data.items()}
    srs.set_voltage(0)
    srs.set_output(output=False)

    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    maxpol_settings = max(pol_data_avg, key=pol_data_avg.get)
    minpol_settings = min(pol_data_avg, key=pol_data_avg.get)
    logger.info(f"max pol settings: {maxpol_settings}, cps: {pol_data_avg[maxpol_settings]}")
    logger.info(f"min pol settings: {minpol_settings}, cps: {pol_data_avg[minpol_settings]}")

    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}_pol_data_snspd_splice{snspd_splice}__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(pol_data_avg, file)
    logger.info(f"pol_data_avg saved to: {filepath}")
    logger.info("Completed: Algorithm S3.1. SDE Counts Measurement - Polarization Sweep")
    return filepath

# Algorithm S3.2. SDE Counts Measurement - True Counting
def SDE_Counts_Measurement(now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', pol_counts_filepath='', attval=30, name='', trigger_voltage=0.01, ):    
    logger.info("Starting: Algorithm S3.2. SDE Counts Measurement - True Counting")
    logger.warning("Ensure detector fiber it spliced to SNSPD")
    with open(pol_counts_filepath, 'rb') as file:
        pol_counts = pickle.load(file)

    for i in (np.arange(len(pol_counts)-1, -1, -1)):
        if pol_counts[i][1]==None or pol_counts[i][1]==0:
            pol_counts.pop(i)

    # Find the tuple with the highest count
    maxpol_settings = max(pol_counts, key=lambda item: item[1])[0]
    minpol_settings = min(pol_counts, key=lambda item: item[1])[0]

    # Perform measurements
    ic = get_ic(IV_pickle_filepath)
    num_biases = 100
    Cur_Array = np.linspace(ic * 0.2, ic * 1.1, num_biases)

    # Dark counts measurement
    sw.set_route(monitor_port)
    for att in att_list:
        att.disable()
    logging.info("Collecting dark counts")
    Dark_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    logging.info("Dark counts collected")

    # Max and min polarization measurements
    sw.set_route(detector_port)
    for att in att_list:
        att.enable()
        att.set_att(attval)
    
    # Measure counts at max polarization
    pc.set_waveplate_positions(maxpol_settings)
    logging.info("Collecting max polarization counts")
    Maxpol_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    logging.info("Max polarization counts collected")

    # Measure counts at min polarization
    pc.set_waveplate_positions(minpol_settings)
    logging.info("Collecting min polarization counts")
    Minpol_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    logging.info("Min polarization counts collected")

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
    logger.info(f"data_dict saved to: {filepath}")
    logger.info("Completed: Algorithm S3.2. SDE Counts Measurement - True Counting")
    return filepath

# %%
if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())

    tau = 2.5
    attval = 29

    # snspd_sde_setup()
        
    # optical_switch_calibration_filepath = optical_switch_calibration(now_str=now_str, )
    
    # nonlinearity_factor_filepath = nonlinearity_factor_raw_power_measurements(now_str=now_str, tau=tau)

    IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    # IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "SK3_IV_curve_data__20250110-122541.pkl")

    trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    # trigger_voltage = 0.127

    pol_counts_filepath = sweep_polarizations(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, attval=attval, name=name, num_pols=num_pols, trigger_voltage=trigger_voltage, counting_time=0.5, N=3)
    # pol_counts_filepath = os.path.join(current_file_dir, "data_sde", "SK3_pol_data_snspd_splice1__20250110-125128.pkl")

    # nonlinearity_factor_filepath = nonlinearity_factor_raw_power_measurements(now_str=now_str, tau=2.5)

    attvals = [29, 30, 28, 31, 27, 32, 26, 33, 25, 34, 24, 35, 23, 36, 22, 37, 21, 38]
    for attval in attvals:
        data_filepath = SDE_Counts_Measurement(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval, name=name, trigger_voltage=trigger_voltage)
        attenuator_calibration_filepath = attenuator_calibration(now_str=now_str, attval=attval)

    # taus = [2, 3, 1.5]
    # for tau in taus:
    #     nonlinearity_factor_filepath = nonlinearity_factor_raw_power_measurements(now_str=now_str, tau=tau)