import os
import time
import pickle

import numpy as np
import pandas as pd

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.ando_aq8204 import AndoAQ8204
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.agilent_8163a import Agilent8163A
from amcc.instruments.fiberControl_MPC101 import FiberControlMPC101

sim900port = 5
srs = SIM928('GPIB0::2::INSTR', sim900port)
ando = AndoAQ8204('GPIB0::4::INSTR')
counter = Agilent53131a('GPIB0::5::INSTR')
cpm = Agilent8163A('GPIB0::9::INSTR')
pc = FiberControlMPC101('GPIB0::3::INSTR')

laser_ch = 1
att1_ch = 2
att2_ch = 3
att3_ch = 4
att_list = [att1_ch, att2_ch, att3_ch]
sw_ch = 5
mpm_ch = 6

monitor_port = '1'
detector_port = '2'

attval = 60 #dB
rngval = 10
ic = 10e-6 #A
vpol = 10

bias_resistor = 100e3 #Ohms

#%% Initialize and turn on laser
laser_wavelength = 1566.314  # nm
ando.aq82011_std_init(laser_ch)
ando.aq82011_set_lambda(laser_ch, laser_wavelength)
ando.aq82011_disable(laser_ch)

ando.aq82012_set_lambda(mpm_ch, laser_wavelength)

# Set power meter range and zero
ando.aq82012_set_range(mpm_ch, rngval)
ando.aq82012_zero(mpm_ch)

ando.aq82011_enable(laser_ch)
#%% Alforithm S1.1 Missing Algorithm (optical switch calibration)
# For this section, the "detector" fiber must be spliced to the calibrated polarization controller (cpm)
def optical_switch_calibration():
    N = 100

    data = []

    for _ in range(N):
        ando.aq82014_set_route(sw_ch, monitor_port)
        power_mpm = ando.aq82012_get_power(mpm_ch)

        ando.aq82014_set_route(sw_ch, detector_port)
        power_cpm = cpm.read_power()

        data.append((power_mpm, power_cpm))

    columns = ['power_mpm', 'power_cpm']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as a pickle file
    pickle_file = 'optical_switch_calibration_data.pkl'
    os.makedirs("data", exist_ok=True)
    pickle_filepath = os.path.join("data", pickle_file)
    df.to_pickle(pickle_filepath)
# The "detector" fiber can now be cut and respliced to the SNSPD
#%% Algorithm S1. Nonlinearity factor raw power meaurements
def nonlinearity_factor_raw_power_meaurements():
    ando.aq82014_set_route(sw_ch, monitor_port)

    N = 10
    xlist = [20, 15]
    xlist.extend(np.arange(10, 0.9, -0.5))  # Use extend to append elements from the array
    xlist.extend(np.arange(0.95, 0.5, -0.05))  # Fix the step size to avoid empty arrays

    xlist = np.array(xlist)  # Convert to a numpy array
    base_array = np.round(10 - 10 * np.log10(xlist))  # Use np.round for rounding
    base_array = base_array - np.min(base_array)
    att_setting = {}

    # rng is range setting
    for rng in [-10, -20, -30, -40, -50, -60]:
        att_setting[rng] = base_array - (rng+10) -3


    data = []

    # Iterate through the ranges and settings
    for rng in [-10, -20, -30, -40, -50, -60]:
        ando.aq82012_set_range(mpm_ch, rng)
        ando.aq82012_zero(mpm_ch)
        for a in att_setting[rng]:
            ando.aq820133_set_att(att1_ch, a)
            for att_step in [0, 3]:
                ando.aq820133_set_att(att2_ch, att_step)
                for i in range(N):
                    power = ando.aq82012_get_power(mpm_ch)
                    # Append the data as a tuple
                    data.append((rng, a, att_step, i, power))

    # Convert the data to a pandas DataFrame
    columns = ['Range', 'Attenuation Setting', 'Attenuation Step', 'Iteration', 'Power']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as a pickle file
    pickle_file = 'nonlinearity_factor_raw_power_meaurements_data.pkl'
    os.makedirs("data", exist_ok=True)
    pickle_filepath = os.path.join("data", pickle_file)
    df.to_pickle(pickle_filepath)



#%% Algorithm S2. Attenuator Calibration
def attenuator_calibration():
    ando.aq82014_set_route(sw_ch, monitor_port)

    # Parameters
    N = 5
    init_rng = -10
    powers_data = []

    # Initialize all attenuators to 0 dB
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, 0)

    # Calibrate each attenuator in att_list
    for att_ch in att_list:
        # Step 1: Set initial range and disable all channels
        ando.aq82012_set_range(mpm_ch, init_rng)
        for att_ch_ in att_list:
            ando.aq820133_disable(att_ch_)
        
        # Step 2: Zero the power meter
        ando.aq82012_zero(mpm_ch)

        # Step 3: Enable all channels and measure initial power
        for att_ch_ in att_list:
            ando.aq820133_enable(att_ch_)
        
        # Measure power N times
        initial_powers = []
        for _ in range(N):
            power = ando.aq82012_get_power(mpm_ch)
            initial_powers.append(power)
        
        # Store initial power data
        powers_data.append({
            'Attenuator': att_ch,
            'Attenuation (dB)': 0,
            'Range': init_rng,
            'Powers': initial_powers
        })

        # Step 4: Apply attenuation and repeat measurements
        ando.aq820133_set_att(att_ch, attval)
        ando.aq82012_set_range(mpm_ch, rngval)

        # Disable and zero again
        for att_ch_ in att_list:
            ando.aq820133_disable(att_ch_)
        ando.aq82012_zero(mpm_ch)

        # Enable and measure power again
        for att_ch_ in att_list:
            ando.aq820133_enable(att_ch_)
        
        attenuated_powers = []
        for _ in range(N):
            power = ando.aq82012_get_power(mpm_ch)
            attenuated_powers.append(power)
        
        # Store attenuated power data
        powers_data.append({
            'Attenuator': att_ch,
            'Attenuation (dB)': attval,
            'Range': rngval,
            'Powers': attenuated_powers
        })

        # Reset the attenuator to 0 dB
        ando.aq820133_set_att(att_ch, 0)

    # Save the calibration data to a file
    output_file = "attenuator_calibration_data.pkl"
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", output_file)

    with open(output_path, 'wb') as f:
        pickle.dump(powers_data, f)


#%% Algorithm S3. SDE Counts Measurement
# At this point, the "detector" fiber MUST be spliced to the SNSPD
# If you have not done that yet, do so now
counting_time = 0.75

def get_counts(Cur_Array):
    """
    Measures counts for a given array of currents.
    """
    srs.set_volt(0)
    srs.enable()

    Count_Array = np.zeros(len(Cur_Array))

    for i in range(len(Cur_Array)):
        this_volt = round(Cur_Array[i] * 1e-6 * bias_resistor, 3)
        srs.set_volt(this_volt)
        time.sleep(0.1)
        Count_Array[i] = counter.timed_count(counting_time=counting_time)
        print(f"Voltage: {this_volt} V, Counts: {Count_Array[i]}")
    
    srs.set_volt(0)
    srs.disable()
    return Count_Array

def maximize_counts(pc, counter, maxpol_output):
    """
    Adjusts polarization controller to maximize counts.
    """
    best_settings = None
    max_counts = -np.inf

    # Iterate through polarization settings (example assumes 3D grid search)
    for setting in pc.generate_settings():
        pc.set(setting)
        time.sleep(0.1)  # Allow time for the setting to stabilize
        counts = counter.timed_count(counting_time=counting_time)
        print(f"Testing setting: {setting}, Counts: {counts}")

        if counts > max_counts:
            max_counts = counts
            best_settings = setting

    maxpol_output['counts'] = max_counts
    return best_settings

def minimize_counts(pc, counter, minpol_output):
    """
    Adjusts polarization controller to minimize counts.
    """
    best_settings = None
    min_counts = np.inf

    # Iterate through polarization settings (example assumes 3D grid search)
    for setting in pc.generate_settings():
        pc.set(setting)
        time.sleep(0.1)  # Allow time for the setting to stabilize
        counts = counter.timed_count(counting_time=counting_time)
        print(f"Testing setting: {setting}, Counts: {counts}")

        if counts < min_counts:
            min_counts = counts
            best_settings = setting

    minpol_output['counts'] = min_counts
    return best_settings

def SDE_Counts_Measurement():

    # Perform measurements
    num_biases = 100
    Cur_Array = np.round(np.linspace(0, ic * 1.1, num_biases), 8)

    # Set all attenuators to `attval`
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, attval)

    # Dark counts measurement
    ando.aq82014_set_route(sw_ch, monitor_port)
    for att_ch in att_list:
        ando.aq820133_disable(att_ch)
    Dark_Count_Array = get_counts(Cur_Array)

    # Max and min polarization measurements
    ando.aq82014_set_route(sw_ch, detector_port)
    for att_ch in att_list:
        ando.aq820133_enable(att_ch)
    

    # Maximize counts
    srs.set_volt(vpol)
    out_maxpol = {}
    maxpol_settings = maximize_counts(pc, counter, out_maxpol)

    # Minimize counts
    out_minpol = {}
    minpol_settings = minimize_counts(pc, counter, out_minpol)

    # Measure counts at max polarization
    pc.set(maxpol_settings)
    Maxpol_Count_Array = get_counts(Cur_Array)

    # Measure counts at min polarization
    pc.set(minpol_settings)
    Minpol_Count_Array = get_counts(Cur_Array)

    # Save data
    data_dict = {
        'Cur_Array': list(Cur_Array),
        'Dark_Count_Array': list(Dark_Count_Array),
        'Maxpol_Count_Array': list(Maxpol_Count_Array),
        'Minpol_Count_Array': list(Minpol_Count_Array),
        'Maxpol_Settings': maxpol_settings,
        'Minpol_Settings': minpol_settings,
    }
    data_filename = "data_dict.pkl"
    os.makedirs("data", exist_ok=True)
    data_filepath = os.path.join("data", data_filename)
    with open(data_filepath, "wb") as file:
        pickle.dump(data_dict, file)

    # Reset for further measurements
    ando.aq82014_set_route(sw_ch, monitor_port)

# Call Algorithm S2
attenuator_calibration()