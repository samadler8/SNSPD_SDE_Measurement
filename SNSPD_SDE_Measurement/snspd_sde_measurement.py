# %% Imports and parameters
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
from amcc.instruments.agilent_34411a import Agilent34411A

from helper_functions import get_ic

sim900port = 5
srs = SIM928('GPIB0::2::INSTR', sim900port)
ando = AndoAQ8204('GPIB0::4::INSTR')
counter = Agilent53131a('GPIB0::5::INSTR')
cpm = Agilent8163A('GPIB0::9::INSTR')
pc = FiberControlMPC101('GPIB0::3::INSTR')
multi = Agilent34411A('GPIB0::21::INSTR')

laser_ch = 1
att1_ch = 2
att2_ch = 3
att3_ch = 4
att_list = [att1_ch, att2_ch, att3_ch]
sw_ch = 5
mpm_ch = 6

monitor_port = 1
detector_port = 2

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

wavelength = 1566.314  # nm
attval = 31 #dBm - for each attenuator
bias_resistor = 97e3 #Ohms
trigger_voltage = 0.125 #V - No clue why it's so high
counting_time = 0.75 #s
num_pols = 13


# %% Initialize and turn off everything

ando.aq82011_std_init(laser_ch)
ando.aq82011_set_lambda(laser_ch, wavelength)
ando.aq82011_disable(laser_ch)

for att_ch in att_list:
    ando.aq820133_set_att(att_ch, 0)
    ando.aq820133_disable(att_ch)
    
ando.aq82012_set_lambda(mpm_ch, wavelength)
ando.aq82012_set_range(mpm_ch, 'A')
# ando.aq82012_zero(mpm_ch)

cpm.set_wavelength(wavelength)

counter.basic_setup()
counter.set_impedance(ohms=50, channel=1)
counter.setup_timed_count(channel=1)
counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

srs.set_voltage(0)
srs.set_output(output=False)

# # %% Algorithm S1.1 Missing Algorithm (optical switch calibration)
# # For this section, the "detector" fiber must be spliced to the calibrated polarization controller (cpm)

# def optical_switch_calibration():
#     time_str = time.strftime("%Y%m%d-%H%M%S")
#     ando.aq82011_enable(laser_ch)
#     ando.aq8201418_set_route(sw_ch, monitor_port)
#     for att_ch in att_list:
#         ando.aq820133_set_att(att_ch, 0)
#         ando.aq820133_enable(att_ch)
#     N = 100
#     rgnvals = ['A', 'G']
#     ando.aq82012_set_range(mpm_ch, 'A')
#     data = []

#     for i in range(N):
#         ando.aq8201418_set_route(sw_ch, monitor_port)
#         power_mpm_dict = {}
#         for rngval in rgnvals:
#             ando.aq82012_set_range(mpm_ch, rngval)
#             time.sleep(0.1)
#             rngval_meas = ando.aq82012_get_range(mpm_ch)
#             power_mpm_dict[f'{rngval_meas}'] = ando.aq82012_get_power(mpm_ch)

#         ando.aq8201418_set_route(sw_ch, detector_port)
#         time.sleep(0.1)
#         power_cpm = cpm.read_power()

        
        
#         data_temp = (power_mpm_dict, power_cpm)
#         print(data_temp)
#         data.append(data_temp)
#         print(f"{round(100*i/N, 2)}%")

#     ando.aq82011_disable(laser_ch)
#     ando.aq8201418_set_route(sw_ch, monitor_port)
#     for att_ch in att_list:
#         ando.aq820133_set_att(att_ch, 0)
#         ando.aq820133_disable(att_ch)

#     columns = ['power_mpm_dict', 'power_cpm']
#     df = pd.DataFrame(data, columns=columns)

#     # Save the DataFrame as a pickle file
#     optical_switch_calibration_filename = f'optical_switch_calibration_data_{time_str}.pkl'
#     os.makedirs("data", exist_ok=True)
#     optical_switch_calibration_filepath = os.path.join("data", optical_switch_calibration_filename)
#     df.to_pickle(optical_switch_calibration_filepath)
#     return

# print("STARTING: Algorithm S1.1 Missing Algorithm (optical switch calibration)")
# optical_switch_calibration_filepath = optical_switch_calibration()
# print("COMPLETED: Algorithm S1.1 Missing Algorithm (optical switch calibration)")

# #
# # The "detector" fiber can now be cut and respliced to the SNSPD
# #

# %% Algorithm S1. Nonlinearity factor raw power meaurements

def nonlinearity_factor_raw_power_meaurements():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    ando.aq8201418_set_route(sw_ch, monitor_port)
    ando.aq82011_enable(laser_ch)
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, 0)
        ando.aq820133_enable(att_ch)

    N = 10
   
    base_array = range(-7, 7)
    att_setting = {}

    rng_settings = [0, -10, -20, -30, -40, -50, -60]

    # rng is range setting
    total_data = 0
    for rng in rng_settings:
        att_setting[rng] = [value for value in (val - rng - 10 for val in base_array) if value >= 0]
        total_data += len(att_setting[rng])

    total_data *= 2*N


    

    data = []

    # Iterate through the ranges and settings
    i = 0
    for rng in rng_settings:
        ando.aq82012_set_range(mpm_ch, rng)
        # ando.aq82012_zero(mpm_ch)
        for a in att_setting[rng]:
            ando.aq820133_set_att(att1_ch, a)
            for att_step in [0, 3]:
                ando.aq820133_set_att(att2_ch, att_step)
                time.sleep(0.1)
                for j in range(N):
                    power = ando.aq82012_get_power(mpm_ch)
                    # Append the data as a tuple
                    data_temp = (rng, a, att_step, j, power)
                    data.append(data_temp)
                    print(f"data_temp: {data_temp}")
                    print(f"{100*i/total_data}%")

    ando.aq82011_disable(laser_ch)
    ando.aq8201418_set_route(sw_ch, monitor_port)
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, 0)
        ando.aq820133_disable(att_ch)

    # Convert the data to a pandas DataFrame
    columns = ['Range', 'Attenuation Setting', 'Attenuation Step', 'Iteration', 'Power']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as a pickle file
    nonlinearity_factor_filename = f'nonlinearity_factor_raw_power_meaurements_data_{time_str}.pkl'
    os.makedirs("data", exist_ok=True)
    nonlinearity_factor_filepath = os.path.join("data", nonlinearity_factor_filename)
    df.to_pickle(nonlinearity_factor_filepath)

    return nonlinearity_factor_filepath

print("STARTING: Algorithm S1. Nonlinearity factor raw power meaurements")
nonlinearity_factor_filepath = nonlinearity_factor_raw_power_meaurements()
print("COMPLETED: Algorithm S1. Nonlinearity factor raw power meaurements")

# %% Algorithm S2. Attenuator Calibration

def attenuator_calibration():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    ando.aq8201418_set_route(sw_ch, monitor_port)

    # Parameters
    N = 10
    init_rng = -10
    att_rng = round((init_rng - attval)/10) * 10 
    powers_data = []

    # Step 1: Set initial range and disable all channels
    ando.aq82012_set_range(mpm_ch, init_rng)
    for att_chi in att_list:
        ando.aq820133_set_att(att_chi, 0)
        ando.aq820133_disable(att_chi)
    
    # Step 2: Zero the power meter
    # ando.aq82012_zero(mpm_ch)

    # Step 3: Enable all channels and measure initial power
    for att_chi in att_list:
        ando.aq820133_enable(att_chi)
        ando.aq820133_set_att(att_chi, 0)
    time.sleep(0.1)
    
    # Measure power N times
    initial_powers = []
    for _ in range(N):
        power = ando.aq82012_get_power(mpm_ch)
        initial_powers.append(power)
    
    # Store initial power data
    data_temp = {
        'Attenuator': None,
        'Attenuation (dB)': 0,
        'Range': init_rng,
        'Powers': initial_powers
    }
    print(data_temp)
    powers_data.append(data_temp)

    # Calibrate each attenuator in att_list
    for i, att_chi in enumerate(att_list):
        # # Disable and zero again
        # for att_ch_ in att_list:
        #     ando.aq820133_disable(att_ch_)
        # ando.aq82012_zero(mpm_ch)

        for att_chj in att_list:
            ando.aq820133_enable(att_chj)
            ando.aq820133_set_att(att_chj, 0)
        
        # Step 4: Apply attenuation and repeat measurements
        ando.aq820133_set_att(att_chi, attval)
        ando.aq82012_set_range(mpm_ch, att_rng)
        time.sleep(0.1)

        attenuated_powers = []
        for _ in range(N):
            power = ando.aq82012_get_power(mpm_ch)
            attenuated_powers.append(power)
        
        # Store attenuated power data
        data_temp = {
            'Attenuator': att_chi,
            'Attenuation (dB)': attval,
            'Range': att_rng,
            'Powers': attenuated_powers
        }
        print(data_temp)
        powers_data.append(data_temp)

        # Reset the attenuator to 0 dB
        ando.aq820133_set_att(att_chi, 0)

        print(f"{round(100*i/len(att_list), 2)}%")

    ando.aq82011_disable(laser_ch)
    ando.aq8201418_set_route(sw_ch, monitor_port)
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, 0)
        ando.aq820133_disable(att_ch)

    # Save the calibration data to a file
    attenuator_calibration_filename = f"attenuator_calibration_data_{time_str}.pkl"
    os.makedirs("data", exist_ok=True)
    attenuator_calibration_filepath = os.path.join("data", attenuator_calibration_filename)
    with open(attenuator_calibration_filepath, 'wb') as f:
        pickle.dump(powers_data, f)

    return attenuator_calibration_filepath
print("STARTING: Algorithm S2. Attenuator Calibration")
attenuator_calibration_filepath = attenuator_calibration()
print("COMPLETED: Algorithm S2. Attenuator Calibration")

# %% Name the SNSPD

name = 'SK3'

# %% Algorithm S3.0.1. SNSPD IV Curve
def SNSPD_IV_Curve(max_cur):
    time_str = time.strftime("%Y%m%d-%H%M%S")

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
    IV_pickle_file = f"{name}_IV_curve_data_{time_str}.pkl"
    os.makedirs("data", exist_ok=True)  # Ensure the "data" directory exists
    IV_pickle_filepath = os.path.join("data", IV_pickle_file)
    df.to_pickle(IV_pickle_filepath)

    print(f"IV curve data saved to {IV_pickle_filepath}")

    return IV_pickle_filepath

print("STARTING: Algorithm S3.0.1. SNSPD IV Curve")
IV_pickle_filepath = SNSPD_IV_Curve(15e-6)
print("COMPLETED: Algorithm S3.0.1. SNSPD IV Curve")

#
# At this point, the "detector" fiber MUST be spliced to the SNSPD
# If you have not done that yet, do so now
#

# %% Algorithm S3.1. SDE Counts Measurement - Polarization Sweep

def sweep_polarizations(num_pols=13, IV_pickle_filepath=''):
    """Sweep through all polarizations to find max and min counts.
    Args:
        detector (object): Detector object with a `get_counts()` method.
        step (float): Step size in degrees for the sweep.
    Returns:
        dict: A dictionary with max and min polarization settings and counts.
    """
    time_str = time.strftime("%Y%m%d-%H%M%S")

    ic = get_ic(IV_pickle_filepath)

    srs.set_voltage(0)
    srs.set_output(output=True)
    this_volt = round(ic*0.92 * bias_resistor, 3)
    srs.set_voltage(this_volt)

    ando.aq82011_enable(laser_ch)
    ando.aq8201418_set_route(sw_ch, detector_port)
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, attval)
        ando.aq820133_enable(att_ch)

    counter.basic_setup()
    counter.set_impedance(ohms=50, channel=1)
    counter.setup_timed_count(channel=1)
    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

    N = 3
    positions = np.linspace(-99.0, 100.0, num_pols)
    pol_counts = []
    for i, x in enumerate(positions):
        for j, y in enumerate(positions):
            for k, z in enumerate(positions):
                position = (x, y, z)
                pc.set_waveplate_positions(position)
                time.sleep(0.1)  # Wait for the motion to complete
                temp_counts = np.empty(N, dtype=float)
                for l in np.arange(temp_counts.size):
                    temp_counts[l] = counter.timed_count(counting_time=counting_time)
                counts = np.mean(temp_counts)
                temp_data = (position, counts)
                print(temp_data)
                pol_counts.append(temp_data)
                print(f"{round(100*(i*positions.size**2 + j*positions.size + k)/((positions.size)**3), 2)}%")
    srs.set_voltage(0)
    srs.set_output(output=False)

    ando.aq82011_disable(laser_ch)
    ando.aq8201418_set_route(sw_ch, monitor_port)
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, 0)
        ando.aq820133_disable(att_ch)

    pol_counts_filename = f"{name}_pol_counts_{time_str}.pkl"
    os.makedirs("data", exist_ok=True)
    pol_counts_filepath = os.path.join("data", pol_counts_filename)
    with open(pol_counts_filepath, "wb") as file:
        pickle.dump(pol_counts, file)
    return pol_counts_filepath

# IV_pickle_filepath = os.path.join('data', 'SK3_IV_curve_data_20241209-191713.pkl')
print("STARTING: Algorithm S3.2. SDE Counts Measurement")
pol_counts_filepath = sweep_polarizations(num_pols=num_pols, IV_pickle_filepath=IV_pickle_filepath)
print("COMPLETED: Algorithm S3.2. SDE Counts Measurement")

# %% Algorithm S3.2. SDE Counts Measurement - True Counting
def get_counts(Cur_Array):
    """
    Measures counts for a given array of currents.
    """
    srs.set_voltage(0)
    srs.set_output(output=True)

    counter.basic_setup()
    counter.set_impedance(ohms=50, channel=1)
    counter.setup_timed_count(channel=1)
    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

    ando.aq82011_enable(laser_ch)
    ando.aq8201418_set_route(sw_ch, detector_port)
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, attval)
        ando.aq820133_enable(att_ch)

    Count_Array = np.zeros(len(Cur_Array))

    for i in range(len(Cur_Array)):
        this_volt = round(Cur_Array[i] * bias_resistor, 3)
        srs.set_voltage(this_volt)
        time.sleep(0.1)
        Count_Array[i] = counter.timed_count(counting_time=counting_time)
        print(f"Voltage: {this_volt} V, Counts: {Count_Array[i]}")
    
    srs.set_voltage(0)
    srs.set_output(output=False)

    ando.aq82011_disable(laser_ch)
    ando.aq8201418_set_route(sw_ch, monitor_port)
    for att_ch in att_list:
        ando.aq820133_set_att(att_ch, 0)
        ando.aq820133_disable(att_ch)

    return Count_Array

def SDE_Counts_Measurement(IV_pickle_filepath = '', pol_counts_filepath = ''):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    
    with open(pol_counts_filepath, 'rb') as file:
        pol_counts = pickle.load(file)

    # Find the tuple with the highest count
    maxpol_settings = max(pol_counts, key=lambda item: item[1])[0]
    minpol_settings = min(pol_counts, key=lambda item: item[1])[0]

    # Perform measurements
    ic = get_ic(IV_pickle_filepath)
    num_biases = 100
    Cur_Array = np.linspace(ic * 0.2, ic * 1.1, num_biases)

    # Dark counts measurement
    ando.aq8201418_set_route(sw_ch, monitor_port)
    ando.aq82011_disable(laser_ch)
    for att_ch in att_list:
        ando.aq820133_disable(att_ch)
    Dark_Count_Array = get_counts(Cur_Array)
    print("Got dark counts")

    # Max and min polarization measurements
    ando.aq8201418_set_route(sw_ch, detector_port)
    ando.aq82011_enable(laser_ch)
    for att_ch in att_list:
        ando.aq820133_enable(att_ch)
        ando.aq820133_set_att(att_ch, attval)
    
    # Measure counts at max polarization
    pc.set_waveplate_positions(maxpol_settings)
    Maxpol_Count_Array = get_counts(Cur_Array)
    print("Got counts for max polarization")

    # Measure counts at min polarization
    pc.set_waveplate_positions(minpol_settings)
    Minpol_Count_Array = get_counts(Cur_Array)
    print("Got counts for min polarization")

    # Save data
    data_dict = {
        'Cur_Array': list(Cur_Array),
        'Dark_Count_Array': list(Dark_Count_Array),
        'Maxpol_Count_Array': list(Maxpol_Count_Array),
        'Minpol_Count_Array': list(Minpol_Count_Array),
        'Maxpol_Settings': maxpol_settings,
        'Minpol_Settings': minpol_settings,
    }
    data_filename = f"{name}_data_dict_{time_str}.pkl"
    os.makedirs("data", exist_ok=True)
    data_filepath = os.path.join("data", data_filename)
    with open(data_filepath, "wb") as file:
        pickle.dump(data_dict, file)
    return data_filepath

print("STARTING: Algorithm S3.2. SDE Counts Measurement")
data_filepath = SDE_Counts_Measurement(IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath)
print("COMPLETED: Algorithm S3.2. SDE Counts Measurement")

# # Call Algorithm S2
# attenuator_calibration()
# %%
