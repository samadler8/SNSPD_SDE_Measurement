#%% Imports and parameters
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

attval = 10 #dBm - for each attenuator
bias_resistor = 100e3 #Ohms

#%% Initialize and turn on laser

wavelength = 1566.314  # nm
ando.aq82011_std_init(laser_ch)
ando.aq82011_set_lambda(laser_ch, wavelength)
ando.aq82011_disable(laser_ch)

for att_ch in att_list:
    ando.aq820133_enable(att_ch)
for att_ch in att_list:
    ando.aq820133_set_att(att_ch, 0)

ando.aq82012_set_lambda(mpm_ch, wavelength)
cpm.set_wavelength(wavelength)

# Set power meter range and zero
ando.aq82012_set_range(mpm_ch, 'A')
# ando.aq82012_zero(mpm_ch)

ando.aq82011_enable(laser_ch)

#%% Algorithm S1.1 Missing Algorithm (optical switch calibration)
# For this section, the "detector" fiber must be spliced to the calibrated polarization controller (cpm)

def optical_switch_calibration():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    N = 100
    rgnvals = ['A', 'G']
    ando.aq82012_set_range(mpm_ch, 'A')
    data = []

    for i in range(N):
        ando.aq8201418_set_route(sw_ch, monitor_port)
        power_mpm_dict = {}
        for rngval in rgnvals:
            ando.aq82012_set_range(mpm_ch, rngval)
            time.sleep(0.1)
            rngval_meas = ando.aq82012_get_range(mpm_ch)
            power_mpm_dict[f'{rngval_meas}'] = ando.aq82012_get_power(mpm_ch)

        ando.aq8201418_set_route(sw_ch, detector_port)
        time.sleep(0.1)
        power_cpm = cpm.read_power()

        
        
        data_temp = (power_mpm_dict, power_cpm)
        print(data_temp)
        data.append(data_temp)
        print(f"{100*i/N}%")

    columns = ['power_mpm_dict', 'power_cpm']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as a pickle file
    pickle_file = f'optical_switch_calibration_data_{time_str}.pkl'
    os.makedirs("data", exist_ok=True)
    pickle_filepath = os.path.join("data", pickle_file)
    df.to_pickle(pickle_filepath)
    return

print("STARTING: Algorithm S1.1 Missing Algorithm (optical switch calibration)")
optical_switch_calibration()
print("COMPLETED: Algorithm S1.1 Missing Algorithm (optical switch calibration)")
# The "detector" fiber can now be cut and respliced to the SNSPD

#%% Algorithm S1. Nonlinearity factor raw power meaurements

def nonlinearity_factor_raw_power_meaurements():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    ando.aq8201418_set_route(sw_ch, monitor_port)

    N = 50
   
    base_array = range(-7, 7)
    att_setting = {}

    rng_settings = [0, -10, -20, -30, -40, -50, -60]

    # rng is range setting
    for rng in rng_settings:
        att_setting[rng] = [value for value in (val - rng - 10 for val in base_array) if value >= 0]

    data = []

    # Iterate through the ranges and settings
    for i, rng in enumerate(rng_settings):
        ando.aq82012_set_range(mpm_ch, rng)
        # ando.aq82012_zero(mpm_ch)
        for j, a in enumerate(att_setting[rng]):
            ando.aq820133_set_att(att1_ch, a)
            for k, att_step in enumerate([0, 3]):
                ando.aq820133_set_att(att2_ch, att_step)
                for l in range(N):
                    power = ando.aq82012_get_power(mpm_ch)
                    # Append the data as a tuple
                    data_temp = (rng, a, att_step, l, power)
                    data.append(data_temp)
                    print(f"data_temp: {data_temp}")
        print(f"{100*(i)/(len(rng_settings))}%")

    # Convert the data to a pandas DataFrame
    columns = ['Range', 'Attenuation Setting', 'Attenuation Step', 'Iteration', 'Power']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as a pickle file
    pickle_file = f'nonlinearity_factor_raw_power_meaurements_data_{time_str}.pkl'
    os.makedirs("data", exist_ok=True)
    pickle_filepath = os.path.join("data", pickle_file)
    df.to_pickle(pickle_filepath)
    return

print("STARTING: Algorithm S1. Nonlinearity factor raw power meaurements")
nonlinearity_factor_raw_power_meaurements()
print("COMPLETED: Algorithm S1. Nonlinearity factor raw power meaurements")

#%% Algorithm S2. Attenuator Calibration

def attenuator_calibration():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    ando.aq8201418_set_route(sw_ch, monitor_port)

    # Parameters
    N = 100
    init_rng = -10
    att_rng = -20
    powers_data = []

    # Step 1: Set initial range and disable all channels
    ando.aq82012_set_range(mpm_ch, init_rng)
    for att_ch_ in att_list:
        ando.aq820133_set_att(att_ch, 0)
        ando.aq820133_disable(att_ch_)
    
    # Step 2: Zero the power meter
    # ando.aq82012_zero(mpm_ch)

    # Step 3: Enable all channels and measure initial power
    for att_ch in att_list:
        ando.aq820133_enable(att_ch)
        ando.aq820133_set_att(att_ch, 0)
    
    # Measure power N times
    initial_powers = []
    for _ in range(N):
        time.sleep(0.1)
        power = ando.aq82012_get_power(mpm_ch)
        initial_powers.append(power)
    
    # Store initial power data
    data_temp = {
        'Attenuator': att_ch,
        'Attenuation (dB)': 0,
        'Range': init_rng,
        'Powers': initial_powers
    }
    print(data_temp)
    powers_data.append(data_temp)

    # Calibrate each attenuator in att_list
    for i, att_ch in enumerate(att_list):
        # # Disable and zero again
        # for att_ch_ in att_list:
        #     ando.aq820133_disable(att_ch_)
        # ando.aq82012_zero(mpm_ch)

        for att_ch_ in att_list:
            ando.aq820133_enable(att_ch_)
            ando.aq820133_set_att(att_ch_, 0)
        
        # Step 4: Apply attenuation and repeat measurements
        ando.aq820133_set_att(att_ch, attval)
        ando.aq82012_set_range(mpm_ch, att_rng)

        attenuated_powers = []
        for _ in range(N):
            time.sleep(0.1)
            power = ando.aq82012_get_power(mpm_ch)
            attenuated_powers.append(power)
        
        # Store attenuated power data
        data_temp = {
            'Attenuator': att_ch,
            'Attenuation (dB)': attval,
            'Range': att_rng,
            'Powers': attenuated_powers
        }
        print(data_temp)
        powers_data.append(data_temp)

        # Reset the attenuator to 0 dB
        ando.aq820133_set_att(att_ch, 0)

        print(f"{i/len(att_list)}%")

    # Save the calibration data to a file
    output_file = f"attenuator_calibration_data_{time_str}.pkl"
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(powers_data, f)
    return
print("STARTING: Algorithm S2. Attenuator Calibration")
attenuator_calibration()
print("COMPLETED: Algorithm S2. Attenuator Calibration")

#%% Algorithm S3.0.1. SNSPD IV Curve
def SNSPD_IV_Curve(max_cur):
    time_str = time.strftime("%Y%m%d-%H%M%S")

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
    srs.set_voltage(0)
    srs.set_output(output=False)

    # Combine data into a pandas DataFrame
    IV_curve_data = {
        "Current": Cur_Array,
        "Voltage": Volt_Meas
    }
    df = pd.DataFrame(IV_curve_data)

    # Save the DataFrame as a pickle file
    pickle_file = f"IV_curve_data_{time_str}.pkl"
    os.makedirs("data", exist_ok=True)  # Ensure the "data" directory exists
    pickle_filepath = os.path.join("data", pickle_file)
    df.to_pickle(pickle_filepath)

    print(f"IV curve data saved to {pickle_filepath}")

    
    
    # Find critical current
    df = pd.read_pickle(pickle_filepath)

    # Find the first current where voltage exceeds 1e-5
    threshold = 1e-5
    filtered_df = df[df['Voltage'] > threshold]  # Filter rows where Voltage > threshold

    if not filtered_df.empty:
        ic = filtered_df['Current'].iloc[0]  # Get the first current value
        print(f"Got Critical current: {ic}")
    else:
        print("No voltage exceeded the threshold of 1e-5.")
        ic = None
    return ic

print("STARTING: Algorithm S3.0.1. SNSPD IV Curve")
ic = SNSPD_IV_Curve(15e-6)
print("COMPLETED: Algorithm S3.0.1. SNSPD IV Curve")
# At this point, the "detector" fiber MUST be spliced to the SNSPD
# If you have not done that yet, do so now
#%% Algorithm S3.1. SDE Counts Measurement - Polarization Sweep

counting_time = 0.75

def get_counts(Cur_Array):
    """
    Measures counts for a given array of currents.
    """
    srs.set_voltage(0)
    srs.set_output(output=True)

    Count_Array = np.zeros(len(Cur_Array))

    for i in range(len(Cur_Array)):
        this_volt = round(Cur_Array[i] * bias_resistor, 3)
        srs.set_voltage(this_volt)
        time.sleep(0.1)
        Count_Array[i] = counter.timed_count(counting_time=counting_time)
        print(f"Voltage: {this_volt} V, Counts: {Count_Array[i]}")
    
    srs.set_voltage(0)
    srs.set_output(output=False)
    return Count_Array

def sweep_polarizations(step=5.0):
    """Sweep through all polarizations to find max and min counts.
    Args:
        detector (object): Detector object with a `get_counts()` method.
        step (float): Step size in degrees for the sweep.
    Returns:
        dict: A dictionary with max and min polarization settings and counts.
    """
    time_str = time.strftime("%Y%m%d-%H%M%S")

    srs.set_voltage(0)
    srs.set_output(output=True)
    this_volt = round(ic*0.97 * bias_resistor, 3)
    srs.set_voltage(this_volt)

    N = 10
    positions = np.arange(-99.0, 100.0, step)
    pol_counts = []
    for i, x in enumerate(positions):
        for j, y in enumerate(positions):
            for k, z in enumerate(positions):
                pc.set_waveplate_position('X', x)
                pc.set_waveplate_position('Y', y)
                pc.set_waveplate_position('Z', z)
                time.sleep(0.1)  # Wait for the motion to complete
                temp_counts = np.empty(N, dtype=float)
                for l in np.arange(temp_counts.size):
                    temp_counts[l] = counter.timed_count(counting_time=counting_time*2)
                counts = np.mean(temp_counts)
                pol_counts.append(((x, y, z), counts))
                print(f"{(i*j*k + j*k + k)/((len(positions))**3)}")
    srs.set_voltage(0)
    srs.set_output(output=False)

    pol_counts_filename = f"pol_counts_{time_str}.pkl"
    os.makedirs("data", exist_ok=True)
    pol_counts_filepath = os.path.join("data", pol_counts_filename)
    with open(pol_counts_filepath, "wb") as file:
        pickle.dump(pol_counts, file)
    return

print("STARTING: Algorithm S3.2. SDE Counts Measurement")
sweep_polarizations()
print("COMPLETED: Algorithm S3.2. SDE Counts Measurement")

#%% Algorithm S3.2. SDE Counts Measurement - True Counting

def SDE_Counts_Measurement():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    pol_counts_filename = "pol_counts.pkl"
    pol_counts_filepath = os.path.join("data", pol_counts_filename)
    with open(pol_counts_filepath, 'rb') as file:
        pol_counts = pickle.load(file)

    # Find the tuple with the highest count
    maxpol_settings = max(pol_counts, key=lambda item: item[1])[0]
    minpol_settings = min(pol_counts, key=lambda item: item[1])[0]

    # Perform measurements
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
    pc.set(maxpol_settings)
    Maxpol_Count_Array = get_counts(Cur_Array)
    print("Got counts for max polarization")

    # Measure counts at min polarization
    pc.set(minpol_settings)
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
    data_filename = f"data_dict_{time_str}.pkl"
    os.makedirs("data", exist_ok=True)
    data_filepath = os.path.join("data", data_filename)
    with open(data_filepath, "wb") as file:
        pickle.dump(data_dict, file)
    return

print("STARTING: Algorithm S3.2. SDE Counts Measurement")
SDE_Counts_Measurement()
print("COMPLETED: Algorithm S3.2. SDE Counts Measurement")

# Call Algorithm S2
attenuator_calibration()
