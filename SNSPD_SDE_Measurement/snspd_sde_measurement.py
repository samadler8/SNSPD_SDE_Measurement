# %% Imports, parameters, initialization
import os
import time
import pickle
import scipy

import numpy as np
import pandas as pd

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

from SNSPD_SDE_Measurement.measurement_helpers import *

srs = SIM928('GPIB0::2::INSTR', 5)
laser = AndoAQ82011('GPIB0::4::INSTR', 1)
att1 = AndoAQ820133('GPIB0::4::INSTR', 2)
att2 = AndoAQ820133('GPIB0::4::INSTR', 3)
att3 = AndoAQ820133('GPIB0::4::INSTR', 4)
sw = AndoAQ8201418('GPIB0::4::INSTR', 5)
mpm = AndoAQ82012('GPIB0::4::INSTR', 6)
counter = Agilent53131a('GPIB0::5::INSTR')
cpm = Agilent8163A('GPIB0::9::INSTR')
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

wavelength = 1566.314  # nm
attval = 31 #dBm - for each attenuator
max_cur = 15e-6 # A
bias_resistor = 97e3 #Ohms
counting_time = 0.5 #s
num_pols = 13
name = 'SK3'


# Initialize and turn off everything
laser.std_init()
laser.set_lambda(wavelength)
laser.enable()

for att in att_list:
    att.set_att(0)
    att.disable()
    
mpm.set_lambda(wavelength)
mpm.set_range('A')
# mpm.zero()

cpm.set_wavelength(wavelength)

counter.basic_setup()
counter.set_impedance(ohms=50, channel=1)
counter.setup_timed_count(channel=1)

srs.set_voltage(0)
srs.set_output(output=False)

# Algorithm S1.1 Missing Algorithm (optical switch calibration)
# For this section, the "detector" fiber must be spliced to the calibrated polarization controller (cpm)

def optical_switch_calibration(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), ):
    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.enable()
    N = 100
    rgnvals = ['A', 'G']
    mpm.set_range('A')
    data = []

    for i in range(N):
        sw.set_route(monitor_port)
        power_mpm_dict = {}
        for rngval in rgnvals:
            mpm.set_range(rngval)
            time.sleep(0.1)
            rngval_meas = mpm.get_range()
            power_mpm_dict[f'{rngval_meas}'] = mpm.get_power()

        sw.set_route(detector_port)
        time.sleep(0.1)
        power_cpm = cpm.read_power()
        
        data_temp = (power_mpm_dict, power_cpm)
        print(data_temp)
        data.append(data_temp)
        print(f"{round(100*i/N, 2)}%")

    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    columns = ['power_mpm_dict', 'power_cpm']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as a pickle file
    optical_switch_calibration_filename = f'optical_switch_calibration_data__{now_str}.pkl'
    os.makedirs("data_sde", exist_ok=True)
    optical_switch_calibration_filepath = os.path.join("data_sde", optical_switch_calibration_filename)
    df.to_pickle(optical_switch_calibration_filepath)
    return

# Algorithm S1. Nonlinearity factor raw power meaurements
def nonlinearity_factor_raw_power_meaurements(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), ):
    
    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.enable()

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
        mpm.set_range(rng)
        # mpm.zero()
        for a in att_setting[rng]:
            att1.set_att(a)
            for att_step in [0, 3]:
                att2.set_att(att_step)
                time.sleep(0.1)
                for j in range(N):
                    power = mpm.get_power()
                    # Append the data as a tuple
                    data_temp = (rng, a, att_step, j, power)
                    data.append(data_temp)
                    print(f"data_temp: {data_temp}")
                    print(f"{100*i/total_data}%")

    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Convert the data to a pandas DataFrame
    columns = ['Range', 'Attenuation Setting', 'Attenuation Step', 'Iteration', 'Power']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame as a pickle file
    nonlinearity_factor_filename = f'nonlinearity_factor_raw_power_meaurements_data__{now_str}.pkl'
    os.makedirs("data_sde", exist_ok=True)
    nonlinearity_factor_filepath = os.path.join("data_sde", nonlinearity_factor_filename)
    df.to_pickle(nonlinearity_factor_filepath)

    return nonlinearity_factor_filepath

# Algorithm S2. Attenuator Calibration
def attenuator_calibration(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), ):
    sw.set_route(monitor_port)

    # Parameters
    N = 10
    init_rng = -10
    att_rng = round((init_rng - attval) / 10) * 10 

    # Initialize an empty DataFrame to store results
    columns = ['Attenuator', 'Attenuation (dB)', 'Range', 'Power Measurement']
    powers_df = pd.DataFrame(columns=columns)

    # Calibrate each attenuator in att_list
    init_powers = []
    for i, atti in enumerate(att_list):
        # Step 1: Detector setup
        sw.set_route(detector_port)
        for attj in att_list:
            attj.disable()
        time.sleep(0.1)
        mpm.set_range('A')
        time.sleep(0.1)
        mpm.zero()

        # Step 2: Monitor setup for initial power measurements
        sw.set_route(monitor_port)
        for attj in att_list:
            attj.set_att(0)
            attj.enable()
        time.sleep(0.1)
        mpm.set_range(init_rng)

        # Measure initial power
        time.sleep(0.1)
        for _ in range(N):
            init_powers.append(mpm.get_power())
        
        # Reset for next measurement
        sw.set_route(detector_port)
        for attj in att_list:
            attj.disable()
        time.sleep(0.1)
        mpm.set_range('A')
        time.sleep(0.1)
        mpm.zero()

        # Step 3: Apply attenuation and measure power
        sw.set_route(monitor_port)
        for attj in att_list:
            attj.set_att(0)
            attj.enable()
        atti.set_att(attval)
        mpm.set_range(att_rng)

        temp_powers = []
        time.sleep(0.1)
        for _ in range(N):
            temp_powers.append(mpm.get_power())
        powers_df = powers_df.append({
            'Attenuator': atti,
            'Attenuation (dB)': attval,
            'Range': mpm.get_range(),
            'Power': temp_powers
        }, ignore_index=True)

        print(f"{round(100 * i / len(att_list), 2)}% completed")

    powers_df = powers_df.append({
        'Attenuator': None,
        'Attenuation (dB)': 0,
        'Range': mpm.get_range(),
        'Power': init_powers
    }, ignore_index=True)

    # Reset attenuators
    for att in att_list:
        att.set_att(0)
        att.disable()

    # Save the calibration data to a file
    os.makedirs("data_sde", exist_ok=True)
    attenuator_calibration_filename = f"attenuator_calibration_data__{now_str}.pkl"
    attenuator_calibration_filepath = os.path.join("data_sde", attenuator_calibration_filename)
    powers_df.to_pickle(attenuator_calibration_filepath)

    return attenuator_calibration_filepath


# At this point, the "detector" fiber MUST be spliced to the SNSPD
# If you have not done that yet, do so now

# Algorithm S3.1. SDE Counts Measurement - Polarization Sweep
def sweep_polarizations(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', name='', trigger_voltage=0.01, num_pols=13, counting_time=0.5, N=1):
    """Sweep through all polarizations to find max and min counts.
    Args:
        detector (object): Detector object with a `get_counts()` method.
        step (float): Step size in degrees for the sweep.
    Returns:
        dict: A dictionary with max and min polarization settings and counts.
    """
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

    # bounds = [(-99, 100)] * 3
    # def neg_meas_counts(position, *args):
    #     return -meas_counts(position, *args)
    # initial_guess = np.array([0, 0, 0])
    # res_min = scipy.optimize.minimize(meas_counts, initial_guess, args=(instruments, N, counting_time), bounds=bounds)
    # res_max = scipy.optimize.minimize(neg_meas_counts, initial_guess, args=(instruments, N, counting_time), bounds=bounds)
    # pol_counts = [(res_min['x'], res_min['fun']), (res_max['x'], res_max['fun'])]
    # print(pol_counts)

    positions = np.linspace(-99.0, 100.0, num_pols)
    pol_counts = []
    for i, x in enumerate(positions):
        x = round(x, 3)
        for j, y in enumerate(positions):
            y = round(y, 3)
            for k, z in enumerate(positions):
                z = round(z, 3)
                position = (x, y, z)
                counts = meas_counts(position, instruments, N=N, counting_time=counting_time)
                temp_data = (position, counts)
                print(temp_data)
                pol_counts.append(temp_data)
                print(f"{round(100*(i*positions.size**2 + j*positions.size + k)/((positions.size)**3), 2)}%")
    
    
    srs.set_voltage(0)
    srs.set_output(output=False)

    sw.set_route(monitor_port)
    for att in att_list:
        att.set_att(0)
        att.disable()

    pol_counts_filename = f"{name}_pol_counts__{now_str}.pkl"
    os.makedirs("data_sde", exist_ok=True)
    pol_counts_filepath = os.path.join("data_sde", pol_counts_filename)
    with open(pol_counts_filepath, "wb") as file:
        pickle.dump(pol_counts, file)
    return pol_counts_filepath

# Algorithm S3.2. SDE Counts Measurement - True Counting
def SDE_Counts_Measurement(now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', pol_counts_filepath='', name='', trigger_voltage=0.01, ):    
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
    Dark_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    print("Got dark counts")

    # Max and min polarization measurements
    sw.set_route(detector_port)
    for att in att_list:
        att.enable()
        att.set_att(attval)
    
    # Measure counts at max polarization
    pc.set_waveplate_positions(maxpol_settings)
    Maxpol_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    print("Got counts for max polarization")

    # Measure counts at min polarization
    pc.set_waveplate_positions(minpol_settings)
    Minpol_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    print("Got counts for min polarization")

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
    data_filename = f"{name}_data_dict__{now_str}.pkl"
    os.makedirs("data_sde", exist_ok=True)
    data_filepath = os.path.join("data_sde", data_filename)
    with open(data_filepath, "wb") as file:
        pickle.dump(data_dict, file)

    return data_filepath

# %%
if __name__ == '__main__':
    # now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    # print("STARTING: Algorithm S1.1 Missing Algorithm (optical switch calibration)")
    # optical_switch_calibration_filepath = optical_switch_calibration(now_str=now_str, )
    # print("COMPLETED: Algorithm S1.1 Missing Algorithm (optical switch calibration)")

    # #
    # # The "detector" fiber can now be cut and respliced to the SNSPD
    # #

    # now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    # print("STARTING: Algorithm S1. Nonlinearity factor raw power meaurements")
    # nonlinearity_factor_filepath = nonlinearity_factor_raw_power_meaurements(now_str=now_str, )
    # print("COMPLETED: Algorithm S1. Nonlinearity factor raw power meaurements")

    # now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    # print("STARTING: Algorithm S3.0.1. SNSPD IV Curve")
    # IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name):
    # print("COMPLETED: Algorithm S3.0.1. SNSPD IV Curve")

    #
    # At this point, the "detector" fiber MUST be spliced to the SNSPD
    # If you have not done that yet, do so now
    #

    # now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    # print("STARTING: Sweeping Trigger Voltage")
    # trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    # print(trigger_voltage)
    # print("COMPLETED: Sweeping Trigger Voltage")

    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    print("STARTING: Algorithm S3.1. SDE Counts Measurement - Polarization Sweep")
    pol_counts_filepath = sweep_polarizations(now_str=now_str, IV_pickle_filepath='data/SK3_IV_curve_data_20241211-172258.pkl', name=name, num_pols=33, trigger_voltage=trigger_voltage, counting_time=0.5, N=1)
    print("COMPLETED: Algorithm S3.1. SDE Counts Measurement - Polarization Sweep")

    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    print("STARTING: Algorithm S3.2. SDE Counts Measuremen - True Counting")
    data_filepath = SDE_Counts_Measurement(now_str=now_str, IV_pickle_filepath='data/SK3_IV_curve_data_20241211-172258.pkl', pol_counts_filepath=pol_counts_filepath, name=name, trigger_voltage=trigger_voltage)
    print("COMPLETED: Algorithm S3.2. SDE Counts Measurement - True Counting")
    print("STARTING: Algorithm S2. Attenuator Calibration")
    attenuator_calibration_filepath = attenuator_calibration(now_str=now_str)
    print("COMPLETED: Algorithm S2. Attenuator Calibration")
# %%
