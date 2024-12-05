import os
import time
import pickle

import numpy as np
import pandas as pd

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.ando_aq8204 import AndoAQ8204
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.fiberControl_MPC101 import FiberControlMPC101

srs = SIM928('GPIB0::2::INSTR')
ando = AndoAQ8204('GPIB0::5::INSTR')
counter = Agilent53131a('GPIB0::3::INSTR')
pc = FiberControlMPC101('GPIB0::1::INSTR')

laser_ch = 1
att1_ch = 2
att2_ch = 3
att3_ch = 4
att_list = [att1_ch, att2_ch, att3_ch]
sw_ch = 5
pm_ch = 6

attval = 60 #dB
rngval = 10
ic = 10e-6 #A
vpol = 10

bias_resistor = 100e3 #Ohms

#%% Initialize and turn of laser
ando.aq820113_std_init(laser_ch)
ando.aq82011_enable(laser_ch)

#%% Algorithm S1. Nonlinearity factor raw power meaurements
N = 10
xlist = [20, 15]
xlist.append(np.arange(10, 0.9, -0.5))
xlist.append(np.arange(0.95, 0.5, -0.5))
base_array = round(10 - 10*np.log10(xlist))
base_array = base_array - np.min(base_array)
att_setting = {}

# rng is range setting
for rng in [-10, -20, -30, -40, -50, -60]:
    att_setting[rng] = base_array - (rng+10) -3


data = []

# Iterate through the ranges and settings
for rng in [-10, -20, -30, -40, -50, -60]:
    ando.aq820121_set_range(pm_ch, rng)
    ando.aq820121_zero(pm_ch)
    for a in att_setting[rng]:
        ando.aq820133_set_att(att1_ch, a)
        for att_step in [0, 3]:
            ando.aq820133_set_att(att2_ch, att_step)
            for i in range(N):
                power = ando.aq820121_get_power(pm_ch)
                # Append the data as a tuple
                data.append((rng, a, att_step, i, power))

# Convert the data to a pandas DataFrame
columns = ['Range', 'Attenuation Setting', 'Attenuation Step', 'Iteration', 'Power']
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame as a pickle file
pickle_file = 'power_data.pkl'
os.makedirs("data", exist_ok=True)
pickle_filepath = os.path.join("data", pickle_file)
df.to_pickle(pickle_filepath)

#%% Algorithm S2. Attenuator calibration
N = 5
init_rng = -10
for att_ch in att_list:
    ando.aq820133_set_att(att_ch, 0)

for att_ch in att_list:
    ando.aq820121_set_range(pm_ch, init_rng)
    for att_ch_ in att_list:
        ando.aq820133_disable(att_ch_)
    ando.aq820121_zero(pm_ch)
    for att_ch_ in att_list:
        ando.aq820133_enable(att_ch_)
    powers = {}
    for i in range(N):
        powers.append(ando.aq820121_get_power(pm_ch))
    write(att_list.get_att(), powers, init_rng)
    
    ando.aq820133_set_att(att_ch, attval)
    ando.aq820121_set_range(pm_ch, rngval)
    for att_ch_ in att_list:
        ando.aq820133_disable(att_ch_)
    ando.aq820121_zero(pm_ch)
    for att_ch_ in att_list:
        ando.aq820133_enable(att_ch_)
    powers = {}
    for i in range(N):
        powers.append(ando.aq820121_get_power(pm_ch))
    write(att_list.get_att(), powers, init_rng)
    ando.aq820133_set_att(att_ch, 0)


#%% Algorithm S3. SDE counts measurement
counting_time = 0.75
def get_counts(Cur_Array):
    srs.set_volt(0)
    srs.enable()

    Count_Array=np.zeros(len(Cur_Array))

    for i in np.arange(len(Cur_Array)):
        this_volt=round(Cur_Array[i]*1e-6*bias_resistor,3)
        
        srs.set_volt(this_volt)
        
        time.sleep(.1)
        Count_Array[i] = Agilent53131a.timed_count(counting_time=counting_time)
        print(f"{this_volt} - {Count_Array[i]}")
    
    srs.set_volt(0)
    srs.disable()
    
    return Count_Array

num_biases = 100
Cur_Array=np.round(np.linspace(0, ic*1.1, num_biases), 8)

N = 10
for att_ch in att_list:
    ando.aq820133_set_att(att_ch, attval)

ando.aq820143_set_route(sw_ch, 'monitor_port')
for att_ch in att_list:
    ando.aq820133_disable(att_ch)
Dark_Count_Array = get_counts(Cur_Array)



for att_ch in att_list:
    ando.aq820133_enable(att_ch)
ando.aq820143_set_route(sw_ch, 'detector_port')
srs.set_volt(vpol)
maxpol_settings = maximize_counts(pc, counter, out_maxpol)
minpol_settings = minimize_counts(pc, counter, out_minpol)
srs.set_volt(0)

pc.set(maxpol_settings)
Maxpol_Count_Array = get_counts(Cur_Array)

pc.set(minpol_settings)
Minpol_Count_Array = get_counts(Cur_Array)

data_dict = {
    'Cur_Array': list(Cur_Array),
    'Dark_Count_Array': list(Dark_Count_Array),
    'Maxpol_Count_Array': list(Maxpol_Count_Array),
    'Minpol_Count_Array': list(Minpol_Count_Array),
    }
data_filename = "data_dict.pkl"
os.makedirs("data", exist_ok=True)
data_filepath = os.path.join("data", data_filename)
with open(data_filepath, "wb") as file:
    pickle.dump(data_dict, file)

ando.aq820143_set_route(sw_ch, 'monitor_port')
CONSOLE_CAL(pm_ch, att_list, attval, rngval, out_att) # Algorithm S2