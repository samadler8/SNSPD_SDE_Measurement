import os
import time

import numpy as np

srs = srs_('GPIB0::2::INSTR')
ando = AndoAQ8204('GPIB0::5::INSTR')
counter = Agilent53131a('GPIB0::5::INSTR')

att1_ch = 
att2_ch = 
att3_ch = 
att_list = [att1_ch, att2_ch, att3_ch]
pm_ch = 
sw_ch = 
pc_ch = 

attval = 
rngval = 
ic = 
bias_step = 
vpol = 

bias_resistor = 

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

for rng in [-10, -20, -30, -40, -50, -60]:
    ando.aq820121_set_range(pm_ch, rng)
    ando.aq820121_zero(pm_ch)
    for a in att_setting[rng]:
        ando.aq820133_set_att(att1_ch, a)
        for att_step in [0, 3]:
            ando.aq820133_set_att(att2_ch, att_step)
            for i in range(1, N):
                power = ando.aq820121_get_power(pm_ch)
                write(a, att_step, rng, power)


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
    for i in range(1, N):
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
    for i in range(1, N):
        powers.append(ando.aq820121_get_power(pm_ch))
    write(att_list.get_att(), powers, init_rng)
    ando.aq820133_set_att(att_ch, 0)


#%% Algorithm S3. SDE counts measurement
def get_counts(Cur_Array):
    srs.enable()
    srs.set_volt(0)

    counting_time = 0.3
    Count_Array=np.zeros(len(Cur_Array))

    for i in np.arange(len(Cur_Array)):
        this_volt=round(Cur_Array[i]*1e-6*bias_resistor,3)
        
        srs.set_volt(this_volt)
        
        time.sleep(.1)
        Count_Array[i] = Agilent53131a.timed_count(counting_time=counting_time)
        print(str(this_volt)+' - '+str(Count_Array[i]))
    
    srs.set_volt(0)
    srs.disable()

    
    return Count_Array

Cur_Array=np.round(np.linspace(0, ic, bias_step), 8)

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
maxpol_settings = maximize_counts(pc_ch, counter, out_maxpol)
minpol_settings = minimize_counts(pc_ch, counter, out_minpol)
srs.set_volt(0)

pc.set(maxpol_settings)
Maxpol_Count_Array = get_counts(Cur_Array)

pc.set(minpol_settings)
Minpol_Count_Array = get_counts(Cur_Array)

ando.aq820143_set_route(sw_ch, 'monitor_port')
CONSOLE_CAL(pm, att_list, attval, rngval, out_att) # Slgorithm S2