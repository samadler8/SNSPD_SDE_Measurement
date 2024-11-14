# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:40:11 2023

@author: qittlab
"""

import os
import sys
import math
import time
import numpy as np
import pyvisa as visa

amcc_path = os.path.join('Z:\\', 'Adler', 'python_packages', 'amcc-measurement')
base_parentpath = os.path.join('Z:\\', 'Adler', 'Mem_Dend_Data')
base_path = os.path.join('Z:\\', 'Adler', 'Mem_Dend_Data', 'CJJ')

# amcc_path = os.path.join('Q:\\', '68610', 'Adler', 'python_packages', 'amcc-measurement')
# base_parentpath = os.path.join('Q:\\', '68610', 'Adler', 'Mem_Dend_Data')
# base_path = os.path.join('Q:\\', '68610', 'Adler', 'Mem_Dend_Data', 'CJJ')

sys.path.append(amcc_path)
from amcc.instruments import lecroy_620zi, rigol_dg5000

sys.path.append(base_parentpath)
from Mem_Instr_Funcs import *
sys.path.append(base_path)
from cjj_funcs import *
from cjj_plot import *

# Connect to Instruments 
rm = visa.ResourceManager()
srs = rm.open_resource('GPIB0::2::INSTR')
lecroy = lecroy_620zi.LeCroy620Zi('TCPIP0::169.254.137.164::inst0::INSTR')
yoko_top = rm.open_resource('GPIB0::3::INSTR')
yoko_bot = rm.open_resource('GPIB0::1::INSTR')
siglent =  rm.open_resource('USB0::0xF4ED::0xEE3A::SDG10GAQ1R1166::INSTR')
rigol = rigol_dg5000.RigolDG5000('GPIB0::8::INSTR')
# multi = rm.open_resource('GPIB0::21::INSTR')
# instruments = np.array([srs, yoko_top, yoko_bot, siglent, rigol, lecroy, multi])
# instruments = np.array([srs, yoko_top, yoko_bot, siglent, rigol, lecroy])
# instruments = np.array([srs, yoko_top, yoko_bot, siglent, rigol, '', multi])

# Saving figures
device_name = 'CJJ_7_30_2024_10u_10u'
die_coordinate = '(2_3)'
base_save_path = os.path.join('Z:\\', 'Adler', 'Mem_Dend_Data', 'CJJ', device_name, die_coordinate)
test_data_folder = os.path.join(base_save_path, 'Test_Data')
final_data_folder = os.path.join(base_save_path, 'Final_Data')
os.makedirs(test_data_folder, exist_ok=True)
os.makedirs(final_data_folder, exist_ok=True)

# Resistors
R_SPD_A = 97.5e3 # 100 kilohms
R_SPD_B = 91.4e3 # 100 kilohms
R_AF_DEN = 100 # 100 ohms
R_AF_RO = 100 # 100 ohms

R_JJ_A = 10e3 # 100 kilohms
R_JJ_B = 10e3 # 100 kilohms

R_JJ = R_JJ_A

syn_a_tau = 10.00005e-6
syn_b_tau = 10.00005e-6

delay_max_right = 2*syn_a_tau
delay_max_left = 2*syn_b_tau

tau_max = max([syn_a_tau, syn_b_tau])

syn_weight_width = 3*tau_max




#%% Readout SQUID
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'Readout_SQUID_Resp'

is_final = 1

if is_final:
    folder = final_data_folder
    I_SQ_res = 5e-6
    avgs = 200
else:
    folder = test_data_folder
    I_SQ_res = 20e-6
    avgs = 50

# Channels
# Siglent instrument channel
ch_siglent_af_ro  = 'C2'

# LeCroy instrument channels
ch_lecroy_af_ro = 'C4'
ch_lecroy_data = 'C3'

# Define parameters for the measurement
I_SQ_min = 120e-6
I_SQ_max = 250e-6
I_SQs = np.arange(I_SQ_min, I_SQ_max, I_SQ_res)
I_AF_max = 12e-3  # A
ramp_freq = 1e3  # Hz
yoko_name = 'yoko_bot'

instruments = np.array([yoko_top, yoko_bot, siglent, lecroy])


params = {
    'time_str': time_str,

    'ch_siglent_af': ch_siglent_af_ro,
    'ch_lecroy_af': ch_lecroy_af_ro,
    'ch_lecroy_data': ch_lecroy_data,

    'I_SQs': I_SQs,
    'R_AF': R_AF_RO,

    'avgs': avgs,
    'ramp_freq': ramp_freq,
    'I_AF_max': I_AF_max,
    'yoko_name': yoko_name,
}

path = os.path.join(folder, time_str[:8], meas_name)
os.makedirs(path, exist_ok=True)
filepath_temp = get_sq_response(instruments, params, path=path)

title = 'Readout SQUID Response CJJ'
plot_sq_response(filepath_temp, title=title, std=False)
plot_sq_response_V_F(filepath_temp, title=title, std=False)

#%% Readout SQUID DC Measurement
# time_str = time.strftime("%Y%m%d-%H%M%S")
# meas_name = 'Readout_DC_SQUID_Resp'

# is_final = 0

# if is_final:
#     folder = final_data_folder
#     I_SQ_res = 10e-6
#     num_I_AF = 150
#     avgs = 200
# else:
#     folder = test_data_folder
#     I_SQ_res = 30e-6
#     num_I_AF = 50
#     avgs = 10

# # Define parameters for the measurement
# I_SQ_min = 120e-6
# I_SQ_max = 220e-6
# I_SQs = np.arange(I_SQ_min, I_SQ_max, I_SQ_res)
# I_AF_max = 12e-3
# I_AFs = np.linspace(-I_AF_max, I_AF_max, num_I_AF)
# yoko_name = 'yoko_bot'
# ch_srs_af_ro  = 'C2'

# # LeCroy instrument channels
# ch_lecroy_af_ro = 'C4'
# ch_lecroy_data = 'C3'

# params = {
#     'time_str': time_str,

#     'ch_srs_af': ch_srs_af_ro,
#     'yoko_name': yoko_name,

#     'I_SQs': I_SQs,
#     'I_AFs': I_AFs,
#     'R_AF': R_AF_RO,

#     'avgs': avgs,
# }

# path = os.path.join(folder, time_str[:8], meas_name)
# os.makedirs(path, exist_ok=True)
# filepath_temp = get_dc_sq_response(instruments, params, path=path)

# title = 'Readout DC SQUID Response'
# plot_dc_sq_response(filepath_temp, title=title)

# # turn_all_instruments_off(instruments, srs_chs=srs_chs)    
#%% Readout SQUID DC Measurement Using Lecroy
# time_str = time.strftime("%Y%m%d-%H%M%S")
# meas_name = 'Readout_DC_SQUID_Resp_Lecroy'

# is_final = 0

# if is_final:
#     folder = final_data_folder
#     I_SQ_res = 10e-6
#     num_I_AF = 150
#     avgs = 200
# else:
#     folder = test_data_folder
#     I_SQ_res = 30e-6
#     num_I_AF = 50
#     avgs = 10

# # Define parameters for the measurement
# I_SQ_min = 120e-6
# I_SQ_max = 220e-6
# I_SQs = np.arange(I_SQ_min, I_SQ_max, I_SQ_res)
# I_AF_max = 12e-3
# I_AFs = np.linspace(-I_AF_max, I_AF_max, num_I_AF)
# yoko_name = 'yoko_bot'
# ch_srs_af_ro  = '4'

# # LeCroy instrument channels
# ch_lecroy_data = 'C3'

# params = {
#     'time_str': time_str,

#     'ch_srs_af': ch_srs_af_ro,
#     'ch_lecroy_data': ch_lecroy_data,
#     'yoko_name': yoko_name,

#     'I_SQs': I_SQs,
#     'I_AFs': I_AFs,
#     'R_AF': R_AF_RO,
# }

# ch_siglent_global_trigger = 'C1'
# ch_siglent_laser = 'C2'
# ch_lecroy_laser = 'C4'

# period = 1e-6
# #Sweep Delays
# siglent_trigger_pulse(siglent, freq=1/period, h_level=5.0, channel=ch_siglent_global_trigger)
# siglent_laser_pulses(siglent, N=1, freq=10/period, h_level=2.0, delay=0, channel=ch_siglent_laser)
# time.sleep(0.1)
# turn_on_siglent(siglent)

# time_per_div = lecroy.round_up_lockstep(period/10)
# time_offset = 0
# lecroy.set_trigger(source=ch_lecroy_laser, volt_level=1, slope='positive')
# lecroy.set_horizontal_scale(time_per_div=time_per_div, time_offset=time_offset)
    

# path = os.path.join(folder, time_str[:8], meas_name)
# os.makedirs(path, exist_ok=True)
# filepath_temp = get_dc_sq_response_lecroy(instruments, params, path=path)

# title = 'Readout DC SQUID Response using LeCroy'
# plot_dc_sq_response(filepath_temp, title=title)

# # turn_all_instruments_off(instruments, srs_chs=srs_chs)    

#%%
I_AF_RO = 0.140/R_AF_RO  # A
I_RO = 0.190e-3  # A

#%% Reading Dendrite SQUID
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'Dendrite_SQUID_Resp'

is_final = 1

if is_final:
    folder = final_data_folder
    I_SQ_res = 5e-6
    avgs = 200
else:
    folder = test_data_folder
    I_SQ_res = 20e-6
    avgs = 10
    
ch_srs_af_ro  = '4'

ch_siglent_af_den  = 'C2'

ch_lecroy_af_den = 'C4'
ch_lecroy_data = 'C3'

V_AF_RO = I_AF_RO*R_AF_RO

SRS_init(srs, ch_srs_af_ro, V_AF_RO)
yoko_init(yoko_bot, I_RO)

# Define parameters for the measurement
I_SQ_min = 150e-6
I_SQ_max = 240e-6
I_SQs = np.arange(I_SQ_min, I_SQ_max, I_SQ_res)
I_AF_max = 6e-3  # Max current in amperes
ramp_freq = 1e3  # Frequency in Hz
yoko_name = 'yoko_top'

params = {
    'time_str': time_str,

    'ch_siglent_af': ch_siglent_af_den,
    'ch_lecroy_af': ch_lecroy_af_den,
    'ch_lecroy_data': ch_lecroy_data,

    'I_SQs': I_SQs,
    'R_AF': R_AF_DEN,

    'avgs': avgs,
    'ramp_freq': ramp_freq,
    'I_AF_max': I_AF_max,
    'yoko_name': yoko_name,
}

instruments = yoko_top, yoko_bot, siglent, lecroy

path = os.path.join(folder, time_str[:8], meas_name)
os.makedirs(path, exist_ok=True)
filepath_temp = get_sq_response(instruments, params, path=path)

title = 'Dendrite SQUID Response CJJ'
plot_sq_response(filepath_temp, title=title, std=False)
plot_sq_response_V_F(filepath_temp, title=title, std=False)

#%% Dendrite DC SQUID Measurement
# time_str = time.strftime("%Y%m%d-%H%M%S")
# meas_name = 'Dendrite_DC_SQUID_Resp'

# is_final = 1

# if is_final:
#     folder = final_data_folder
#     I_SQ_res = 10e-6
#     num_I_AF = 150
#     avgs = 200
# else:
#     folder = test_data_folder
#     I_SQ_res = 30e-6
#     num_I_AF = 50
#     avgs = 10

# # Define parameters for the measurement
# I_SQ_min = 130e-6
# I_SQ_max = 240e-6
# I_SQs = np.arange(I_SQ_min, I_SQ_max, I_SQ_res)
# I_AF_max = 36e-3
# I_AFs = np.linspace(-I_AF_max, I_AF_max, num_I_AF)
# yoko_name = 'yoko_top'
# ch_srs_af_ro  = '4'
# ch_srs_af_den  = '3'
# ch_lecroy_af_den = 'C4'
# ch_lecroy_data = 'C3'

# SRS_init(srs, ch_srs_af_ro, V_AF_RO)
# yoko_init(yoko_bot, I_RO)

# params = {
#     'time_str': time_str,

#     'ch_srs_af': ch_srs_af_den,
#     'yoko_name': yoko_name,

#     'I_SQs': I_SQs,
#     'I_AFs': I_AFs,
#     'R_AF': R_AF_RO,

#     'avgs': avgs,
# }

# path = os.path.join(folder, time_str[:8], meas_name)
# os.makedirs(path, exist_ok=True)
# filepath_temp = get_dc_sq_response(instruments, params, path=path)

# title = 'Dendrite DC SQUID Response'
# plot_dc_sq_response(filepath_temp, title=title)


#%%
# Channels
# srs channels
ch_srs_spd_a = '1'
ch_srs_spd_b = '2'
ch_srs_af_den = '3'
ch_srs_af_ro = '4'

# Siglent channels
ch_siglent_laser = 'C1'
ch_siglent_af_den = 'C2'

# LeCroy channels
ch_lecroy_data = 'C3'
ch_lecroy_af_den = 'C4'

# Fixed Biases
I_SPD_A = 13.5e-6 # A
I_SPD_B = 13.5e-6 # A

#%% Dendrite SQUID Response to Synaptic Weight
# time_str = time.strftime("%Y%m%d-%H%M%S")
# meas_name = 'dendrite_response_sweep_synaptic'
# is_final = 1
    
# if is_final:
#     folder = final_data_folder
#     I_DEN_res = 10e-6
#     num_I_JJs = 7
#     avgs = 200
# else:
#     folder = test_data_folder
#     I_DEN_res = 20e-6
#     num_I_JJs = 5
#     avgs = 5

# # parameters
# I_DEN_min = 130e-6
# I_DEN_max = 220e-6
# I_DENs = np.arange(I_DEN_min, I_DEN_max, I_DEN_res)
# I_JJ_min = 20e-6
# I_JJ_max = 50e-6
# I_JJs = np.linspace(I_JJ_min, I_JJ_max, num_I_JJs)
# I_AF_max = 6e-3
# ramp_freq = 1e3 #Hz

# # Initial guess paramaters
# guess_background = 75e-3 #V
# guess_amplitude_p2p = 60e-3 #V
# guess_period = 3.5e-3 #A
# guess_a = guess_amplitude_p2p/2
# guess_p = guess_period/(2*math.pi)
# guess_phi = 0
# guess_c = guess_background + guess_amplitude_p2p/2
# guess_m = (5e-3)/(4e-3)
# parameters_0 = [guess_a, guess_p, guess_phi, guess_c, guess_m]

# params = {
#     'time_str': time_str,

#     'ch_lecroy_data':  ch_lecroy_data,
#     'ch_lecroy_af_den':  ch_lecroy_af_den,
    
#     'ch_siglent_af_den':  ch_siglent_af_den,
#     'ch_siglent_laser':  ch_siglent_laser,
    
#     'ch_srs_spd_a':  ch_srs_spd_a,
#     'ch_srs_spd_b':  ch_srs_spd_b,
#     'ch_srs_jj_a':  ch_srs_jj_a,
#     'ch_srs_jj_b':  ch_srs_jj_b,

#     'R_AF_DEN': R_AF_DEN,
#     'R_JJ': R_JJ,
#     'ramp_freq': ramp_freq,
    
#     'V_SPD_A': V_SPD_A,
#     'V_SPD_B': V_SPD_B,
#     'V_AF_RO': V_AF_RO,
#     'I_RO': I_RO,

#     'avgs': avgs,
#     'I_JJs': I_JJs,
#     'I_DENs': I_DENs,
#     'I_AF_max': I_AF_max,
#     'syn_weight_states': ['syn_a_weighted', 'syn_b_weighted'],
    
#     'parameters_0': parameters_0,
#     }

# all_data = {
#     'params': params
#     }

# SRS_init(srs, ch_srs_af_ro, V_AF_RO)
# yoko_init(yoko_bot, I_RO)

# instruments = np.array([srs, yoko_top, yoko_bot, siglent, lecroy])

# path = os.path.join(folder, time_str[:8], meas_name)
# os.makedirs(path, exist_ok=True)

# sub_params = params.copy()
# sub_params['laser_on_off'] = False
# sub_params['synapses_on_off'] = False
# sub_path = os.path.join(path, 'synapses_off__laser_off')
# all_data['synapses_off__laser_off'] = run_den_sq_sweep_weight(instruments, sub_params, sub_path)

# sub_params = params.copy()
# sub_params['laser_on_off'] = False
# sub_params['synapses_on_off'] = True
# sub_path = os.path.join(path, 'synapses_on__laser_off')
# all_data['synapses_on__laser_off'] = run_den_sq_sweep_weight(instruments, sub_params, sub_path)

# # sub_params = params.copy()
# # sub_params['laser_on_off'] = True
# # sub_params['synapses_on_off'] = False
# # sub_path = os.path.join(path, 'synapses_off__laser_on')
# # all_data['synapses_off__laser_on'] = run_den_sq_sweep_weight(instruments, sub_params, sub_path)


# filepath = os.path.join(path, time_str)
# compressed_pickle(filepath, all_data)
# # plot_dc_den_sq_compare_effects(all_data['synapses_off__laser_off'], all_data['synapses_on__laser_off'], path=path, title='SNSPDs')
# # plot_dc_den_sq_compare_effects(all_data['synapses_off__laser_off'], all_data['synapses_off__laser_on'], path=path, title='SNSPDs')

# for srs_ch in srs_chs:
#     write_str = 'SNDT ' + str(srs_ch) + ',\"' + 'OPOF' + '\"'
#     srs.write(write_str)
# yoko_top.write(":OUTP OFF")
# yoko_bot.write(":OUTP OFF")
# siglent.write('C1:OUTP OFF')
# siglent.write('C2:OUTP OFF')


#%% Dendrite SQUID Response to Synaptic Weight
# time_str = time.strftime("%Y%m%d-%H%M%S")
# meas_name = 'dendrite_dc_response_sweep_synaptic'
# is_final = 0
    
# if is_final:
#     folder = final_data_folder
#     I_DEN_res = 10e-6
#     num_I_JJs = 7
#     num_I_AFs = 150
#     avgs = 200
# else:
#     folder = test_data_folder
#     I_DEN_res = 19e-6
#     num_I_JJs = 5
#     num_I_AFs = 30
#     avgs = 5

# # parameters
# I_DEN_min = 140e-6
# I_DEN_max = 200e-6
# I_DENs = np.arange(I_DEN_min, I_DEN_max, I_DEN_res)
# I_JJ_min = 20e-6
# I_JJ_max = 50e-6
# I_JJs = np.linspace(I_JJ_min, I_JJ_max, num_I_JJs)
# I_AF_max = 36e-3
# I_AFs = np.linspace(-I_AF_max, I_AF_max, num_I_AFs)


# # Initial guess paramaters
# guess_background = 75e-3 #V
# guess_amplitude_p2p = 30e-3 #V
# guess_period = 4e-3 #A
# guess_a = guess_amplitude_p2p/2
# guess_p = guess_period/(2*math.pi)
# guess_phi = 0
# guess_c = guess_background + guess_amplitude_p2p/2
# guess_m = (25e-3)/(20e-3)
# parameters_0 = [guess_a, guess_p, guess_phi, guess_c, guess_m]

# params = {
#     'time_str': time_str,

#     'ch_srs_af_den':  ch_srs_af_den,
#     'ch_srs_jj_a':  ch_srs_jj_a,
#     'ch_srs_jj_b':  ch_srs_jj_b,

#     'R_AF_DEN': R_AF_DEN,
#     'R_JJ': R_JJ,
    
#     'V_SPD_A': V_SPD_A,
#     'V_SPD_B': V_SPD_B,

#     'avgs': avgs,
#     'I_JJs': I_JJs,
#     'I_DENs': I_DENs,
#     'I_AFs': I_AFs,
#     'syn_weight_states': ['syn_a_weighted', 'syn_b_weighted'],
    
#     'parameters_0': parameters_0,
#     }

# all_data = {
#     'params': params
#     }

# path = os.path.join(folder, time_str[:8], meas_name)
# os.makedirs(path, exist_ok=True)

# sub_params = params.copy()
# sub_params['laser_on_off'] = False
# sub_params['synapses_on_off'] = False
# sub_path = os.path.join(path, 'synapses_off__laser_off')
# all_data['synapses_off__laser_off'] = run_den_dc_sq_sweep_weight(instruments, sub_params, sub_path)

# sub_params = params.copy()
# sub_params['laser_on_off'] = False
# sub_params['synapses_on_off'] = True
# sub_path = os.path.join(path, 'synapses_on__laser_off')
# all_data['synapses_on__laser_off'] = run_den_dc_sq_sweep_weight(instruments, sub_params, sub_path)

# sub_params = params.copy()
# sub_params['laser_on_off'] = True
# sub_params['synapses_on_off'] = False
# sub_path = os.path.join(path, 'synapses_off__laser_on')
# all_data['synapses_off__laser_on'] = run_den_dc_sq_sweep_weight(instruments, sub_params, sub_path)


# filepath = os.path.join(path, time_str)
# compressed_pickle(filepath, all_data)
# # plot_dc_den_sq_compare_effects(all_data['synapses_off__laser_off'], all_data['synapses_on__laser_off'], path=path, title='SNSPDs')
# # plot_dc_den_sq_compare_effects(all_data['synapses_off__laser_off'], all_data['synapses_off__laser_on'], path=path, title='SNSPDs')

# turn_all_instruments_off(instruments, srs_chs=srs_chs)

#%% Define JJ Critical Current/Voltage
JJ_Ic = 38.5e-6

#%% Find Minimum Delay
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'find_min_delay'
is_final = 1

if is_final:
    folder = final_data_folder
    avgs = 200
    delay_res = 50e-9
else:
    folder = test_data_folder
    avgs = 20
    delay_res = 500e-9

path = os.path.join(folder, time_str[0:8], meas_name)
os.makedirs(path, exist_ok=True)

ch_siglent_global_trigger = 'C1'
ch_siglent_laser = 'C2'

ch_lecroy_syn_a = 'C1'
ch_lecroy_syn_b = 'C2'
ch_lecroy_data = 'C3'
ch_lecroy_laser = 'C4'


edges = ['rising', 'falling']
synapses = ['a', 'b']
delays = np.arange(0, 3e-6, delay_res)
params = {
    'delays': delays,
}

siglent_trigger_pulse(siglent, freq=1/(2.5*2*tau_max), h_level=5.0, channel=ch_siglent_global_trigger)
siglent_laser_pulses(siglent, N=1, freq=2e6, h_level=2.0, delay=0)
time.sleep(0.1)
turn_on_siglent(siglent)

JJ_Vc = JJ_Ic*R_JJ

for edge in edges:
    if edge == 'rising':
        rigol_delay = 1e-6
        laser_delay_0 = 0
    elif edge == 'falling':
        rigol_delay = 0
        laser_delay_0 = 1e-6
    for synapse in synapses:
        if synapse == 'a':
            vhigh1 = JJ_Vc
            vhigh2 = 0
            ch_lecroy_syn = ch_lecroy_syn_a
        elif synapse == 'b':
            vhigh1 = 0
            vhigh2 = JJ_Vc
            ch_lecroy_syn = ch_lecroy_syn_b

        rigol_progamming(rigol, N_1=1, N_2=1, prog_freq=1/(2*10e-6), vhigh1=vhigh1, vhigh2=vhigh2, delay_1=rigol_delay, delay_2=rigol_delay, prog_width=3e-6)
        time.sleep(0.1)
        turn_on_rigol(rigol)

        max_delay = np.max(np.abs(delays))
        time_per_div = lecroy.round_up_lockstep(8e-6/10)
        time_offset = -1e-6
        lecroy.set_trigger(source=ch_lecroy_syn, volt_level=JJ_Vc/2, slope='positive')
        lecroy.set_horizontal_scale(time_per_div=time_per_div, time_offset=time_offset)

        data = {
            'params': params,
        }

        for i, delay in enumerate(delays):
            delay_data = {}
            print(f'{round(100*i/len(delays))}%')
            time.sleep(0.1)
            laser_delay = delay + laser_delay_0
            siglent.write('C2:BTWV DLAY,' + str(laser_delay))
            time.sleep(0.1)
            
            trace_data = get_traces(lecroy, avgs=avgs, channel=ch_lecroy_data, avg_std=True)
            laser_data = get_traces(lecroy, avgs=math.ceil(avgs/20), channel=ch_lecroy_laser, avg_std=True)
            delay_data['trace_data'] = trace_data
            delay_data['laser_data'] = laser_data
            delay_data['time_offset'] = time_offset
            delay_data['peak'] = np.max(trace_data[1])
            data[delay] = delay_data

        filename = f'{edge}_edge_synapse_{synapse}_traces_delay_sweep__{time_str}'
        filepath = os.path.join(path, filename)
        compressed_pickle(filepath, data)
        plot_sweep_delays(filepath, title='Traces at Different Delays')
        plot_sweep_delays_peaks(filepath)

#%% Find Minimum Delay double switch
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'find_min_delay_double_switch'
is_final = 1

if is_final:
    folder = final_data_folder
    avgs = 200
    delay_res = 50e-9
else:
    folder = test_data_folder
    avgs = 20
    delay_res = 500e-9

path = os.path.join(folder, time_str[0:8], meas_name)
os.makedirs(path, exist_ok=True)

ch_siglent_global_trigger = 'C1'
ch_siglent_laser = 'C2'

ch_lecroy_syn_a = 'C1'
ch_lecroy_syn_b = 'C2'
ch_lecroy_data = 'C3'
ch_lecroy_laser = 'C4'


first_synapses = ['a', 'b']
delays = np.arange(0, 6e-6, delay_res)
params = {
    'delays': delays,
}

siglent_trigger_pulse(siglent, freq=1/(2.5*2*tau_max), h_level=5.0, channel=ch_siglent_global_trigger)
siglent_laser_pulses(siglent, N=1, freq=2e6, h_level=2.0, delay=0)
time.sleep(0.1)
turn_on_siglent(siglent)

JJ_Vc = JJ_Ic*R_JJ

laser_delay_0 = 3e-6 
for first_synapse in first_synapses:
    if first_synapse == 'a':
        delay_1 = 0e-6
        delay_2 = 6e-6
        ch_lecroy_syn = ch_lecroy_syn_a
    elif first_synapse == 'b':
        delay_1 = 6e-6
        delay_2 = 0e-6
        ch_lecroy_syn = ch_lecroy_syn_b

    rigol_progamming(rigol, N_1=1, N_2=1, prog_freq=1/(2*10e-6), vhigh1=JJ_Vc, vhigh2=JJ_Vc, delay_1=delay_1, delay_2=delay_2, prog_width=6e-6)
    time.sleep(0.1)
    turn_on_rigol(rigol)

    max_delay = np.max(np.abs(delays))
    time_per_div = lecroy.round_up_lockstep(8e-6/10)
    time_offset = -6e-6
    lecroy.set_trigger(source=ch_lecroy_syn, volt_level=JJ_Vc/2, slope='positive')
    lecroy.set_horizontal_scale(time_per_div=time_per_div, time_offset=time_offset)

    data = {
        'params': params,
    }

    for i, delay in enumerate(delays):
        delay_data = {}
        print(f'{round(100*i/len(delays))}%')
        time.sleep(0.1)
        laser_delay = delay + laser_delay_0
        siglent.write('C2:BTWV DLAY,' + str(laser_delay))
        time.sleep(0.1)
        
        trace_data = get_traces(lecroy, avgs=avgs, channel=ch_lecroy_data, avg_std=True)
        laser_data = get_traces(lecroy, avgs=math.ceil(avgs/20), channel=ch_lecroy_laser, avg_std=True)
        delay_data['trace_data'] = trace_data
        delay_data['laser_data'] = laser_data
        delay_data['time_offset'] = time_offset
        delay_data['peak'] = np.max(trace_data[1])
        data[delay] = delay_data

    filename = f'first_synapse_{first_synapse}_traces_delay_sweep__{time_str}'
    filepath = os.path.join(path, filename)
    compressed_pickle(filepath, data)
    plot_sweep_delays(filepath, title='Traces at Different Delays')
    plot_sweep_delays_peaks(filepath)

#%% Define minimum delay
delay_min = 300e-9
#%%
# # Variable Biases for synaptic weight sweep and individual traces

# I_JJ_min = 30e-6
# I_JJ_fixed = 34e-6

# I_AF_DEN_130 = 0.0
# I_AF_DEN_140 = 0.0
# I_AF_DEN_150 = 0.0
# I_AF_DEN_160 = 0.0
# I_AF_DEN_170 = 0.0
# I_AF_DEN_180 = 730e-6
# I_AF_DEN_190 = 730e-6
# I_AF_DEN_200 = 730e-6
# I_AF_DEN_210 = 730e-6
# I_AF_DEN_220 = 730e-6
# I_AF_DEN_230 = 730e-6
# I_AF_DEN_240 = 730e-6
# I_AF_DEN_250 = 730e-6
# I_AF_DEN_260 = 730e-6
# I_AF_DEN_270 = 730e-6
# I_AF_DEN_280 = 730e-6
# I_AF_DEN_290 = 730e-6
# I_AF_DEN_300 = 730e-6
# I_AF_DEN_310 = 730e-6
# I_AF_DEN_320 = 730e-6
# I_AF_DEN_330 = 730e-6
# I_AF_DEN_340 = 730e-6
# I_AF_DEN_350 = 730e-6
# I_AF_DEN_360 = 730e-6
# I_AF_DEN_370 = 730e-6
# I_AF_DEN_380 = 730e-6
# I_AF_DEN_390 = 730e-6
# I_AF_DEN_400 = 730e-6
# I_AF_DEN_410 = 730e-6
# I_AF_DEN_420 = 730e-6
# I_AF_DEN_430 = 730e-6
# I_AF_DEN_440 = 730e-6
# I_AF_DEN_450 = 730e-6
# I_AF_DEN_460 = 730e-6
# I_AF_DEN_470 = 0.0
# I_AF_DEN_480 = 0.0


# I_DEN_dict = {
#     # 130e-6: {
#     #     'I_AF_DEN': I_AF_DEN_130,
#     #     },
#     # 140e-6: {
#     #     'I_AF_DEN': I_AF_DEN_140,
#     #     },
#     # 150e-6: {
#     #     'I_AF_DEN': I_AF_DEN_150,
#     #     },
#     # 160e-6: {
#     #     'I_AF_DEN': I_AF_DEN_160,
#     #     },
#     170e-6: {
#         'I_AF_DEN': I_AF_DEN_170,
#         },
#     180e-6: {
#         'I_AF_DEN': I_AF_DEN_180,
#         },
#     190e-6: {
#         'I_AF_DEN': I_AF_DEN_190,
#         },
#     # 200e-6: {
#     #     'I_AF_DEN': I_AF_DEN_200,
#     #     },
#     # 210e-6: {
#     #     'I_AF_DEN': I_AF_DEN_210,
#     #     },
#     # 220e-6: {
#     #     'I_AF_DEN': I_AF_DEN_220,
#     #     },
#     # 230e-6: {
#     #     'I_AF_DEN': I_AF_DEN_230,
#     #     },
#     # 240e-6: {
#     #     'I_AF_DEN': I_AF_DEN_240,
#     #     },
#     # 250e-6: {
#     #     'I_AF_DEN': I_AF_DEN_250,
#     #     },
#     # 260e-6: {
#     #     'I_AF_DEN': I_AF_DEN_260,
#     #     },
#     # 270e-6: {
#     #     'I_AF_DEN': I_AF_DEN_270,
#     #     },
#     # 280e-6: {
#     #     'I_AF_DEN': I_AF_DEN_280,
#     #     },
#     # 290e-6: {
#     #     'I_AF_DEN': I_AF_DEN_190,
#     #     },
#     # 300e-6: {
#     #     'I_AF_DEN': I_AF_DEN_200,
#     #     },
#     # 310e-6: {
#     #     'I_AF_DEN': I_AF_DEN_210,
#     #     },
#     # 320e-6: {
#     #     'I_AF_DEN': I_AF_DEN_220,
#     #     },
#     # 330e-6: {
#     #     'I_AF_DEN': I_AF_DEN_230,
#     #     },
#     # 340e-6: {
#     #     'I_AF_DEN': I_AF_DEN_240,
#     #     },
#     # 350e-6: {
#     #     'I_AF_DEN': I_AF_DEN_250,
#     #     },
#     # 360e-6: {
#     #     'I_AF_DEN': I_AF_DEN_260,
#     #     },
#     # 370e-6: {
#     #     'I_AF_DEN': I_AF_DEN_270,
#     #     },
#     # 380e-6: {
#     #     'I_AF_DEN': I_AF_DEN_380,
#     #     },
#     # 390e-6: {
#     #     'I_AF_DEN': I_AF_DEN_390,
#     #     },
#     # 400e-6: {
#     #     'I_AF_DEN': I_AF_DEN_200,
#     #     },
#     # 410e-6: {
#     #     'I_AF_DEN': I_AF_DEN_210,
#     #     },
#     # 420e-6: {
#     #     'I_AF_DEN': I_AF_DEN_220,
#     #     },
#     # 430e-6: {
#     #     'I_AF_DEN': I_AF_DEN_230,
#     #     },
#     # 440e-6: {
#     #     'I_AF_DEN': I_AF_DEN_240,
#     #     },
#     # 450e-6: {
#     #     'I_AF_DEN': I_AF_DEN_250,
#     #     },
#     # 460e-6: {
#     #     'I_AF_DEN': I_AF_DEN_260,
#     #     },
#     # 470e-6: {
#     #     'I_AF_DEN': I_AF_DEN_270,
#     #     },
#     # 480e-6: {
#     #     'I_AF_DEN': I_AF_DEN_280,
#     #     },
#     }

#%%
# Channels
# srs channels
ch_srs_spd_a = '1'
ch_srs_spd_b = '2'
ch_srs_af_den = '3'
ch_srs_af_ro = '4'

# Siglent channels
ch_siglent_global_trigger = 'C1'
ch_siglent_laser = 'C2'

# LeCroy channels
ch_lecroy_syn_a = 'C1'
ch_lecroy_syn_b = 'C2'
ch_lecroy_data = 'C3'
ch_lecroy_laser = 'C4'

I_AF_RO = -0.048/R_AF_RO  # A
I_RO = 0.170e-3  # A
I_SPD_A = 1.350/R_SPD_A
I_SPD_B = 1.350/R_SPD_B
delay_min = 333.33333e-9
#%%Sequence Response sweeping JJ bias at a specific I_DEN bias
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'synaptic_weights_sweep'
is_final = 0

instruments = np.array([srs, yoko_top, yoko_bot, siglent, rigol, lecroy])


if is_final:
    folder = final_data_folder
    avgs = 200
    num_I_JJs = 17
    delay_res = 333.3333e-9
else:
    folder = test_data_folder
    avgs = 10
    num_I_JJs = 6
    delay_res = 1000e-9

# parameters
I_JJ_min = 30e-6
I_JJ_max = 41e-6
I_JJs = np.linspace(I_JJ_min, I_JJ_max, num_I_JJs)
delays = np.concatenate([np.flip(-1*np.arange(delay_min, delay_max_left, delay_res)), [0], np.arange(delay_min, delay_max_right, delay_res)])

srs_chs = [ch_srs_spd_a, ch_srs_spd_b, ch_srs_af_den, ch_srs_af_ro]

I_DEN = 190e-6
I_AF_DEN = -0.095/R_AF_DEN

srs_Vs = [I_SPD_A*R_SPD_A, I_SPD_B*R_SPD_B, I_AF_DEN*R_AF_DEN, I_AF_RO*R_AF_RO]

SRS_init(srs, srs_chs, srs_Vs)
yoko_init(yoko_bot, I_RO)
yoko_init(yoko_top, I_DEN)

params = {
    'time_str': time_str,

    'ch_siglent_global_trigger': ch_siglent_global_trigger,
    'ch_siglent_laser': ch_siglent_laser,
    'ch_lecroy_syn_a': ch_lecroy_syn_a,
    'ch_lecroy_syn_b': ch_lecroy_syn_b,
    'ch_lecroy_data': ch_lecroy_data,
    'ch_lecroy_laser': ch_lecroy_laser,

    'I_SPD_A': I_SPD_A,
    'I_SPD_B': I_SPD_B,
    'I_AF_DEN': I_AF_DEN,
    'I_AF_RO': I_AF_RO,
    'I_DEN': I_DEN,
    'I_RO': I_RO,
    'R_JJ_1': R_JJ_A,
    'R_JJ_2': R_JJ_B,
    'R_JJ': R_JJ,

    'I_JJs': I_JJs,
    'delays': delays,
    'avgs': avgs,
    'syn_weight_width': syn_weight_width,
    'delay_min': delay_min,
}

path = os.path.join(folder, time_str[0:8], meas_name, f'I_den_{1e6 * I_DEN:.3g}uA', f'I_AF_{1e3*I_AF_DEN:.{3}g}mA')
os.makedirs(path, exist_ok=True)
filepath_temp = sweep_JJbias_delays(instruments, params, path=path)
plot_JJbias_delays(filepath_temp)

instruments =  [srs, yoko_top, yoko_bot, siglent, rigol, _, _]
turn_all_instruments_off(instruments, srs_chs=srs_chs)    

#%%Individual Sequence Responses - fixed JJ_height and I_DEN
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'individual_traces'

is_final = 0
if is_final:
    folder = final_data_folder
    avgs = 1000
    delay_res = 333.3333e-9
else:
    folder = test_data_folder
    avgs = 20
    delay_res = 1000e-9

# parameters
delays = np.concatenate([np.flip(-1*np.arange(delay_min, delay_max_left, delay_res)), [0], np.arange(delay_min, delay_max_right, delay_res)])

I_JJ_fixed = 39e-6
I_DEN = 190e-6
I_AF_DEN = -0.095/R_AF_DEN

srs_Vs = [I_SPD_A*R_SPD_A, I_SPD_B*R_SPD_B, I_AF_DEN*R_AF_DEN, I_AF_RO*R_AF_RO]

SRS_init(srs, srs_chs, srs_Vs)
yoko_init(yoko_bot, I_RO)
yoko_init(yoko_top, I_DEN)

params = {
    'time_str': time_str,

    'ch_siglent_global_trigger': ch_siglent_global_trigger,
    'ch_lecroy_syn_a': ch_lecroy_syn_a,
    'ch_lecroy_syn_b': ch_lecroy_syn_b,
    'ch_lecroy_data': ch_lecroy_data,
    'ch_lecroy_laser': ch_lecroy_laser,
    
    'R_JJ': R_JJ,
    'R_JJ_1': R_JJ_A,
    'R_JJ_2': R_JJ_B,
    
    'I_SPD_A': I_SPD_A,
    'I_SPD_B': I_SPD_B,
    'I_AF_DEN': I_AF_DEN,
    'I_AF_RO': I_AF_RO,
    'I_DEN': I_DEN,
    'I_RO': I_RO,

    'avgs': avgs,
    'delays': delays,
    'delay_min': delay_min,
    'syn_weight_width': syn_weight_width,
    'I_JJ_fixed': I_JJ_fixed,
    'I_JJ_1': I_JJ_fixed,
    'I_JJ_2': I_JJ_fixed,
}

instruments = np.array([srs, yoko_top, yoko_bot, siglent, rigol, lecroy])

path = os.path.join(folder, time_str[0:8], meas_name, f'I_den_{1e6 * I_DEN:.3g}uA', f'I_AF_{1e3 * I_AF_DEN:.{3}g}mA', f'I_JJ_fixed_{1e6 * I_JJ_fixed:.{3}g}uA')
os.makedirs(path, exist_ok=True) 
filepath_temp = sweep_delays(instruments, params, path=path, remove_background=False)
plot_individual_coincidence_traces(filepath_temp)


instruments =  [srs, yoko_top, yoko_bot, siglent, rigol, _, _]
turn_all_instruments_off(instruments, srs_chs=srs_chs)    

#%%Sequence Response sweeping I_DEN at a specific JJ bias
# Also hold AF_DEN constant otherwise it doesn't make sense
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'dendrite_bias_sweep'
is_final = 0

if is_final:
    folder = final_data_folder
    avgs = 200
    num_I_DENs = 13
    delay_res = 333.3333333e-9
else:
    folder = test_data_folder
    avgs = 10
    num_I_DENs = 6
    delay_res = 1000e-9

I_JJ_fixed = 39e-6
I_AF_DEN = -0.095/R_AF_DEN

srs_Vs = [I_SPD_A*R_SPD_A, I_SPD_B*R_SPD_B, I_AF_DEN*R_AF_DEN, I_AF_RO*R_AF_RO]

SRS_init(srs, srs_chs, srs_Vs)
yoko_init(yoko_bot, I_RO)
yoko_init(yoko_top, I_DEN)

# parameters
I_DEN_min = 150e-6
I_DEN_max = 230e-6
I_DENs = np.linspace(I_DEN_min, I_DEN_max, num_I_DENs)
delays = np.concatenate([np.flip(-1*np.arange(delay_min, delay_max_left, delay_res)), [0], np.arange(delay_min, delay_max_right, delay_res)])

params = {
    'time_str': time_str,

    'ch_siglent_global_trigger': ch_siglent_global_trigger,
    'ch_lecroy_syn_a': ch_lecroy_syn_a,
    'ch_lecroy_syn_b': ch_lecroy_syn_b,
    'ch_lecroy_data': ch_lecroy_data,
    'ch_lecroy_laser': ch_lecroy_laser,

    'I_SPD_A': I_SPD_A,
    'I_SPD_B': I_SPD_B,
    'I_AF_DEN': I_AF_DEN,
    'I_AF_RO': I_AF_RO,
    
    'I_RO': I_RO,
    'I_DENs': I_DENs,
    'R_JJ': R_JJ,
    'R_JJ_1': R_JJ_A,
    'R_JJ_2': R_JJ_B,
    'I_JJ_fixed': I_JJ_fixed,

    'avgs': avgs,
    'I_DENs': I_DENs,
    'delays': delays,
    'delay_min': delay_min,
    'syn_weight_width': syn_weight_width,
}

instruments = np.array([srs, yoko_top, yoko_bot, siglent, rigol, lecroy])

path = os.path.join(folder, time_str[0:8], meas_name, f'I_AF_{1e3 * I_AF_DEN:.{3}g}mA', f'I_JJ_{1e6 * I_JJ_fixed:.{3}g}uA', )
os.makedirs(path, exist_ok=True)
filepath_temp = sweep_Iden_delays(instruments, params, path=path)
plot_sweep_Iden_delays(filepath_temp)

instruments =  [srs, yoko_top, yoko_bot, siglent, rigol, _, _]
turn_all_instruments_off(instruments, srs_chs=srs_chs)    
#%%Sequence Response sweeping dendrite add flux at a specific JJ bias
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'dendrite_af_sweep'
is_final = 0

if is_final:
    folder = final_data_folder
    avgs = 200
    num_AFs = 13
    delay_res = 333.3333333e-9
else:
    folder = test_data_folder
    avgs = 20
    num_AFs = 5
    delay_res = 3000e-9
    
# Channels
# srs channels
ch_srs_spd_a = '5'
ch_srs_spd_b = '2'
ch_srs_af_den = '3'
ch_srs_af_ro = '4'
srs_chs = [ch_srs_spd_a, ch_srs_spd_b, ch_srs_af_den, ch_srs_af_ro]

# Rigol channels
ch_rigol_syn_a = 1
ch_rigol_syn_b = 2

# Siglent channels
ch_siglent_global_trigger = 'C1'
ch_siglent_laser = 'C2'

# LeCroy channels
ch_lecroy_syn_a = 'C1'
ch_lecroy_syn_b = 'C2'
ch_lecroy_data = 'C3'
ch_lecroy_laser = 'C4'

I_RO = 170*1e-6 # A
I_DEN = 190e-6

# parameters
delays = np.concatenate([np.flip(-1*np.arange(delay_min, delay_max_left, delay_res)), [0], np.arange(delay_min, delay_max_right, delay_res)])
I_AF_min = -0.150/R_AF_DEN
I_AF_max = -0.000/R_AF_DEN
I_AF_DENs = np.linspace(I_AF_min, I_AF_max, num_AFs)

srs_Vs = [I_SPD_A*R_SPD_A, I_SPD_B*R_SPD_B, I_AF_DEN*R_AF_DEN, I_AF_RO*R_AF_RO]

SRS_init(srs, srs_chs, srs_Vs)
yoko_init(yoko_bot, I_RO)
yoko_init(yoko_top, I_DEN)

params = {
    'time_str': time_str,

    'ch_siglent_global_trigger': ch_siglent_global_trigger,
    'ch_srs_af_den': ch_srs_af_den,
    'ch_lecroy_syn_a': ch_lecroy_syn_a,
    'ch_lecroy_syn_b': ch_lecroy_syn_b,
    'ch_lecroy_data': ch_lecroy_data,
    'ch_lecroy_laser': ch_lecroy_laser,

    'I_SPD_A': I_SPD_A,
    'I_SPD_B': I_SPD_B,
    'I_AF_RO': I_AF_RO,
    'I_RO': I_RO,
    'I_DEN': I_DEN,
    'R_AF_DEN': R_AF_DEN,
    'R_JJ': R_JJ,
    'R_JJ_1': R_JJ_A,
    'R_JJ_2': R_JJ_B,
    'I_JJ_fixed': I_JJ_fixed,

    'avgs': avgs,
    'I_AF_DENs': I_AF_DENs,
    'delays': delays,
    'delay_min': delay_min,
    'syn_weight_width': syn_weight_width,
}

instruments = np.array([srs, yoko_top, yoko_bot, siglent, rigol, lecroy])

path = os.path.join(folder, time_str[0:8], meas_name, f'I_den_{1e6 * I_DEN:.3g}uA', f'I_JJ_{1e6 * I_JJ_fixed:.{3}g}uA')
os.makedirs(path, exist_ok=True)
filepath_temp = sweep_den_AF_delays(instruments, params, path=path)
plot_sweep_den_AF_delays(filepath_temp)

instruments =  [srs, yoko_top, yoko_bot, siglent, rigol, _, _]
turn_all_instruments_off(instruments, srs_chs=srs_chs)    
#%%
# # Variable Biases for asymmetric synaptic weight sweeps

# V_JJ_min = 0.26

# V_AF_DEN_130 = 0.0
# V_AF_DEN_140 = 0.0
# V_AF_DEN_150 = 0.0
# V_AF_DEN_160 = 0.0
# V_AF_DEN_170 = 0.0
# V_AF_DEN_180 = 0.073
# V_AF_DEN_190 = 0.073
# V_AF_DEN_200 = 0.073
# V_AF_DEN_210 = 0.073
# V_AF_DEN_220 = 0.066
# V_AF_DEN_230 = 0.062
# V_AF_DEN_240 = 0.051
# V_AF_DEN_250 = 0.041
# V_AF_DEN_260 = 0.031
# V_AF_DEN_270 = 0.0
# V_AF_DEN_280 = 0.0
# V_AF_DEN_290 = 0.073
# V_AF_DEN_300 = 0.073
# V_AF_DEN_310 = 0.073
# V_AF_DEN_320 = 0.066
# V_AF_DEN_330 = 0.062
# V_AF_DEN_340 = 0.051
# V_AF_DEN_350 = 0.041
# V_AF_DEN_360 = 0.031
# V_AF_DEN_370 = 0.0
# V_AF_DEN_380 = -0.043
# V_AF_DEN_390 = -0.054
# V_AF_DEN_400 = 0.073
# V_AF_DEN_410 = 0.073
# V_AF_DEN_420 = 0.066
# V_AF_DEN_430 = 0.062
# V_AF_DEN_440 = 0.051
# V_AF_DEN_450 = 0.041
# V_AF_DEN_460 = 0.031
# V_AF_DEN_470 = 0.0
# V_AF_DEN_480 = 0.0


# I_DEN_dict = {
#     # 130e-6: {
#     #     'V_AF_DEN': V_AF_DEN_130,
#     #     },
#     # 140e-6: {
#     #     'V_AF_DEN': V_AF_DEN_140,
#     #     },
#     # 150e-6: {
#     #     'V_AF_DEN': V_AF_DEN_150,
#     #     },
#     # 160e-6: {
#     #     'V_AF_DEN': V_AF_DEN_160,
#     #     },
#     # 170e-6: {
#     #     'V_AF_DEN': V_AF_DEN_170,
#     #     },
#     # 180e-6: {
#     #     'V_AF_DEN': V_AF_DEN_180,
#     #     },
#     # 190e-6: {
#     #     'V_AF_DEN': V_AF_DEN_190,
#     #     },
#     # 200e-6: {
#     #     'V_AF_DEN': V_AF_DEN_200,
#     #     },
#     # 210e-6: {
#     #     'V_AF_DEN': V_AF_DEN_210,
#     #     },
#     # 220e-6: {
#     #     'V_AF_DEN': V_AF_DEN_220,
#     #     },
#     # 230e-6: {
#     #     'V_AF_DEN': V_AF_DEN_230,
#     #     },
#     # 240e-6: {
#     #     'V_AF_DEN': V_AF_DEN_240,
#     #     },
#     # 250e-6: {
#     #     'V_AF_DEN': V_AF_DEN_250,
#     #     },
#     # 260e-6: {
#     #     'V_AF_DEN': V_AF_DEN_260,
#     #     },
#     # 270e-6: {
#     #     'V_AF_DEN': V_AF_DEN_270,
#     #     },
#     # 280e-6: {
#     #     'V_AF_DEN': V_AF_DEN_280,
#     #     },
#     # 290e-6: {
#     #     'V_AF_DEN': V_AF_DEN_190,
#     #     },
#     # 300e-6: {
#     #     'V_AF_DEN': V_AF_DEN_200,
#     #     },
#     # 310e-6: {
#     #     'V_AF_DEN': V_AF_DEN_210,
#     #     },
#     # 320e-6: {
#     #     'V_AF_DEN': V_AF_DEN_220,
#     #     },
#     # 330e-6: {
#     #     'V_AF_DEN': V_AF_DEN_230,
#     #     },
#     # 340e-6: {
#     #     'V_AF_DEN': V_AF_DEN_240,
#     #     },
#     # 350e-6: {
#     #     'V_AF_DEN': V_AF_DEN_250,
#     #     },
#     # 360e-6: {
#     #     'V_AF_DEN': V_AF_DEN_260,
#     #     },
#     # 370e-6: {
#     #     'V_AF_DEN': V_AF_DEN_270,
#     #     },
#     380e-6: {
#         'V_AF_DEN': V_AF_DEN_380,
#         },
#     390e-6: {
#         'V_AF_DEN': V_AF_DEN_390,
#         },
#     # 400e-6: {
#     #     'V_AF_DEN': V_AF_DEN_200,
#     #     },
#     # 410e-6: {
#     #     'V_AF_DEN': V_AF_DEN_210,
#     #     },
#     # 420e-6: {
#     #     'V_AF_DEN': V_AF_DEN_220,
#     #     },
#     # 430e-6: {
#     #     'V_AF_DEN': V_AF_DEN_230,
#     #     },
#     # 440e-6: {
#     #     'V_AF_DEN': V_AF_DEN_240,
#     #     },
#     # 450e-6: {
#     #     'V_AF_DEN': V_AF_DEN_250,
#     #     },
#     # 460e-6: {
#     #     'V_AF_DEN': V_AF_DEN_260,
#     #     },
#     # 470e-6: {
#     #     'V_AF_DEN': V_AF_DEN_270,
#     #     },
#     # 480e-6: {
#     #     'V_AF_DEN': V_AF_DEN_280,
#     #     },
#     }
#%%Sequence Response sweeping only one JJ bias at a specific I_DEN bias
time_str = time.strftime("%Y%m%d-%H%M%S")
meas_name = 'synaptic_weight_sweep_only_one'
is_final = 1

if is_final:
    folder = final_data_folder
    avgs = 200
    num_V_JJs = 19
    delay_res = 500e-9
else:
    folder = test_data_folder
    avgs = 20
    num_V_JJs = 5
    delay_res = 3000e-9

# parameters
V_JJs = np.linspace(V_JJ_min, JJ_Vc, num_V_JJs)
delays = np.concatenate([np.flip(-1*np.arange(delay_min, delay_max_left, delay_res)), [0], np.arange(delay_min, delay_max_right, delay_res)])

for i, (key, value) in enumerate(I_DEN_dict.items()):
    print(f'Dendrite bias {i+1} of {len(I_DEN_dict.items())}')
    I_DEN = key
    V_AF_DEN = value['V_AF_DEN']
    
    srs_Vs = [V_SPD_A, V_SPD_B, V_AF_DEN, V_AF_RO]

    SRS_init(srs, srs_chs, srs_Vs)
    yoko_init(yoko_bot, I_RO)
    yoko_init(yoko_top, I_DEN)  
    
    params = {
        'time_str': time_str,

        'ch_siglent_global_trigger': ch_siglent_global_trigger,
        'ch_lecroy_syn_a': ch_lecroy_syn_a,
        'ch_lecroy_syn_b': ch_lecroy_syn_b,
        'ch_lecroy_data': ch_lecroy_data,
        'ch_lecroy_laser': ch_lecroy_laser,
        
        'R_JJ': R_JJ,
        'R_JJ_1': R_JJ_A,
        'R_JJ_2': R_JJ_B,
        
        'V_SPD_A': V_SPD_A,
        'V_SPD_B': V_SPD_B,
        'V_AF_DEN': V_AF_DEN,
        'V_AF_RO': V_AF_RO,
        'I_DEN': I_DEN,
        'I_RO': I_RO,

        'avgs': avgs,
        'delays': delays,
        'V_JJ_fixed': V_JJ_fixed,
        'V_JJs': V_JJs,
        'delay_min': delay_min,
        'syn_weight_width': syn_weight_width,
        'V_JJ_fixed': V_JJ_fixed,
        'V_JJ_1': V_JJ_fixed,
        'V_JJ_2': V_JJ_fixed,
    }
    
    path = os.path.join(folder, time_str[0:8], meas_name, f'I_den_{1e6 * I_DEN:.3g}uA')
    os.makedirs(path, exist_ok=True)
    filepaths_temp = sweep_one_JJbias_delays(instruments, params, path=path)
    plot_sweep_one_JJbias_delays(filepaths_temp)
    
turn_all_instruments_off(instruments, srs_chs=srs_chs)    

#%%Inhibition Response
meas_name = f'inhibition_traces'
time_str = time.strftime("%Y%m%d-%H%M%S")
is_final = 0

if is_final:
    folder = final_data_folder
    avgs = 1000
else:
    folder = test_data_folder
    avgs = 50
    
V_AF_DEN = .085 # V
srs_Vs = [V_SPD_A, V_SPD_B, V_AF_DEN, V_AF_RO]

I_DEN = 240e-6

V_JJ_min = 0.26

SRS_init(srs, srs_chs, srs_Vs)
yoko_init(yoko_top, I_DEN)
yoko_init(yoko_bot, I_RO)

# parameters. Initialize for test
delay_pulses = 5
initial_pulses = 5
inhibition_pulses = 5
inhibition_tolerance = 'neutral'
inhibitor = 'A'
laser_period = 1000e-9
syn_tau = 10e-6

params = {
    'time_str': time_str, 

    'ch_siglent_global_trigger': ch_siglent_global_trigger,
    'ch_siglent_laser': ch_siglent_laser,
    'ch_lecroy_syn_a': ch_lecroy_syn_a,
    'ch_lecroy_syn_b': ch_lecroy_syn_b,
    'ch_lecroy_data': ch_lecroy_data,
    'ch_lecroy_laser': ch_lecroy_laser,
    'ch_srs_spd_a': ch_srs_spd_a,
    'ch_srs_spd_b': ch_srs_spd_b,

    'R_JJ_1': R_JJ_A,
    'R_JJ_2': R_JJ_B,
    'syn_tau': syn_tau,

    'avgs': avgs,
    'V_SPD_A': V_SPD_A,
    'V_SPD_B': V_SPD_B,
    'V_AF_DEN': V_AF_DEN,
    'V_AF_RO': V_AF_RO,
    'I_DEN': I_DEN,
    'I_RO': I_RO,
    'V_JJ_A': V_JJ_fixed,
    'V_JJ_B': V_JJ_fixed,
    'laser_period': laser_period,
    'inhibitor': inhibitor,
    'inhibition_tolerance': inhibition_tolerance,
    'inhibition_pulses': inhibition_pulses,
    'initial_pulses': initial_pulses,
    'delay_pulses': delay_pulses,
}

# Test
parent_path = os.path.join(folder, time_str[0:8], meas_name, f'I_den_{1e6 * I_DEN:.3g}uA', f'I_AF_{1e3 * V_AF_DEN/R_AF_DEN:.{3}g}mA')
os.makedirs(parent_path, exist_ok=True) 
filepath_tamp = inhibition_trace(instruments, params, path=parent_path)
plot_inhibition_trace(filepath_tamp)

# Check inhibition is not electrical
path = os.path.join(parent_path, 'not_electrical')
os.makedirs(path, exist_ok=True) 
params['laser_period'] = 3000e-9
inhibition_not_electrical(instruments, params, path=path)

# Sweep inhibition pulses
path = os.path.join(parent_path, 'sweep_inhibitory_pulses')
os.makedirs(path, exist_ok=True) 
params['laser_period'] = 3000e-9
params['max_inhibition_pulses'] = 5
inhibitory_pulses_sweep(instruments, params, path=path)

# Sweep inhibition bias
path = os.path.join(parent_path, 'sweep_inhibitory_weight')
os.makedirs(path, exist_ok=True) 
params['laser_period'] = 3000e-9
params['inhibition_pulses'] = 1
params['inhibition_tolerance'] = 'neutral'
params['V_JJ_min'] = V_JJ_min
params['V_JJ_max'] = JJ_Vc
params['V_JJ_fixed'] = V_JJ_fixed
params['num_V_JJs'] = 5
inhibitory_weight_sweep(instruments, params, path=path)

turn_all_instruments_off(instruments, srs_chs=srs_chs)    