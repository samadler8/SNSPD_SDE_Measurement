# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:36:41 2023

@author: qittlab
"""

import numpy as np
from matplotlib import pyplot as plt
import bz2
import _pickle as cPickle
import time

#%% Data Pickling
# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
 
# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data
#%% 
def turn_all_instruments_off(instruments, srs_chs=['1', '2', '3', '4', '5', '6']):
    SRS, yoko_top, yoko_bot, siglent, rigol, _, _ = instruments
    for srs_ch in srs_chs:
        write_str = 'SNDT ' + str(srs_ch) + ',\"' + 'OPOF' + '\"'
        SRS.write(write_str)
    yoko_top.write(":OUTP OFF")
    yoko_bot.write(":OUTP OFF")
    siglent.write('C1:OUTP OFF')
    siglent.write('C2:OUTP OFF')
    rigol.set_output(False, channel = 1)
    rigol.set_output(False, channel = 2)
    return

def initialize_all_instruments(instruments, biases):
    SRS, yoko_top, yoko_bot, siglent, rigol, _ = instruments
    srs_V_1, srs_V_2, srs_V_3, srs_V_4, srs_V_5, srs_V_6, I_yoko_top, I_yoko_bot = biases

    mods = ['1', '2', '3', '4', '5', '6']
    SRS_init(SRS, mods, np.array([srs_V_1, srs_V_2, srs_V_3, srs_V_4, srs_V_5, srs_V_6]))
    yoko_init(yoko_top, I_yoko_top)
    yoko_init(yoko_bot, I_yoko_bot)
    siglent.write('C1:OUTP ON')
    siglent.write('C2:OUTP ON')
    rigol.set_output(True, channel = 1)
    rigol.set_output(True, channel = 2)
    return
    
            
#%% Turn on siglent and rigol
def turn_on_siglent(siglent, channels = ['C1', 'C2']):
    for channel in channels:
        siglent.write(f'{channel}:OUTP ON')
    return

def turn_on_rigol(rigol, channels = [1, 2]):
    for channel in channels:
        rigol.set_output(True, channel = channel)
    return

#%% SRS Functions
def SRS_write(SRS, mod, v):
    SRS.write("conn "+mod+",'xyz'")
    SRS.write('VOLT '+str(np.round(v, 3)))
    SRS.write('xyz')
    
def SRS_init(SRS, mods, Vs):
    if type(mods) != list:
        mods = [mods]
    if type(Vs) != list:
        Vs = [Vs]
    for i,mod in enumerate(mods):
        SRS_write(SRS, mod, Vs[i])
        write_str = 'SNDT ' + str(mod) + ',\"' + 'OPON' + '\"'
        SRS.write(write_str) 
            
#%% Lecroy Functions

def get_traces(lecroy, avgs=100, channel='C3', avg_std=True):
    time_ax, _ = lecroy.get_single_trace(channel = channel)
    curves = np.zeros_like(time_ax)
    for n in range(avgs):
        _, V_sq = lecroy.get_single_trace(channel = channel)
        curves = np.vstack((curves, V_sq))
        #curves[n] = I_ss
    lecroy.set_trigger_mode(trigger_mode = 'Normal')
    if avg_std:
        curves = avg_and_std(time_ax, curves)
    return curves

def get_V_avg(lecroy, channel='C3'):
    _, V_arr = lecroy.get_single_trace(channel=channel)
    return np.mean(V_arr)

def get_many_traces(lecroy, avgs=100, channels=['C1', 'C2', 'C3', 'C4'], avg_std=True):
    time_ax, _ = lecroy.get_multiple_traces(channels=channels)
    curves = np.empty(len(channels), dtype=object)
    for i in np.arange(len(curves)):
        curves[i] = np.zeros_like(time_ax[i], dtype=float)
    for n in range(avgs):
        _, V_sq = lecroy.get_multiple_traces(channels=channels)
        for i in np.arange(len(channels)):
            curves[i] = np.vstack((curves[i], V_sq[i]))
    lecroy.set_trigger_mode(trigger_mode = 'Normal')
    if avg_std:
        for i in np.arange(len(channels)):
            curves[i] = avg_and_std(time_ax[i], curves[i])
    return curves

def avg_and_std(time, curves):
    STD = np.std(curves, axis = 0)
    avg_curve = np.average(curves, axis = 0)
    return [time, avg_curve, STD]


def get_averaged_data(lecroy, averages = 100, channel = 'C3'):
    lecroy.clear_sweeps()
    time.sleep(0.1)
    for n in range(averages + 10):
        lecroy.get_single_trace(channel = channel)
    trace = lecroy.get_single_trace(channel = channel)
    return trace
#%% Agilent Functions
#%% Rigol Functions
def rigol_progamming(rigol, N_1 = 10, N_2 = 10, prog_freq = 400e3, vhigh1 = 2.0, vhigh2 = 2.0, delay_1 = 0, delay_2 = 10e-6, prog_width = 100e-9):
    rigol.setup_pulse(freq=prog_freq, vlow = 0.0, vhigh=vhigh1, width = prog_width, delay = 0, channel = 1)
    rigol.setup_pulse(freq=prog_freq, vlow = 0.0, vhigh=vhigh2, width = prog_width, delay = 0, channel = 2)    
    rigol.set_burst_mode(num_cycles = N_1, channel = 1, trigger_source = 'EXT', delay = delay_1)
    rigol.set_burst_mode(num_cycles = N_2, channel = 2, trigger_source = 'EXT', delay = delay_2)
    return

def rigol_progamming_2(rigol, N_1 = 10, N_2 = 10, prog_freq_1 = 400e3, prog_freq_2 = 400e3, vhigh1 = 2.0, vhigh2 = 2.0, delay_1 = 0, delay_2 = 10e-6, prog_width = 100e-9, trig = 'EXT'):
    rigol.setup_pulse(freq=prog_freq_1, vlow = 0.0, vhigh=vhigh1, width = prog_width, delay = 0, channel = 1)
    rigol.setup_pulse(freq=prog_freq_2, vlow = 0.0, vhigh=vhigh2, width = prog_width, delay = 0, channel = 2)    
    rigol.set_burst_mode(num_cycles = N_1, channel = 1, trigger_source = trig, delay = delay_1)
    rigol.set_burst_mode(num_cycles = N_2, channel = 2, trigger_source = trig, delay = delay_2)
    return


def program_arb_rigol(rigol, voltages, freq, channel = 1, delay = 0, v_pulse = 2.2):
    rigol.set_output(False, channel = channel)
    rigol.write(':SOURCE'+str(channel)+':FUNCTION USER')
    rigol.write(':SOURCE'+str(channel)+':FUNCTION:ARB:MODE INT')
    rigol.write(':DATA:POINts VOLATILE, '+str(len(voltages)))
    rigol.set_burst_mode(num_cycles = 1, channel = channel, trigger_source = 'EXT', delay = delay)
    rigol.set_freq(freq = freq, channel = channel)
    rigol.set_vhighlow(vlow=0.0, vhigh=v_pulse, channel = channel)
    voltage_scale = 16383
    for i,v in enumerate(voltages):
        prog_string = ':SOURCE'+str(channel)+':DATA:VAL VOLATILE,'+str(i) + ','+ str(v*voltage_scale)
        rigol.write(prog_string)

    rigol.set_output(True, channel = channel)
    return
    
def random_pulse_train(rigol, N_clock = 10, F_clock = 700e3, PW = 500e-9, N_reset = 50, reset_pad = 10, start_pulse = 1, v_pulse = 2.2, pulse_sequences = ([0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 1])):
    if pulse_sequences == 'Random':
        voltages_ch1 = np.asarray(np.random.randint(2, size=N_clock))
        voltages_ch2 = np.asarray(np.random.randint(2, size=N_clock))
    else:
        voltages_ch1, voltages_ch2 = pulse_sequences
        N_clock = len(voltages_ch1)
    for i,v1 in enumerate(voltages_ch1):
        if voltages_ch2[i] == v1:
            voltages_ch2[i] = 0
    new_voltages_ch1 = pad_random_train(voltages_ch1, PW = PW, F_clock = F_clock, start_pulse = start_pulse)
    new_voltages_ch2 = pad_random_train(voltages_ch2, PW = PW, F_clock = F_clock, reset = True, reset_num = N_reset, start_pulse = 0, reset_pad = reset_pad)
    wf_freq_ch1 = F_clock/(N_clock+1)#F_clock/len(new_voltages_ch1)
    wf_freq_ch2 = F_clock/(N_clock + N_reset + reset_pad+1)#F_clock/len(new_voltages_ch2)#(N_clock + N_reset)/F_clock
    program_arb_rigol(rigol, new_voltages_ch1, freq = wf_freq_ch1, channel = 1, delay = 0, v_pulse = v_pulse)
    program_arb_rigol(rigol, new_voltages_ch2, freq = wf_freq_ch2, channel = 2, delay = 0, v_pulse = v_pulse)
    plt.figure()
    plt.plot(new_voltages_ch2, 'o-')
    plt.plot(0.5*new_voltages_ch1, 'o-')
    return (new_voltages_ch1, new_voltages_ch2)

def pad_random_train(voltages, PW = 500e-9, F_clock = 700e3, reset = False, reset_num = 50, start_pulse = None, reset_pad = 10):
    voltages = np.insert(voltages, 0, 0)
    # if start_pulse is not None:
    #     voltages[1] = start_pulse
    new_voltages = []
    
    N_points = int(np.round(1/(F_clock*PW))) 
    if N_points < 4:
        PW = 1/(4*F_clock)
        N_points = 4
    else:
        PW = 1/(N_points*F_clock)
        
    zero_pad = np.zeros(N_points)
    one_pad = np.zeros_like(zero_pad)
    one_pad[0] = 1
    one_pad[1] = 1
    
    for v in voltages:
        if v == 0:
            new_voltages = np.append(new_voltages, zero_pad)
        if v == 1:
            new_voltages = np.append(new_voltages, one_pad)
    if reset == True:
        for i in range(0,reset_pad):
            new_voltages = np.append(new_voltages, zero_pad)
        for i in range(0,reset_num):
            new_voltages = np.append(new_voltages, one_pad)
    return new_voltages

#%% Yoko Functions
def yoko_init(yoko, I):
    yoko.write(":SOUR:FUNC CURR")
    yoko.write(":SOUR:LEV "+ str(I))
    yoko.write(":OUTP ON")

#%% Siglent Functions

def siglent_laser_pulses(siglent, N = 10, freq = 700e3, h_level = 2.0, delay = 0.0, channel='C2'):
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV TRSR,EXT')
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP LOAD,HZ')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV WVTP,PULSE')
    time.sleep(0.3)
    siglent.write(f'{channel}:BSWV FRQ,{freq}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV HLEV,' + str(h_level))
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV LLEV, 0')
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP ON')
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV STATE,ON')
    time.sleep(0.1)
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV TIME,{N}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV GATE_NCYC,NCYC')
    time.sleep(0.1)
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV TIME,{N}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV DLAY,' + str(delay))
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV TIME,{N}')
    return


def siglent_trigger(siglent, freq = 10e3, h_level=5.0):
    time.sleep(0.1)
    siglent.write('C1:OUTP LOAD,HZ')
    time.sleep(0.1)
    siglent.write('C1:BSWV WVTP,SQUARE')
    time.sleep(0.1)
    siglent.write('C1:BSWV FRQ,' + str(freq))
    time.sleep(0.1)
    siglent.write('C1:BSWV HLEV,' + str(h_level))
    time.sleep(0.1)
    siglent.write('C1:BSWV LLEV, 0')
    time.sleep(0.1)
    siglent.write('C1:OUTP ON')
    return

def siglent_constant_laser(siglent, freq=10e3, h_level=5.0, channel='C2'):
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP LOAD,HZ')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV WVTP,PULSE')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV FRQ,' + str(freq))
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV HLEV,' + str(h_level))
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV LLEV, 0')
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP ON')
    return

# def siglent_trigger_pulse(siglent, N=1, freq = 1e3, h_level=5.0, channel='C1', burst_per = 1/(2e3)):
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BTWV TRSR,INT')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:OUTP LOAD,HZ')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BSWV WVTP,PULSE')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BSWV FRQ,{freq}')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BSWV HLEV,{h_level}')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BSWV LLEV, 0')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:OUTP ON')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BTWV STATE,ON')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BTWV GATE_NCYC,NCYC')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BTWV PRD,{burst_per}')
#     time.sleep(0.1)
#     siglent.write(f'{channel}:BTWV TIME,{N}')
#     time.sleep(0.1)
#     return

def siglent_trigger_pulse(siglent, N=1, freq = 1e3, h_level=5.0, channel='C1'):
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV TRSR,INT')
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP LOAD,HZ')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV WVTP,PULSE')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV FRQ,{freq}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV HLEV,{h_level}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV LLEV, 0')
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP ON')
    time.sleep(0.1)
    return

def siglent_pulse_trigger_setup(siglent, h_level_ch1=5.0, N = 10, freq = 700e3, h_level_ch2 = 2.0, delay = 0.0, burst_per = 1e-3):
    time.sleep(0.1)
    siglent.write('C1:BTWV TRSR,MAN')
    time.sleep(0.1)
    siglent.write('C1:OUTP LOAD,HZ')
    time.sleep(0.1)
    siglent.write('C1:BSWV WVTP,PULSE')
    time.sleep(0.1)
    siglent.write('C1:BSWV HLEV,' + str(h_level_ch1))
    time.sleep(0.1)
    siglent.write('C1:BSWV LLEV, 0')
    time.sleep(0.1)
    siglent.write('C1:OUTP ON')
    time.sleep(0.1)
    siglent.write('C1:BTWV STATE,ON')
    time.sleep(0.1)
    siglent.write('C1:BTWV GATE_NCYC,NCYC')
    time.sleep(0.1)
    siglent.write('C1:BTWV TIME,1')# + str(1))
    time.sleep(0.1)
    siglent.write('C1:BTWV DLAY,0')# + str(delay))
    
    
    time.sleep(0.1)
    siglent.write('C2:BTWV TRSR,INT')
    time.sleep(0.1)
    siglent.write('C2:OUTP LOAD,HZ')
    time.sleep(0.1)
    siglent.write('C2:BSWV WVTP,PULSE')
    time.sleep(0.1)
    siglent.write('C2:BSWV FRQ,' + str(freq))
    time.sleep(0.1)
    siglent.write('C2:BSWV HLEV,' + str(h_level_ch2))
    time.sleep(0.1)
    siglent.write('C2:BSWV LLEV, 0')
    time.sleep(0.1)
    siglent.write('C2:OUTP ON')
    time.sleep(0.1)
    siglent.write('C2:BTWV STATE,ON')
    time.sleep(0.1)
    siglent.write('C2:BTWV GATE_NCYC,NCYC')
    time.sleep(0.1)
    siglent.write('C2:BTWV PRD,'+str(burst_per))
    time.sleep(0.1)
    siglent.write('C2:BTWV TIME,' + str(N))
    time.sleep(0.1)
    siglent.write('C2:BTWV DLAY,' + str(delay))
    return

def siglent_ramp(siglent, freq = 10e3, h_level = 0.5, channel = 'C1', symmetry=99):
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP LOAD,HZ')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV WVTP,RAMP')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV SYM,{symmetry}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV FRQ,{freq}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV HLEV,{h_level}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV LLEV,{-h_level}')
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP ON')
    return

def siglent_ramp_burst(siglent, freq = 10e3, h_level = 0.5, N = 3, channel = 'C1', burst_per = 1e-4, delay = 0, symmetry = 90, source = 'EXT'):
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV TRSR,{source}')
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP LOAD,HZ')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV WVTP,RAMP')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV SYM,{symmetry}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV FRQ,{freq}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV HLEV,{h_level}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BSWV LLEV,{-h_level}')
    time.sleep(0.1)
    siglent.write(f'{channel}:OUTP ON')
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV STATE,ON')
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV GATE_NCYC,NCYC')
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV PRD,{burst_per}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV TIME,{N}')
    time.sleep(0.1)
    siglent.write(f'{channel}:BTWV DLAY,{delay}')
    return



















