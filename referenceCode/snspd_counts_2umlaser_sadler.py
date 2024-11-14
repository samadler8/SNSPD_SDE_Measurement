# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:02:49 2024

@author: sra1
"""

#%% Setup
import serial
import time
import os
import bz2

import _pickle as cPickle
import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt

from my_equipments import counter_setup,counter_impedance,counter_setup_timed_count,counter_set_trigger


# Set up the serial connection
ser = serial.Serial(
    port='COM7',  # Replace 'COM7' with the correct port for your system
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

rm = visa.ResourceManager()
rm.list_resources()
SRS = rm.open_resource('GPIB0::2::INSTR') # SRS SIM 900
counter = rm.open_resource('GPIB0::5::INSTR') # Counter
counter_setup(counter)
counter_impedance(counter,ohms = 50, channel = 1)
counter_setup_timed_count(counter, channel = 1)
counter_set_trigger(counter,trigger_voltage = .045, slope_positive = True, channel = 1)

bias_resistor=97e3
SRS_module=5


#%% Functions
# Function to send a command to the laser
def send_command(command):
    ser.write((command + '\r').encode())  # Append carriage return to the command
    time.sleep(0.1)
    response = ser.read(ser.in_waiting).decode()  # Read the response
    print(f"Response: {response}")
    return response

# Function to turn the laser ON
def laser_on():
    send_command("enable=1")  # Enable the laser

# Function to turn the laser OFF
def laser_off():
    send_command("enable=0")  # Disable the laser

# Function to query laser status
def get_status():
    status = send_command("enable?")
    print(f"Laser status: {'ON' if status.strip() == '1' else 'OFF'}")


def timed_count(self, counting_time = 0.1):
    self.write(':TOT:ARM:STOP:TIM %0.3f' % counting_time) # Set stop time to # of seconds
    dcr = self.query(':READ?')
    dcr = float(dcr)/counting_time
    # time.sleep(counting_time + 0.1)
    return dcr

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
    return
 
# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data


# Sweeps Bias CUrrent and Counts Photons
def get_counts(Cur_Array):
    counting_time = 0.3
    Count_Array=np.zeros(len(Cur_Array))
    
    SRS.write("conn "+str(SRS_module)+",'xyz'")
    SRS.write('VOLT '+str(0))
    SRS.write('xyz')
    SRS.write('SNDT ' + str(SRS_module) + ',\"' + 'OPON' + '\"') 
    time.sleep(1)

    for i in np.arange(len(Cur_Array)):
        this_volt=round(Cur_Array[i]*1e-6*bias_resistor,3)
        
        SRS.write("conn "+str(SRS_module)+",'xyz'")
        SRS.write('VOLT '+str(this_volt))
        SRS.write('xyz')
        
        time.sleep(.1)
        Count_Array[i] = timed_count(counter,counting_time=counting_time)
        print(str(this_volt)+' - '+str(Count_Array[i]))
    
    #bring back bias to zero when done
    SRS.write("conn "+str(SRS_module)+",'xyz'")
    SRS.write('VOLT '+str(0))
    SRS.write('xyz')
    SRS.write('SNDT ' + str(SRS_module) + ',\"' + 'OPOF' + '\"')
    
    return Count_Array

#%% Main code
if __name__ == "__main__":
    minCur=2 #uA
    maxCur=7 #uA
    numCur=100
    
    Cur_Array=np.round(np.linspace(minCur,maxCur,numCur),8)
    
    
    time_str = time.strftime("%Y%m%d-%H%M%S")
    filename = f'lollipop_w130_counts_2micronLight__{time_str}'
    
    
    # Counts
    laser_on()
    Count_Array = get_counts(Cur_Array)
    
    # Dark Counts
    # Turn laser OFF
    laser_off()
    Dark_Count_Array = get_counts(Cur_Array)
    
    
    # save data
    data_dict = {
        'Cur_Array': list(Cur_Array),
        'Count_Array': list(Count_Array),
        'Dark_Count_Array': list(Dark_Count_Array),
        }
    
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    compressed_pickle(filepath, data_dict)
    
    # load and plot data
    data_dict=decompress_pickle(filepath)
    Cur_Array=np.array(data_dict['Cur_Array'])
    Count_Array=np.array(data_dict['Count_Array'])
    Dark_Count_Array=np.array(data_dict['Dark_Count_Array'])
    
    os.makedirs('pics', exist_ok=True)
    figpath = os.path.join('pics', filename)
    
    plt.close('all')
    plt.figure(figsize = [20,10])
    plt.plot(Cur_Array, Count_Array, '-*', color = 'cyan', label = 'Counts')
    plt.plot(Cur_Array, Dark_Count_Array, '-*', color = 'red', label = 'Dark Counts')
    plt.plot(Cur_Array, np.maximum(Count_Array - Dark_Count_Array, 0), '-*', color = 'green', label = 'Counts - Dark Counts')
    plt.title(f'{filename}')
    plt.xlabel('Bias current [uA]')
    plt.ylabel('Counts [per sec]')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
    plt.tight_layout()
    # plt.yscale('log')
    # plt.savefig(f'{figpath}_log.png')
    # plt.savefig(f'{figpath}_log.pdf')
    # plt.yscale('linear')
    plt.savefig(f'{figpath}.png')
    plt.savefig(f'{figpath}.pdf')
    
    ser.close()
