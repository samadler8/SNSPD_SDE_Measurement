# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:01:06 2024

@author: sra1
"""

# %% Setup
import time
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta 

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.thorlabs_lfltm import ThorLabsLFLTM
from amcc.instruments.agilent_34411a import Agilent34411A

from SNSPD_SDE_Measurement.measurement_helpers import *

sim900port = 5
srs = SIM928('GPIB0::2::INSTR', sim900port)
counter = Agilent53131a('GPIB0::5::INSTR')
laser = ThorLabsLFLTM('COM7')
multi = Agilent34411A('GPIB0::21::INSTR')

instruments = {'srs': srs,
    'counter': counter,
    'laser': laser,
    }

bias_resistor=97e3
trigger_voltage = 0.12 #V - not sure why it's so high
counting_time = 1
max_cur = 15e-6

counter.basic_setup()
counter.set_impedance(ohms=50, channel=1)
counter.setup_timed_count(channel=1)
counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

srs.set_voltage(0)
srs.set_output(output=False)

laser.disable()

# %% temperature dependence sweep
def temperature_dependence_sweep(IV_pickle_filepath='', trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=0.5, N=1):
    ic = get_ic(IV_pickle_filepath)

    num_biases = 100
    Cur_Array = np.linspace(ic * 0.2, ic * 1.1, num_biases)

    
    end_time = datetime.now() + timedelta(hours=3) 

    data = []
    temperaturedependence_filename = f'temperature_dependence_2micronLight'
    temperaturedependence_filepath = os.path.join('data_temperatureDependence', temperaturedependence_filename)
    os.makedirs('data_temperatureDependence', exist_ok=True)

    i = 0
    now = datetime.now()
    while now < end_time:
        # Counts
        laser.enable()
        Count_Array = get_counts(Cur_Array, srs=srs, counter=counter, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time, N=N)

        # Dark Counts
        laser.disable()
        Dark_Count_Array = get_counts(Cur_Array, srs=srs, counter=counter, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time, N=N)


        # save data
        data_dict = {
            'Cur_Array': list(Cur_Array),
            'Count_Array': list(Count_Array),
            'Dark_Count_Array': list(Dark_Count_Array),
            }
        
        data.append((now, data_dict))

        i += 1
        if i == 10:
            i = 0
            with open(temperaturedependence_filepath, "wb") as file:
                pickle.dump(data_dict, file)
        now = datetime.now()
    
    with open(temperaturedependence_filepath, "wb") as file:
        pickle.dump(data_dict, file)

    return temperaturedependence_filepath



# %% Main code
if __name__ == '__main__':

    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    print("STARTING: Algorithm S3.0.1. SNSPD IV Curve")
    IV_pickle_filepath = SNSPD_IV_Curve(now_str=now_str, max_cur=max_cur)
    print("COMPLETED: Algorithm S3.0.1. SNSPD IV Curve")
    
    print("STARTING: Temperature Dependence Sweep")
    temperaturedependence_filepath = temperature_dependence_sweep(IV_pickle_filepath='', )
    print("COMPLETED: Temperature Dependence Sweep")