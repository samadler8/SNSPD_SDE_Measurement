# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:01:06 2024

@author: sra1
"""

# %% Setup
import os
import pickle
import logging

import numpy as np

from datetime import datetime, timedelta 

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.thorlabs_lfltm import ThorLabsLFLTM
from amcc.instruments.agilent_34411a import Agilent34411A

from helpers import *
from measure_helpers import *

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("measure_temperature_dependence.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

srs = SIM928('GPIB0::2::INSTR', 5)
counter = Agilent53131a('GPIB0::5::INSTR')
laser = ThorLabsLFLTM('COM7')
multi = Agilent34411A('GPIB0::21::INSTR')

instruments = {'srs': srs,
    'counter': counter,
    'laser': laser,
    }


# %% temperature dependence sweep
def temperature_dependence_sweep(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', trigger_voltage=1.2, bias_resistor=100e3, counting_time=0.5, N=1):
    logger.info("STARTING: Temperature Dependence Sweep")
    ic = get_ic(IV_pickle_filepath)

    Cur_Array = np.linspace(ic * 0.2, ic * 1.1, 100)

    end_time = datetime.now() + timedelta(hours=3) 

    data_dict = {}
    output_dir = os.path.join(current_file_dir, 'data_temperatureDependence')
    os.makedirs('data_temperatureDependence', exist_ok=True)
    filename = f'temperature_dependence_2micronLight__{now_str}'
    filepath = os.path.join(output_dir, filename)
    
    i = 0
    now_i = datetime.now()
    while now_i < end_time:
        # Counts
        
        laser.enable()
        now_i = datetime.now()
        Count_Array = get_counts(Cur_Array, srs=srs, counter=counter, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time, N=N)
        now_f = datetime.now()
        # Dark Counts
        laser.disable()
        Dark_Count_Array = get_counts(Cur_Array, srs=srs, counter=counter, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time, N=N)
        

        # save data
        data_dict_temp = {
            'Cur_Array': np.array(Cur_Array),
            'Count_Array': np.array(Count_Array),
            'Dark_Count_Array': np.array(Dark_Count_Array),
            }
        now = datetime.fromtimestamp((now_i.timestamp() + now_f.timestamp()) / 2)
        data_dict[now] = data_dict_temp

        i += 1
        if i%10 == 0:
            with open(filepath, "wb") as file:
                pickle.dump(data_dict, file)
    with open(filepath, "wb") as file:
        pickle.dump(data_dict, file)

    readable_output_dir = os.path.join(current_file_dir, 'readable_data_temperatureDependence')
    os.makedirs(readable_output_dir, exist_ok=True)
    json_filepath = f'{os.path.splitext(filepath)[0]}.json'
    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    logger.info(f"temperature dependence data saved to: {filepath}")
    logger.info("COMPLETED: Temperature Dependence Sweep")

    return temperaturedependence_filepath



# %% Main code
if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())

    bias_resistor=97e3
    counting_time = 1
    max_cur = 15e-6
    
    IV_pickle_filepath = SNSPD_IV_Curve(now_str=now_str, max_cur=max_cur)

    trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    
    temperaturedependence_filepath = temperature_dependence_sweep(IV_pickle_filepath=IV_pickle_filepath, trigger_voltage=trigger_voltage)
    