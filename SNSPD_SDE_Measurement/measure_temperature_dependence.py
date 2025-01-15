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
from pathlib import Path

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.agilent_53131a import Agilent53131a
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
multi = Agilent34411A('GPIB0::21::INSTR')

instruments = {'srs': srs,
    'counter': counter,
    'multi': multi,
    }

wavelength = 1566.314  # nm

laser_type = 'ando'
# laser_type = 'thor'

if laser_type == 'thor':
    from amcc.instruments.thorlabs_lfltm import ThorLabsLFLTM
    laser = ThorLabsLFLTM('COM7')
if laser_type == 'ando':
    from amcc.instruments.ando_aq82011 import AndoAQ82011
    from amcc.instruments.ando_aq820133 import AndoAQ820133
    from amcc.instruments.ando_aq8201418 import AndoAQ8201418
    laser = AndoAQ82011('GPIB0::4::INSTR', 1)
    att1 = AndoAQ820133('GPIB0::4::INSTR', 2)
    att2 = AndoAQ820133('GPIB0::4::INSTR', 3)
    att3 = AndoAQ820133('GPIB0::4::INSTR', 4)
    sw = AndoAQ8201418('GPIB0::4::INSTR', 5)
    att_list = [att1, att2, att3]
    monitor_port = 1
    detector_port = 2
    instruments['att1'] = att1
    instruments['att2'] = att2
    instruments['att3'] = att3
    instruments['att_list'] = att_list
    instruments['sw'] = sw
    instruments['monitor_port'] = monitor_port
    instruments['detector_port'] = detector_port
    
    attval = 29
instruments['laser'] = laser


# Initialize and turn on everything
def temperature_dependence_setup():
    # if laser_type == 'thor':
    
    if laser_type == 'ando':
        # laser.std_init()
        # laser.set_lambda(wavelength)
        # laser.enable()

        for att in att_list:
            att.set_att(0)
            att.disable()
        
    counter.basic_setup()
    counter.set_impedance(ohms=50, channel=1)
    counter.setup_timed_count(channel=1)

    srs.set_voltage(0)
    srs.set_output(output=False)


# temperature dependence sweep
def temperature_dependence_sweep(
    now_str: str = "{:%Y%m%d-%H%M%S}".format(datetime.now()), 
    IV_pickle_filepath: str = '', 
    trigger_voltage: float = 1.2, 
    bias_resistor: float = 100e3, 
    counting_time: float = 0.5, 
    N: int = 1
) -> str:
    logger.info("STARTING: Temperature Dependence Sweep")
    ic = get_ic(IV_pickle_filepath)

    Cur_Array = np.linspace(ic * 0.2, ic * 1.1, 100)

    end_time = datetime.now() + timedelta(hours=3) 

    data_dict = {}
    output_dir = os.path.join(current_file_dir, 'data_temperatureDependence')
    os.makedirs('data_temperatureDependence', exist_ok=True)
    filename = f'temperature_dependence_wavelength{round(wavelength)}nm__{now_str}'
    filepath = os.path.join(output_dir, filename)
    
    i = 0
    now = datetime.now()
    while now < end_time:
        # Counts
        
        if laser_type == 'thor':
            laser.enable()
        if laser_type == 'ando':
            sw.set_route(detector_port)
            for att in att_list:
                att.enable()
                att.set_att(attval)
        logger.info(f" Getting light counts")
        Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time, N=N)
        now = datetime.now()

        # Dark Counts
        if laser_type == 'thor':
            laser.disable()
        if laser_type == 'ando':
            sw.set_route(monitor_port)
            for att in att_list:
                att.set_att(0)
                att.disable()
        logger.info(f" Getting dark counts")
        Dark_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time, N=N)
        
        # save data
        logger.info(f" Saving data")
        data_dict_temp = {
            'Cur_Array': np.array(Cur_Array),
            'Count_Array': np.array(Count_Array),
            'Dark_Count_Array': np.array(Dark_Count_Array),
            }
        data_dict[now] = data_dict_temp

        i += 1
        if i%10 == 0:
            with open(filepath, "wb") as file:
                pickle.dump(data_dict, file)
    with open(filepath, "wb") as file:
        pickle.dump(data_dict, file)

    readable_output_dir = os.path.join(current_file_dir, 'readable_data_temperatureDependence')
    os.makedirs(readable_output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(filepath)[0])
    json_filename = f'{data_filename}.json'
    json_filepath = os.path.join(readable_output_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f, indent=4, default=lambda x: x.isoformat() if isinstance(x, datetime) else x.tolist() if hasattr(x, 'tolist') else str(x))

    logger.info(f"temperature dependence data saved to: {filepath}")
    logger.info("COMPLETED: Temperature Dependence Sweep")

    return filepath



# %% Main code
if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())

    bias_resistor=97e3
    counting_time = 1
    max_cur = 15e-6
    name = 'SK3'
    
    # IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "SK3_IV_curve_data__20250114-175627.pkl")

    # trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    trigger_voltage = 0.131
    
    temperatureDependence_filepath = temperature_dependence_sweep(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, trigger_voltage=trigger_voltage)
    