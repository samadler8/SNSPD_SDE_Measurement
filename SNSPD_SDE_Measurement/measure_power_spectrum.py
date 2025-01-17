import os
import pickle
import logging

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta 

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.agilent_53131a import Agilent53131a

from measure_helpers import *
from helpers import *

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("measure_power_spectrum.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

srs = SIM928('GPIB0::2::INSTR', 5)
counter = Agilent53131a('GPIB0::5::INSTR')

instruments = {'srs': srs,
    'counter': counter,
    }

max_cur = 15e-6
bias_resistor = 97e3
counting_time = 0.5


# Algorithm S3.2. SDE Counts Measurement - True Counting
def Counts_Measurement(now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', name='', trigger_voltage=0.01, ):    
    logger.info("Starting: ")
   
    # Perform measurements
    ic = get_ic(IV_pickle_filepath)
    num_biases = 30
    Cur_Array = np.linspace(ic * 0.2, ic * 1.1, num_biases)

    # Measure counts at max polarization
    logger.warning("Turn On Laser!!!")
    input("Turn on laser and press enter")
    logger.info("Collecting counts")
    Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    logger.info("Counts collected")

    logger.warning("Turn Off Laser!!!")
    input("Turn off laser and press enter")
    logger.info("Collecting dark counts")
    Dark_Count_Array = get_counts(Cur_Array, instruments, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time)
    logger.info("Dark counts collected")

    # Save data
    data_dict = {
        'Cur_Array': list(Cur_Array),
        'Dark_Count_Array': list(Dark_Count_Array),
        'Count_Array': list(Count_Array),
    }    
    
    output_dir = os.path.join(current_file_dir, 'data_ps')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}_counts_data__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(data_dict, file)

    readable_output_dir = os.path.join(current_file_dir, 'readable_data_ps')
    os.makedirs(readable_output_dir, exist_ok=True)
    _, data_filename = os.path.split(os.path.splitext(filepath)[0])
    json_filename = f'{data_filename}.json'
    json_filepath = os.path.join(readable_output_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    logger.info(f"data_dict saved to: {filepath}")
    logger.info("Completed: ")
    return filepath

if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    name = 'saaed2um_silicone_1965_8dB'

    # IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "saeed2um_IV_curve_data__20250115-194327.pkl")

    # trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    trigger_voltage = 0.146

    data_filepath = Counts_Measurement(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, name=name, trigger_voltage=trigger_voltage)