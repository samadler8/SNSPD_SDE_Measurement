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
from snspd_sde_measurement import SDE_Counts_Measurement, sweep_polarizations, attenuator_calibration

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
    Ando
    Agilent
    Spectrometer
    polaroization controller
    }

max_cur = 15e-6
bias_resistor = 97e3
counting_time = 0.5
num_pols = 6

wavelengths = [1100, 1250, 1350, 1450, 1566.3, 1650, 1700]

def measure_avg_cps(attval, trigger_voltage=0.1):
    """Measure average counts per second (CPS) at a given trigger voltage."""
    Set attenuation value
    Turn off srs
    time.sleep(0.1)
    Turn on SRS
    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=channel)
    time.sleep(0.1)  # Allow system to stabilize
    cps_values = [counter.timed_count(counting_time=counting_time) / counting_time for _ in range(N)]
    if any cps_value == 0:
        avg_cps = inf
    else:
        avg_cps = np.mean(cps_values)
    return avg_cps

def get_att_value(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath='', name='', trigger_voltage=0.12, pol_counts_filepath=None):
    ic = get_ic(IV_pickle_filepath)
    cur = 0.9*ic
    srs.set_voltage(cur*bias_resistor)
    if pol_counts_filepath is None:
        maxpol_settings = (0, 0, 0)
        
    else:
        with open(pol_counts_filepath, 'rb') as file:
            pol_data = pickle.load(file)

        # Find the tuple with the highest count
        maxpol_settings = max(pol_data, key=pol_data.get)

    pc.set_waveplate_positions(maxpol_settings)

    # Binary search for trigger voltage
    low_attval = 0
    high_attval = 40
    tolerance = 0.01*ic
    target_cps = 400000

    while abs(high_attval - low_attval) > tolerance:
        mid_attval = (low_attval + high_attval) / 2
        avg_cps = measure_avg_cps(mid_attval)
        
        logger.info(f"Attenuation Value: {mid_attval:.3f}, Avg CPS: {avg_cps:.3f}")

        if avg_cps > target_cps:
            high_attval = mid_attval
        else:
            low_attval = mid_attval
    
    return mid_attval




if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    name = 'saaed2um'

    # IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "saeed2um_IV_curve_data__20250115-194327.pkl")

    trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    # trigger_voltage = 0.151

    for wavelength in wavelengths:
        name_wav = f'{name}_{wavelength}'
        attval = get_att_value(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, name=name, trigger_voltage=trigger_voltage)

        pol_counts_filepath = sweep_polarizations(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, attval=attval, name=name_wav, num_pols=num_pols, trigger_voltage=trigger_voltage, counting_time=0.5, N=3)
        # pol_counts_filepath = os.path.join(current_file_dir, "data_sde", "saeed2um_pol_data_snspd_splice1__20250115-213240.pkl")

        attval = get_att_value(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, name=name_wav, trigger_voltage=trigger_voltage)

        data_filepath = SDE_Counts_Measurement(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval, name=name_wav, trigger_voltage=trigger_voltage)
        attenuator_calibration_filepath = attenuator_calibration(now_str=now_str, attval=attval)