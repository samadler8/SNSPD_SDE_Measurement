import os
import pickle
import logging
import time

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta



from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.ando_aq8201418 import AndoAQ8201418
from amcc.instruments.ando_aq8201412 import AndoAQ8201412
from amcc.instruments.ando_aq820133 import AndoAQ820133
from amcc.instruments.fiberControl_MPC101 import FiberControlMPC101
from amcc.instruments.agilent_34411a import Agilent34411A

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


srs = SIM928('GPIB1::2::INSTR', 5)
counter = Agilent53131a('GPIB1::5::INSTR')
laser_sw = AndoAQ8201418('GPIB1::4::INSTR',4)
sw = AndoAQ8201412('GPIB1::4::INSTR', 7, 1)
pm_sw = AndoAQ8201412('GPIB1::4::INSTR', 7, 2)
att1 = AndoAQ820133('GPIB1::4::INSTR',5)
att2 = AndoAQ820133('GPIB1::4::INSTR', 6)
pc = FiberControlMPC101('GPIB1::3::INSTR')
multi = Agilent34411A('GPIB1::21::INSTR')


monitor_port = 1
detector_port = 2

ingaas_port = 1
thermal_port = 2

att_list = [att1, att2]


instruments = {'srs': srs,
    'counter': counter,
    'laser_sw': laser_sw,
    'sw': sw,
    'pm_sw': pm_sw,
    'ingaas_port': ingaas_port,
    'thermal_port': thermal_port,
    'multi': multi,
    'att1': att1,
    'att2': att2,
    'att_list': att_list,
    'pc': pc,
    'monitor_port': monitor_port,
    'detector_port': detector_port,
    }

max_cur = 15e-6
bias_resistor = 97e3
counting_time = 0.5
num_pols = 6
N=10

#wavelengths = [1100, 1250, 1350, 1450, 1566.3, 1650, 1700]
wavelengths=[1550]

def measure_avg_cps(attval,):
    """Measure average counts per second (CPS) at a given trigger voltage."""

    att1.set_att(attval)
    att2.set_att(attval)
    srs.set_output(output=False)
    time.sleep(.1)
    srs.set_output(output=True)
    time.sleep(0.1)  # Allow system to stabilize
    cps_values = [counter.timed_count(counting_time=0.5) / counting_time for _ in range(N)]
    if any(x == 0 for x in cps_values):
        avg_cps = float('inf')
    else:
        avg_cps = np.mean(cps_values)
    return avg_cps

def get_att_value(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath=None, name='', trigger_voltage=0.1, pol_counts_filepath=None):
    if IV_pickle_filepath is None:
        ic = 12e-6
    else:
        ic = get_ic(IV_pickle_filepath)
    cur = 0.9*ic
    srs.set_voltage(cur*bias_resistor)
    if pol_counts_filepath is None:
        maxpol_settings = (0, 0, 0)
    else:
        with open(pol_counts_filepath, 'rb') as file:
            pol_data = pickle.load(file)
        maxpol_settings = max(pol_data, key=pol_data.get)
    logger.info(f'maxpol_settings: {maxpol_settings}')

    pc.set_waveplate_positions(maxpol_settings)

    # Binary search for trigger voltage
    low_attval = 0
    high_attval = 100
    tolerance = .01
    target_cps = 200000

    laser_sw.set_route(3)

    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

    while abs(high_attval - low_attval) > tolerance:
        mid_attval = (low_attval + high_attval) / 2
        avg_cps = measure_avg_cps(mid_attval,)

        logger.info(f"Attenuation Value: {mid_attval:.3f}, Avg CPS: {avg_cps:.3f}")

        if avg_cps < target_cps:
            high_attval = mid_attval
        else:
            low_attval = mid_attval

    return mid_attval


 

if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    name = 'saaed2um'
    # IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "saaed2um_IV_curve_data__20250225-141516.pkl")

    # trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    trigger_voltage = 0.01

    for wavelength in wavelengths:
        name_wav = f'{name}_{wavelength}'
        attval = get_att_value(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, name=name, trigger_voltage=trigger_voltage)
       
        pol_counts_filepath=sweep_polarizations(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, attval=attval, name=name_wav, num_pols=num_pols, trigger_voltage=trigger_voltage, counting_time=0.5, N=3)
        #os.path.join(current_file_dir, "data_sde", "saeed2um_pol_data_snspd_splice1__20250115-213240.pkl")

        attval = get_att_value(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, name=name_wav, trigger_voltage=trigger_voltage,pol_counts_filepath=pol_counts_filepath)

        data_filepath = SDE_Counts_Measurement(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval, name=name_wav, trigger_voltage=trigger_voltage)
        attenuator_calibration_filepath = attenuator_calibration(now_str=now_str, attval=attval)
