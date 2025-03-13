# %% Imports, parameters, initialization
import os
import time
import pickle
import logging
import json

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.ando_aq82011 import AndoAQ82011
from amcc.instruments.ando_aq820133 import AndoAQ820133
from amcc.instruments.ando_aq8201418 import AndoAQ8201418
from amcc.instruments.ando_aq82012 import AndoAQ82012
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.agilent_8163a import Agilent8163A
from amcc.instruments.fiberControl_MPC101 import FiberControlMPC101
from amcc.instruments.agilent_34411a import Agilent34411A

from measure_helpers import *
from helpers import *

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("snspd_sde_measurement.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

srs = SIM928('GPIB0::2::INSTR', 5)
laser = AndoAQ82011('GPIB0::4::INSTR', 1)
att1 = AndoAQ820133('GPIB0::4::INSTR', 2)
att2 = AndoAQ820133('GPIB0::4::INSTR', 3)
att3 = AndoAQ820133('GPIB0::4::INSTR', 4)
sw = AndoAQ8201418('GPIB0::4::INSTR', 5)
mpm = AndoAQ82012('GPIB0::4::INSTR', 6)
counter = Agilent53131a('GPIB0::5::INSTR')
cpm = Agilent8163A('GPIB0::9::INSTR', 1)
pc = FiberControlMPC101('GPIB0::3::INSTR')
multi = Agilent34411A('GPIB0::21::INSTR')

att_list = [att1, att2, att3]

monitor_port = 1
detector_port = 2

instruments = {'srs': srs,
    'counter': counter,
    'cpm': cpm,
    'counter': counter,
    'pc': pc,
    'multi': multi,
    'laser': laser,
    'att1': att1,
    'att2': att2,
    'att3': att3,
    'att_list': att_list,
    'sw': sw,
    'mpm': mpm,
    'monitor_port': monitor_port,
    'detector_port': detector_port,
    }

rng_dict = {'A': 'AUTO',
    'C': +30,
    'D': +20,
    'E': +10,
    'F': 0,
    'G': -10,
    'H': -20,
    'I' : -30,
    'J' : -40,
    'K': -50,
    'L' : -60,
    'Z' : 'HOLD'
}

# setup constants
wavelength = 1566.314  # nm
bias_resistor = 97e3 #Ohms
init_rng = 0 #dBm

counting_time = 0.5 #s
num_pols = 13
N_init = 3

max_cur = 15e-6 # A

cpm_splice = 2
snspd_splice = '1connectors'


# %%
if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    name = 'SK3'

    # snspd_sde_setup()
        
    # optical_switch_calibration_filepath = optical_switch_calibration(now_str=now_str, )

    # taus = [2.75, 2.25]
    # nonlinearity_factor_filepath = nonlinearity_factor_raw_power_measurements(now_str=now_str, taus=taus)

    # IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "SK3_IV_curve_data__20250116-130752.pkl")

    trigger_voltage = find_min_trigger_threshold(instruments, now_str=now_str)
    # trigger_voltage = 0.151

    

    attval_init = 28

    pol_counts_filepath = sweep_polarizations(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, attval=attval_init, name=name, num_pols=num_pols, trigger_voltage=trigger_voltage, counting_time=0.5, N=3)
    # pol_counts_filepath = os.path.join(current_file_dir, "data_sde", "saeed2um_pol_data_snspd_splice1__20250115-213240.pkl")

    # data_filepath = SDE_Counts_Measurement(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval_init, name=name, trigger_voltage=trigger_voltage)
    # attenuator_calibration_filepath = attenuator_calibration(now_str=now_str, attval=attattval_initval)
    attvals = [attval_init + math.ceil(i) * (-1) ** (2*i) for i in np.arange(0, 6, 0.5)]
    for attval in attvals:
        attval = int(round(attval))
        data_filepath = SDE_Counts_Measurement(now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval, name=name, trigger_voltage=trigger_voltage)
        attenuator_calibration_filepath = attenuator_calibration(now_str=now_str, attval=attval)
        

    