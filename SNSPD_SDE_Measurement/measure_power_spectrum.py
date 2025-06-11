#%%
import os
import pickle
import logging
import time

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta



from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.ando_aq82011 import AndoAQ82011
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.ando_aq8201418 import AndoAQ8201418
from amcc.instruments.ando_aq8201412 import AndoAQ8201412
from amcc.instruments.ando_aq82013 import AndoAQ82013
from amcc.instruments.ando_aq82012 import AndoAQ82012
from amcc.instruments.fiberControl_MPC101 import FiberControlMPC101
from amcc.instruments.agilent_34411a import Agilent34411A
# from amcc.instruments.horiba_ihr320 import HoribaIHR320
from amcc.instruments.agilent_8164a import Agilent8164A
from amcc.instruments.thorlabs_lfltm import ThorLabsLFLTM
from amcc.instruments.agilent_8163a import Agilent8163A


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

GPIB_num = '0'

srs = SIM928(f'GPIB{GPIB_num}::2::INSTR', 5)
counter = Agilent53131a(f'GPIB{GPIB_num}::5::INSTR')
laser1566 = AndoAQ82011(f'GPIB{GPIB_num}::4::INSTR', 1)
laser1525 = AndoAQ82011(f'GPIB{GPIB_num}::4::INSTR', 2)
# laser1621 = AndoAQ82011(f'GPIB{GPIB_num}::4::INSTR', 3)
laser_sw = AndoAQ8201418(f'GPIB{GPIB_num}::4::INSTR',3)
att1 = AndoAQ82013(f'GPIB{GPIB_num}::4::INSTR', 4)
att2 = AndoAQ82013(f'GPIB{GPIB_num}::4::INSTR', 5)
att3 = AndoAQ82013(f'GPIB{GPIB_num}::4::INSTR', 6)
sw = AndoAQ8201412(f'GPIB{GPIB_num}::4::INSTR', 7, 1)
mpm_sw = AndoAQ8201412(f'GPIB{GPIB_num}::4::INSTR', 7, 2)
ando_pm = AndoAQ82012(f'GPIB{GPIB_num}::4::INSTR', 8)

# laser635 = ThorLabsLFLTM('COMxxxx')
# laser2000 = ThorLabsLFLTM('COMxxxx')

pc = FiberControlMPC101(f'GPIB{GPIB_num}::3::INSTR')
multi = Agilent34411A(f'GPIB{GPIB_num}::21::INSTR')
# ingaas_pm=Agilent8164A('GPIB0::23::INSTR')
# spectrometer=HoribaIHR320()
cpm = Agilent8163A(f'GPIB{GPIB_num}::9::INSTR', 1)


monitor_port = 1
detector_port = 2

laser1566_port=1
laser2000_port=2
# spectrometer_port=3
laser635_port=3
laser1525_port=4
# laser1621_port=5

# ingaas_port = 1
ando_port = 1
thermal_port = 2

FilterLP1000 = 4

att_list = [att1, att2, att3]




instruments = {'srs': srs,
    'counter': counter,
    'laser_sw': laser_sw,
    'sw': sw,
    'mpm_sw': mpm_sw,
    # 'ingaas_port': ingaas_port,
    'ando_port': ando_port,
    'thermal_port': thermal_port,
    'multi': multi,
    'att1': att1,
    'att2': att2,
    'att3': att3,
    'att_list': att_list,
    'pc': pc,
    'cpm': cpm,
    'ando_pm': ando_pm,
    'laser635_port': laser635_port,
    'laser1525_port': laser1525_port,
    # 'spectrometer_port': spectrometer_port,
    'laser1566_port': laser1566_port,
    # 'laser1621_port': laser1621_port,
    'laser2000_port': laser2000_port,
    'monitor_port': monitor_port,
    'detector_port': detector_port,
    # 'spectrometer': spectrometer,
    'FilterLP1000': FilterLP1000,
    # 'ingaas_pm':ingaas_pm,
    }

max_cur = 15e-6
bias_resistor = 97e3
counting_time = 0.5
num_pols = 9


 
#%%
if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    name = 'saaed2um'

    # wavelengths = [635, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1525.661, 1550, 1566.314, 1600, 1620.5, 1650, 1700, 2000]
    wavelengths = [635, 1525.661, 1566.314, 2000]

    NIST_pm_calib_path = os.path.join(current_file_dir, 'calibration_power_meter', 'SWN HP81521B 2933G05261.xlsx')
    calib_df = pd.read_excel(NIST_pm_calib_path, sheet_name='Data')
    calib_df_wav = calib_df['Wav (nm)'].values

    # light_sources = ['ando_laser', 'thor_laser', 'spectrometer']
    light_sources = ['ando_laser', 'thor_laser']

    # mpm_types = ['ando', 'InGaAs', 'thermal']
    mpm_types = ['ando','thermal']

    switch_cal_y_n = input("Do you want to calibrate the optical swtich using the nist calibrated power meter? \n")
    if switch_cal_y_n in ['1', 'Y', 'y']:
        input("Ensure spliced to CPM not SNSPD\nPress anything to continue\n")
        for wavelength in tqdm(wavelengths):
            if wavelength > max(calib_df_wav) or wavelength < min(calib_df_wav):
                continue
            else:
                now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
                mpm_types_temp = mpm_types
                light_sources_temp = light_sources

                if wavelength > 2000:
                    mpm_types_temp.remove('InGaAs')
                    mpm_types_temp.remove('ando')
                    light_sources_temp.remove('spectrometer')
                elif wavelength > 1700:
                    mpm_types_temp.remove('InGaAs')
                    mpm_types_temp.remove('ando')
                if wavelength < 800:
                    mpm_types_temp.remove('InGaAs')
                    mpm_types_temp.remove('ando')
                    light_sources_temp.remove('spectrometer')
                if wavelength < 1000:
                    light_sources_temp.remove('spectrometer')

                if wavelength not in [1566.314, 1525.661, 1620.5]:
                    light_sources_temp.remove('ando_laser')

                if wavelength not in [635, 2000]:
                    light_sources_temp.remove('thor_laser')

                instruments['mpms'] = []
                for mpm_type in mpm_types_temp:
                    if mpm_type == 'ando':
                        instruments['mpms'].append(instruments['ando_pm'])
                    elif mpm_type == 'InGaAs':
                        instruments['mpms'].append(instruments['ingaas_pm'])
                    elif mpm_type == 'thermal':
                        instruments['mpms'].append(None)

                for light_source in light_sources_temp:
                    if light_source == 'ando_laser':
                        if wavelength==1566.314:
                            laser_sw.set_route(laser1566_port)
                            ando_laser = laser1566
                        elif wavelength==1525.661:
                            laser_sw.set_route(laser1525_port)
                            ando_laser = laser1525
                        # elif wavelength==1620.5:
                        #     laser_sw.set_route(laser1621_port)
                        #     ando_laser = laser1621
                        instruments['laser'] = ando_laser
                    
                    elif light_source == 'thor_laser':
                        if wavelength==635:
                            laser_sw.set_route(laser635_port)
                        elif wavelength==2000:
                            laser_sw.set_route(laser2000_port)
                

                    # elif light_source == 'spectrometer':
                    #     spectrometer.set_wavelength(wavelength)
                    #     laser_sw.set_route(spectrometer_port)

                    name_mpms_src_wav = f'{mpm_types_temp}_{light_source}_{wavelength}'
                    sw.set_route(monitor_port)
                    optical_switch_calibration_filepath = optical_switch_calibration(instruments, name=name_mpms_src_wav, mpm_types=mpm_types_temp, wavelength=wavelength)
                    sw.set_route(monitor_port)
                        
    taus = [2.75, 2.25]
    nonlinearity_factor_filepath = nonlinearity_factor_raw_power_measurements(instruments, now_str=now_str, taus=taus)

    input("Ensure SNSPD is properly connected to srs and counter\nPress anything to continue\n")

    IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    # IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "saaed2um_IV_curve_data__20250307-182549.pkl")
 
    input("Ensure spliced to SNSPD not CPM\nPress anything to continue\n")

    trigger_voltage = find_min_trigger_threshold(instruments)
    # trigger_voltage = 0.01

    for wavelength in tqdm(wavelengths):
        for mpm_type in mpm_types:
            now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
            mpm_types_temp = mpm_types
            light_sources_temp = light_sources

            if wavelength > 2000:
                mpm_types_temp.remove('InGaAs')
                mpm_types_temp.remove('ando')
                light_sources_temp.remove('spectrometer')
            elif wavelength > 1700:
                mpm_types_temp.remove('InGaAs')
                mpm_types_temp.remove('ando')
            if wavelength < 800:
                mpm_types_temp.remove('InGaAs')
                mpm_types_temp.remove('ando')
                light_sources_temp.remove('spectrometer')
            if wavelength < 1000:
                light_sources_temp.remove('spectrometer')

            if wavelength not in [1566.314, 1525.661, 1620.5]:
                light_sources_temp.remove('ando_laser')

            if wavelength not in [635, 2000]:
                light_sources_temp.remove('thor_laser')

            instruments['mpms'] = []
            for mpm_type in mpm_types_temp:
                if mpm_type == 'ando':
                    instruments['mpms'].append(instruments['ando_pm'])
                elif mpm_type == 'InGaAs':
                    instruments['mpms'].append(instruments['ingaas_pm'])
                elif mpm_type == 'thermal':
                    instruments['mpms'].append(None)

            for light_source in light_sources_temp:
                if light_source == 'ando_laser':
                    if wavelength==1566.314:
                        laser_sw.set_route(laser1566_port)
                        ando_laser = laser1566
                    elif wavelength==1525.661:
                        laser_sw.set_route(laser1525_port)
                        ando_laser = laser1525
                    # elif wavelength==1620.5:
                    #     laser_sw.set_route(laser1621_port)
                    #     ando_laser = laser1621
                    instruments['laser'] = ando_laser
                
                elif light_source == 'thor_laser':
                    if wavelength==635:
                        laser_sw.set_route(laser635_port)
                    elif wavelength==2000:
                        laser_sw.set_route(laser2000_port)
            

                # elif light_source == 'spectrometer':
                #     spectrometer.set_wavelength(wavelength)
                #     laser_sw.set_route(spectrometer_port)

                name_mpms_src_wav = f'{mpm_types_temp}_{light_source}_{wavelength}'

                # Find ideal attenuation value (that which gets 300,000 cps at max polarization)
                attval = get_att_value(instruments, IV_pickle_filepath=IV_pickle_filepath, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time,)
                pol_counts_filepath = sweep_polarizations(instruments, now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, attval=attval, name=name_mpms_src_wav, num_pols=num_pols, trigger_voltage=trigger_voltage, counting_time=counting_time, N=3)
                # os.path.join(current_file_dir, "data_sde", "saeed2um_pol_data_snspd_splice1__20250115-213240.pkl")
                attval = get_att_value(instruments, IV_pickle_filepath=IV_pickle_filepath, trigger_voltage=trigger_voltage, bias_resistor=bias_resistor, counting_time=counting_time, pol_counts_filepath=pol_counts_filepath)
                # attval = 10

                data_filepath = SDE_Counts_Measurement(instruments, now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval, name=name_mpms_src_wav, trigger_voltage=trigger_voltage,counting_time=counting_time,bias_resistor=bias_resistor)
                attenuator_calibration_filepath = attenuator_calibration(instruments, now_str=now_str, wavelength=wavelength, attval=attval, mpm_type=mpm_type)
                optical_switch_calibration(instruments, name=name_mpms_src_wav, mpm_types=mpm_types_temp, wavelength=wavelength)
# %%
