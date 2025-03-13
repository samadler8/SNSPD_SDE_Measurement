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
from amcc.instruments.ando_aq820133 import AndoAQ820133
from amcc.instruments.fiberControl_MPC101 import FiberControlMPC101
from amcc.instruments.agilent_34411a import Agilent34411A
from amcc.instruments.horiba_ihr320 import HoribaIHR320
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


srs = SIM928('GPIB1::2::INSTR', 5)
counter = Agilent53131a('GPIB1::5::INSTR')
laser1566 = AndoAQ82011('GPIB1::4::INSTR', 1)
laser1525 = AndoAQ82011('GPIB1::4::INSTR', 2)
laser1621 = AndoAQ82011('GPIB1::4::INSTR', 3)
# laser635 = ThorLabsLFLTM('COMxxxx')
# laser2000 = ThorLabsLFLTM('COMxxxx')

laser_sw = AndoAQ8201418('GPIB1::4::INSTR',4)
sw = AndoAQ8201412('GPIB1::4::INSTR', 7, 1)
mpm_sw = AndoAQ8201412('GPIB1::4::INSTR', 7, 2)
att1 = AndoAQ820133('GPIB1::4::INSTR', 5)
att2 = AndoAQ820133('GPIB1::4::INSTR', 6)
pc = FiberControlMPC101('GPIB1::3::INSTR')
multi = Agilent34411A('GPIB1::21::INSTR')
ingaas_pm=Agilent8164A('GPIB0::23::INSTR')
spectrometer=HoribaIHR320()
cpm = Agilent8163A('GPIB1::9::INSTR', 1)


monitor_port = 1
detector_port = 2

laser1566_port=1
laser2000_port=2
spectrometer_port=3
laser635_port=4
laser1525_port=5
laser1621_port=6

ingaas_port = 1
thermal_port = 2

FilterLP1000 = 4

att_list = [att1, att2]




instruments = {'srs': srs,
    'counter': counter,
    'laser_sw': laser_sw,
    'sw': sw,
    'mpm_sw': mpm_sw,
    'ingaas_port': ingaas_port,
    'thermal_port': thermal_port,
    'multi': multi,
    'att1': att1,
    'att2': att2,
    'att_list': att_list,
    'pc': pc,
    'cpm': cpm,
    'laser635_port': laser635_port,
    'laser1525_port': laser1525_port,
    'spectrometer_port': spectrometer_port,
    'laser1566_port': laser1566_port,
    'laser1621_port': laser1621_port,
    'laser2000_port': laser2000_port,
    'monitor_port': monitor_port,
    'detector_port': detector_port,
    'spectrometer': spectrometer,
    'FilterLP1000': FilterLP1000,
    'ingaas_pm':ingaas_pm,
    }

max_cur = 15e-6
bias_resistor = 97e3
counting_time = 0.5
num_pols = 9
N = 10

def get_att_value(instruments, IV_pickle_filepath=None, trigger_voltage=0.1, target_cps=250000, pol_counts_filepath=None):
    srs = instruments['srs']
    pc = instruments['pc']
    counter = instruments['counter']
    laser_sw = instruments['laser_sw']
    sw = instruments['sw']
    att_list = instruments['att_list']
    
    # Set SNSPD bias current to 90%
    if IV_pickle_filepath is None:
        ic = 12e-6
    else:
        ic = get_ic(IV_pickle_filepath)
    cur = 0.9*ic
    srs.set_voltage(cur*bias_resistor)
    
    # Set polarization to the max polarization setting or (0, 0, 0) if unknown
    if pol_counts_filepath is None:
        maxpol_settings = (0, 0, 0)
    else:
        with open(pol_counts_filepath, 'rb') as file:
            pol_data = pickle.load(file)
        maxpol_settings = max(pol_data, key=pol_data.get)
    logger.info(f'maxpol_settings: {maxpol_settings}')
    pc.set_waveplate_positions(maxpol_settings)

    # Binary search for trigger voltage
    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)
    sw.set_route(detector_port)
    for att in att_list:
        att.enable()

    low_attval = 0
    high_attval = 60
    tolerance = .005
    while abs(high_attval - low_attval) > tolerance:
        mid_attval = round((low_attval + high_attval)/2, 3)#Round to 3 decimal places
       
        for att in att_list:
            att.set_att(mid_attval)
        srs.set_output(output=False)
        time.sleep(.1)
        srs.set_output(output=True)
        time.sleep(0.1)  # Allow system to stabilize

        cps_values = [counter.timed_count(counting_time=0.5) / counting_time for _ in range(N)]
        if any(x == 0 for x in cps_values):
            cps_values = [counter.timed_count(counting_time=0.5) / counting_time for _ in range(N)]
        if any(x == 0 for x in cps_values):
            logger.error("Received 0 counts. SNSPD appears latched")
            avg_cps = float('inf')
        else:
            avg_cps = np.mean(cps_values)
        logger.info(f"Attenuation Value: {mid_attval:.3f}, Avg CPS: {avg_cps:.3f}")

        if avg_cps < target_cps:
            high_attval = mid_attval
        else:
            low_attval = mid_attval
    return mid_attval

# def measure_efficiency(now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), IV_pickle_filepath=None, name='', attval='0'):
#     for att in att_list:
#         att.enable()
#         att.set_att(0)

#     sw.set_route(monitor_port)
#     mpm_sw.set_route(ingaas_port)

#     ingaas_pm.set_meter_wav(wavelength_nm = 1550, slot=4)
    
#     full_power=ingaas_pm.get_meter_pow(4)
#     full_pwr_values = [float(ingaas_pm.get_meter_pow(4)) for _ in range(N)]
#     full_power = np.mean(full_pwr_values)
#     measured_att_dB_list = [0 for _ in att_list] 
#     i=0
#     for att in att_list: 
#         att.set_att(attval)
#         time.sleep(1)
#         att_pwr_values = [float(ingaas_pm.get_meter_pow(4)) for _ in range(N)]
#         att_power = np.mean(att_pwr_values)
#         measured_att_dB_list[i]=-10*math.log10(float(att_power)/float(full_power))
#         logger.info(f"Measured Attenuation Value: {measured_att_dB_list[i]:.3f}")
#         att.set_att(0)
#         i=i+1

#     print(np.sum(measured_att_dB_list))
#     all_atten_power=full_power*10**(-np.sum(measured_att_dB_list)/10)
#     number_of_photons = all_atten_power*(wavelength*1e-9)/(1.986454e-25)
#     return target_cps/number_of_photons

    

        
        

  
 

if __name__ == '__main__':
    now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    name = 'saaed2um'

    wavelengths = [635, 1100, 1150, 1250, 1350, 1450, 1566.314, 1525.661, 1620.5, 1650, 1700, 2000]

    
    # taus = [2.75, 2.25]
    # nonlinearity_factor_filepath = nonlinearity_factor_raw_power_measurements(instruments, now_str=now_str, taus=taus)

    # IV_pickle_filepath = SNSPD_IV_Curve(instruments, now_str=now_str, max_cur=max_cur, bias_resistor=bias_resistor, name=name)
    IV_pickle_filepath = os.path.join(current_file_dir, "data_sde", "saaed2um_IV_curve_data__20250307-182549.pkl")

    # trigger_voltage = find_min_trigger_threshold(instruments)
    trigger_voltage = 0.01

    NIST_pm_calib_path = os.path.join(current_file_dir, 'calibration_power_meter', 'SWN HP81521B 2933G05261.xlsx')
    calib_df = pd.read_excel(NIST_pm_calib_path, sheet_name='Data')
    calib_df_wav = calib_df['Wav (nm)'].values

    #light_sources = ['ando_lasers', 'thor_lasers', 'spectrometer']
    light_sources = ['spectrometer']

    # mpm_types = ['InGaAs', 'Thermal']
    mpm_types = ['InGaAs']
    for mpm_type in mpm_types:
        now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
        name_pm = f'{name}_{mpm_type}'

        if mpm_type == 'InGaAs':
            mpm_sw.set_route(ingaas_port)
            instruments['mpm'] = ingaas_pm
        if mpm_type == 'thermal':
            mpm_sw.set_route(thermal_port)

        mpm = instruments['mpm']
        for light_source in light_sources:
            now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
            name_pm_src = f'{name_pm}_{light_source}'

            for wavelength in wavelengths:
                now_str = "{:%Y%m%d-%H%M%S}".format(datetime.now())
                name_pm_src_wav = f'{name_pm_src}_{wavelength}'

                if mpm_type == 'InGaAs':
                    if wavelength < 800 or wavelength > 1700:
                        continue
                    mpm.set_meter_wav(wavelength_nm = wavelength, slot=4)

                if light_source == 'ando_lasers':
                    if wavelength not in [1566.314, 1525.661, 1620.5]:
                        continue
                    elif wavelength==1566.314:
                        laser_sw.set_route(laser1566_port)
                        ando_laser = laser1566
                    elif wavelength==1525.661:
                        laser_sw.set_route(laser1525_port)
                        ando_laser = laser1525
                    elif wavelength==1620.5:
                        laser_sw.set_route(laser1621_port)
                        ando_laser = laser1621
                    instruments['laser'] = ando_laser
                    ando_laser.enable()
                    ando_laser.std_init()

                    # y_n = input("Do you want to calibrate the optical swtich using the nist calibrated power meter? \n")
                    # if y_n in ['1', 'Y', 'y']:
                    #     sw.set_route(monitor_port)
                    #     input("Ensure spliced to CPM not SNSPD\nPress anything to continue\n")
                    #     optical_switch_calibration_filepath = optical_switch_calibration(instruments, now_str=now_str, mpm_type=mpm_type, wavelength=wavelength)
                    #     sw.set_route(monitor_port)
                    #     input("Ensure spliced to SNSPD not CPM\nPress anything to continue\n")

                if light_source == 'thor_lasers':
                    if wavelength not in [635, 2000]:
                        continue
                    elif wavelength==635:
                        laser_sw.set_route(laser635_port)
                        #input=("Please make sure this laser is turned on (No remote connection available)\nPress anything to continue\n")
                    elif wavelength==2000:
                        laser_sw.set_route(laser2000_port)
                        #laser2000.enable()
                   
                if light_source == 'spectrometer':
                    if wavelength < 1000 or wavelength > 2000:
                        continue
                    spectrometer.set_wavelength(wavelength)
                    laser_sw.set_route(spectrometer_port)
                    input=("Please make sure light source is turned on\nPress anything to continue\n")

                
                

                # Find ideal attenuation value (that which gets 300,000 cps at max polarization)
                attval = get_att_value(instruments, IV_pickle_filepath=IV_pickle_filepath, trigger_voltage=trigger_voltage)
                pol_counts_filepath = sweep_polarizations(instruments, now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, attval=attval, name=name_pm_src_wav, num_pols=num_pols, trigger_voltage=trigger_voltage, counting_time=counting_time, N=3)
                # os.path.join(current_file_dir, "data_sde", "saeed2um_pol_data_snspd_splice1__20250115-213240.pkl")
                attval = get_att_value(instruments, IV_pickle_filepath=IV_pickle_filepath, trigger_voltage=trigger_voltage, pol_counts_filepath=pol_counts_filepath)
                # attval = 10

                data_filepath = SDE_Counts_Measurement(instruments, now_str=now_str, IV_pickle_filepath=IV_pickle_filepath, pol_counts_filepath=pol_counts_filepath, attval=attval, name=name_pm_src_wav, trigger_voltage=trigger_voltage,counting_time=counting_time,bias_resistor=bias_resistor)
                attenuator_calibration_filepath = attenuator_calibration(instruments, now_str=now_str, attval=attval, mpm_type=mpm_type)


            
