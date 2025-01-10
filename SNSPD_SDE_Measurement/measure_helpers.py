import os
import pickle
import time
import logging

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("script_log.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

# SNSPD IV Curve
def SNSPD_IV_Curve(instruments, now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), max_cur=15e-6, bias_resistor=100e3, name=''):
    logger.info("STARTING: SNSPD IV Curve")
    sw = instruments['sw']
    monitor_port = instruments['monitor_port']
    att_list = instruments['att_list']
    srs = instruments['srs']
    multi = instruments['multi']

    sw.set_route(monitor_port)
    for att in att_list:
        att.disable()

    srs.set_voltage(0)
    srs.set_output(output=True)

    # Generate current and voltage arrays
    num_biases = 100
    Cur_Array = np.linspace(0, max_cur, num_biases)
    Cur_Array = np.concatenate([Cur_Array, Cur_Array[::-1][1:], -Cur_Array[1:], -Cur_Array[::-1][1:]])
    Volt_Array = np.round(Cur_Array * bias_resistor, decimals=3)

    # Initialize array for measured voltages
    Volt_Meas = np.empty(Volt_Array.size, dtype=float)

    # Measure voltage for each set bias
    for i, volt in enumerate(Volt_Array):
        srs.set_voltage(volt)
        time.sleep(0.1)  # Wait for stabilization
        Volt_Meas[i] = multi.read_voltage()
        logger.info(f"Applied Voltage: {volt}, Measured Voltage: {Volt_Meas[i]}")
        logger.info(f"{round(100*i/Volt_Meas.size, 2)}%")
    srs.set_voltage(0)
    srs.set_output(output=False)

    # Combine data into a pandas DataFrame
    IV_curve_data = {
        "Current": Cur_Array,
        "Voltage": Volt_Meas
    }
    df = pd.DataFrame(IV_curve_data)

    # Save the DataFrame as a pickle file
    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}_IV_curve_data__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    df.to_pickle(filepath)
    logger.info(f"IV curve data saved to: {filepath}")
    logger.info("COMPLETED: SNSPD IV Curve")
    return filepath

def get_counts(Cur_Array, instruments, trigger_voltage=0.12, bias_resistor=100e3, counting_time=1, N=3):
    """
    Measures counts for a given array of currents.
    """
    # Change this function so that it saves multiple mesaurements at each bias current

    srs = instruments['srs']
    counter = instruments['counter']

    srs.set_voltage(0)
    srs.set_output(output=True)

    counter.basic_setup()
    counter.set_impedance(ohms=50, channel=1)
    counter.setup_timed_count(channel=1)
    counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=1)

    Count_Array = np.empty((len(Cur_Array), N), dtype=float)

    for i in range(len(Cur_Array)):
        this_volt = round(Cur_Array[i] * bias_resistor, 3)
        srs.set_voltage(this_volt)
        time.sleep(0.1)
        temp_cps = np.empty(N, dtype=float)
        for j in np.arange(temp_cps.size):
            temp_cps[j] = counter.timed_count(counting_time=counting_time)/counting_time
        Count_Array[i] = temp_cps
        logger.info(f"Voltage: {this_volt} V, Counts: {np.mean(Count_Array[i])}")
    
    srs.set_voltage(0)
    srs.set_output(output=False)

    return Count_Array

def find_min_trigger_threshold(instruments, now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), max_trigger_voltage=0.2, N=3, channel=1, ohms=50, counting_time=0.5):
    srs = instruments['srs']
    counter = instruments['counter']

    srs.set_voltage(0)
    srs.set_output(output=True)

    counter.basic_setup()
    counter.set_impedance(ohms=ohms, channel=channel)
    counter.setup_timed_count(channel=channel)

    data = []
    # set_trigger_voltage = 0
    # trigger_voltages = np.linspace(0, max_trigger_voltage, 500)
    # for trigger_voltage in trigger_voltages:
    #     counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=channel)
    #     time.sleep(0.1)
    #     temp_cps_arr = np.empty(N, dtype=float)
    #     for l in np.arange(temp_cps_arr.size):
    #         temp_cps_arr[l] = counter.timed_count(counting_time=counting_time)/counting_time
    #     temp_cps = np.mean(temp_cps_arr)
    #     if set_trigger_voltage==0 and temp_cps==0:
    #         temp_cps_arr = np.empty(N, dtype=float)
    #         for l in np.arange(temp_cps_arr.size):
    #             temp_cps_arr[l] = counter.timed_count(counting_time=counting_time)/counting_time
    #         temp_cps = np.mean(temp_cps_arr)
    #     data_temp = (trigger_voltage, temp_cps)
    #     logger.info(data_temp)
    #     data.append(data_temp)
    #     if set_trigger_voltage==0 and temp_cps==0:
    #         set_trigger_voltage = trigger_voltage
    #         break

    def get_avg_cps(trigger_voltage):
        counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=channel)
        time.sleep(0.1)
        cps_arr = np.empty(N, dtype=float)
        for l in np.arange(cps_arr.size):
            cps_arr[l] = counter.timed_count(counting_time=counting_time)/counting_time
        avg_cps = np.mean(cps_arr)
        return avg_cps

    i = 0
    trigger_voltage = max_trigger_voltage/2
    temp_cps = get_avg_cps(trigger_voltage)
    while i < 120:
        if temp_cps > 0:
            result = min((x[0] for x in data if x[1] == 0), default=None)
            if result == None:
                result = max_trigger_voltage
            trigger_voltage = (trigger_voltage + result)/2
        else:
            result = max((x[0] for x in data if x[1] != 0), default=None)
            if result == None:
                result = 0
            trigger_voltage = (trigger_voltage + result)/2
        temp_cps = get_avg_cps(trigger_voltage)
        data_temp = (trigger_voltage, temp_cps)
        logger.info(data_temp)
        data.append(data_temp)

        i += 1

    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f'trigger_voltage_data__{now_str}'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(data, file)
    logger.info(f"trigger voltage data saved to: {filepath}")
    logger.info(f"Final trigger voltage: {trigger_voltage}")
    return trigger_voltage


# Daniel Sorensen Code
# def optimizePolarization(instruments, biasV = 0.7, biasChannel=1, ttChannel=5, tMeasure=0.5, minimize=False, attValue=28):
#     sc = serverConnection()
#     cr = ttCountRate(channel=ttChannel)
#     pc = instruments.pc

#     def optFunc(v, grad=None):
#         pc.setAll(v)
#         counts=cr.measureFor(tMeasure)/tMeasure
#         logger.info(f"P: {v}, Counts: {counts}")
#         return counts
    
#     instruments.att1.set_att(attValue)
#     instruments.att2.set_att(attValue)
#     instruments.att3.set_att(attValue)
#     instruments.laser.enable()
#     instruments.switch.set_route(2)
#     sc.setBias(biasChannel,biasV)

#     opt = nlopt.opt(nlopt.LN_SBPLX,3)
#     if minimize:
#         opt.set_min_objective(optFunc)
#     else:
#         opt.set_max_objective(optFunc)
#     opt.set_lower_bounds(np.ones(3)*-99.)
#     opt.set_upper_bounds(np.ones(3)*99.)
#     opt.set_xtol_abs(np.ones(3))
#     opt.set_initial_step(np.ones(3)*30.)
#     startPol = np.zeros(3)
#     opt.optimize(startPol)

#     optX = pc.getAxis('X')
#     optY = pc.getAxis('Y')
#     optZ = pc.getAxis('Z')
#     optPol = (optX,optY,optZ)
#     if minimize:
#         logger.info("Found minimum polarization:")
#     else:
#         logger.info("Found maximum polarization:")
#     logger.info(optPol)
#     return optPol


# My attempt
# bounds = [(-99, 100)] * 3
# def neg_meas_counts(position, *args):
#     return -meas_counts(position, *args)
# initial_guess = np.array([0, 0, 0])
# res_min = scipy.optimize.minimize(meas_counts, initial_guess, args=(instruments, N, counting_time), bounds=bounds)
# res_max = scipy.optimize.minimize(neg_meas_counts, initial_guess, args=(instruments, N, counting_time), bounds=bounds)
# pol_counts = [(res_min['x'], res_min['fun']), (res_max['x'], res_max['fun'])]
# logging.debug(pol_counts)