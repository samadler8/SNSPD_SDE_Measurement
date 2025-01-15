import os
import pickle
import time
import logging
import json

import numpy as np

from pathlib import Path
from datetime import datetime

current_file_dir = Path(__file__).parent
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
    Volt_Meas_Array = np.empty(Cur_Array.shape, dtype=float)

    total_num_biases = Cur_Array.size

    # Measure voltage for each set bias
    for i in range(total_num_biases):
        set_volt = Volt_Array[i]
        srs.set_voltage(set_volt)
        time.sleep(0.1)  # Wait for stabilization
        Volt_Meas = multi.read_voltage()
        logger.info(f"Applied Current: {Cur_Array[i]}, Voltage: {set_volt}; Measured Voltage: {Volt_Meas}")
        Volt_Meas_Array[i] = Volt_Meas
        logger.info(f"{round(100*i/total_num_biases, 2)}%")
    srs.set_voltage(0)
    srs.set_output(output=False)    

    data_dict = {
        'Cur_Array': Cur_Array,
        'Volt_Meas_Array': Volt_Meas_Array, 
    }

    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}_IV_curve_data__{now_str}.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(data_dict, file)

    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    json_filepath = f'{os.path.splitext(filepath)[0]}.json'
    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

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

    Volt_Array = np.round(Cur_Array * bias_resistor, decimals=3)
    Count_Array = np.empty((len(Cur_Array), N), dtype=float)

    for i in range(len(Cur_Array)):
        srs.set_voltage(Volt_Array[i])
        time.sleep(0.1)
        temp_cps = np.empty(N, dtype=float)
        for j in np.arange(temp_cps.size):
            temp_cps[j] = counter.timed_count(counting_time=counting_time)/counting_time
        Count_Array[i] = temp_cps
        logger.info(f"Voltage: {Volt_Array[i]} V, Counts: {np.mean(temp_cps)}")
    
    srs.set_voltage(0)
    srs.set_output(output=False)

    return Count_Array

def find_min_trigger_threshold(
    instruments, 
    now_str="{:%Y%m%d-%H%M%S}".format(datetime.now()), 
    max_trigger_voltage=0.2, 
    N=7, 
    channel=1, 
    ohms=50, 
    counting_time=0.5
):
    """
    Finds the minimum trigger threshold voltage for the counter using binary search.
    
    Parameters:
        instruments (dict): Dictionary containing instrument instances ('srs' and 'counter').
        now_str (str): Timestamp for saving data files.
        max_trigger_voltage (float): Maximum voltage to test for triggering.
        N (int): Number of measurements to average for each voltage.
        channel (int): Counter channel to use.
        ohms (int): Impedance value in ohms.
        counting_time (float): Measurement time for each count.
    
    Returns:
        float: Final calculated trigger voltage.
    """
    srs = instruments['srs']
    counter = instruments['counter']

    # Initialize SRS and counter
    srs.set_voltage(0)
    srs.set_output(output=True)

    counter.basic_setup()
    counter.set_impedance(ohms=ohms, channel=channel)
    counter.setup_timed_count(channel=channel)    

    def measure_avg_cps(trigger_voltage):
        """Measure average counts per second (CPS) at a given trigger voltage."""
        counter.set_trigger(trigger_voltage=trigger_voltage, slope_positive=True, channel=channel)
        time.sleep(0.1)  # Allow system to stabilize
        cps_values = [counter.timed_count(counting_time=counting_time) / counting_time for _ in range(N)]
        avg_cps = np.mean(cps_values)
        return avg_cps

    # Binary search for trigger voltage
    low_voltage = 0
    high_voltage = max_trigger_voltage
    tolerance = 0.01

    data_dict = {}
    while abs(high_voltage - low_voltage) > tolerance:
        mid_voltage = (low_voltage + high_voltage) / 2
        avg_cps = measure_avg_cps(mid_voltage)
        
        data_dict[mid_voltage] = avg_cps
        logger.info(f"Voltage: {mid_voltage:.3f}, Avg CPS: {avg_cps:.3f}")

        if avg_cps == 0:
            high_voltage = mid_voltage
        elif avg_cps < 1:
            low_voltage = (mid_voltage + low_voltage) / 2
        else:
            low_voltage = mid_voltage

    # Apply a safety margin to the final trigger voltage
    final_trigger_voltage = high_voltage * 1.1

    # Save measurement data
    output_dir = os.path.join(current_file_dir, 'data_sde')
    os.makedirs(output_dir, exist_ok=True)
    filename = f'trigger_voltage_data__{now_str}.pkl'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(data_dict, file)

    readable_output_dir = os.path.join(current_file_dir, 'readable_data_sde')
    os.makedirs(readable_output_dir, exist_ok=True)
    json_filepath = f'{os.path.splitext(filepath)[0]}.json'
    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    logger.info(f"Trigger voltage data saved to: {filepath}")
    logger.info(f"Final trigger voltage: {final_trigger_voltage:.3f}")
    return final_trigger_voltage


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
# logger.info(pol_counts)