import os
import pickle
import logging
import math

import numpy as np
import pandas as pd
import scipy.constants as codata

from pathlib import Path
from datetime import datetime, timedelta 

current_file_dir = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING for less verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("processing_power_spectrum.log", mode="a"),
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

wavelength = 1566e-9
wavelength_nm = wavelength*1e9

filter1550shortpass_path = os.path.join(current_file_dir, "data_ps", "1550filter_shortPass.dat")
filter1550_path = os.path.join(current_file_dir, "data_ps", "1550filter.dat")
filter1610_path = os.path.join(current_file_dir, "data_ps", "1610filter.dat")
filter1693_path = os.path.join(current_file_dir, "data_ps", "1693filter.dat")
filter1775_path = os.path.join(current_file_dir, "data_ps", "1775filter.dat")
filter1865_path = os.path.join(current_file_dir, "data_ps", "1865filter.dat")
filter1965_path = os.path.join(current_file_dir, "data_ps", "1965filter.dat")
filter2080_path = os.path.join(current_file_dir, "data_ps", "2080filter.dat")
shortPass_multimode_path = os.path.join(current_file_dir, "data_ps", "shortPass_multimode.dat")
signal_freespace_path = os.path.join(current_file_dir, "data_ps", "signal_freespace.dat")
background_freespace_path = os.path.join(current_file_dir, "data_ps", "background_freespace.dat")

multimode_2dB_path = os.path.join(current_file_dir, "data_ps", "2dB_multimode.dat")
multimode_30dB_path = os.path.join(current_file_dir, "data_ps", "30dB_multimode.dat")
multimode_40dB_path = os.path.join(current_file_dir, "data_ps", "40dB_multimode.dat")
multimode_50dB_path = os.path.join(current_file_dir, "data_ps", "50dB_multimode.dat")
background_multimode_2dB_path = os.path.join(current_file_dir, "data_ps", "background_2dB_multimode.dat")
background_multimode_30dB_path = os.path.join(current_file_dir, "data_ps", "background_30dB_multimode.dat")
background_multimode_40dB_path = os.path.join(current_file_dir, "data_ps", "background_40dB_multimode.dat")
background_multimode_50dB_path = os.path.join(current_file_dir, "data_ps", "background_50dB_multimode.dat")

siliconeFilter_path = os.path.join(current_file_dir, "data_ps", "siliconeFilter.dat")
signal_singlemode_path = os.path.join(current_file_dir, "data_ps", "signal_singlemode.dat")
background_siliconeFilter_path = os.path.join(current_file_dir, "data_ps", "background_siliconeFilter.dat")
background_singlemode_path = os.path.join(current_file_dir, "data_ps", "background_singlemode.dat")

signal_path = os.path.join(current_file_dir, "data_ps", "signal.dat")
background_path = os.path.join(current_file_dir, "data_ps", "background.dat")



data_1550_path = os.path.join(current_file_dir, "data_ps", "saaed2um_silicone_1550_shortPass_30dB_counts_data__20250116-175627.pkl")


# Initialize an empty dictionary

# Open the file and read its contents
def open_dat_file(filepath):
    power_dict = {}

    with open(filepath, "r") as file:
        for line in file:
            # Split each line into key and value
            key, value = line.split()
            # Add the key-value pair to the dictionary
            power_dict[int(key)] = float(value)

    return power_dict

def power_to_photons(power_dict):
    
    photon_dict = {key: value/(codata.h * codata.c / (key*1e-9)) for key, value in power_dict.items()}
    
    return photon_dict

def subtract_dicts(dict_a, dict_b):
    return {wavelength: dict_a[wavelength] - dict_b[wavelength] 
            for wavelength in dict_a}

def scale_dict(d, scalar):
    return {wavelength: d[wavelength] * scalar 
            for wavelength in d}
    
def get_transmission(photon_dict_i, photon_dict_f):
    return {wavelength: photon_dict_f[wavelength] / photon_dict_i[wavelength] 
            for wavelength in photon_dict_i}

def multiply_dicts(dicts):
    return {
        wavelength: math.prod(d[wavelength] for d in dicts)
        for wavelength in dicts[0]
    }

signal_power = open_dat_file(signal_path)
background_power = open_dat_file(background_path)
background_siliconeFilter_power = open_dat_file(background_siliconeFilter_path)
siliconeFilter_power = open_dat_file(siliconeFilter_path)
multimode_30dB_power = open_dat_file(multimode_30dB_path)
background_multimode_30dB_power = open_dat_file(background_multimode_30dB_path)
filter1550shortpass_power = open_dat_file(filter1550shortpass_path)
filter2080_power = open_dat_file(filter2080_path)
signal_freespace_power = open_dat_file(signal_freespace_path)
background_freespace_power = open_dat_file(background_freespace_path)

signal_photons = power_to_photons(signal_power)
background_photons = power_to_photons(background_power)
siliconeFilter_photons = power_to_photons(siliconeFilter_power)
background_siliconeFilter_photons = power_to_photons(background_siliconeFilter_power)
multimode_30dB_photons = power_to_photons(multimode_30dB_power)
background_multimode_30dB_photons = power_to_photons(background_multimode_30dB_power)
filter1550shortpass_photons = power_to_photons(filter1550shortpass_power)
filter2080_photons = power_to_photons(filter2080_power)
background_freespace_photons = power_to_photons(background_freespace_power)
signal_freespace_photons = power_to_photons(signal_freespace_power)

signal_background = subtract_dicts(signal_photons, background_photons)
siliconeFilter_background = subtract_dicts(siliconeFilter_photons, background_siliconeFilter_photons)
multi30dB_background = subtract_dicts(multimode_30dB_photons, background_multimode_30dB_photons)
filter1550shortpass_background = subtract_dicts(filter1550shortpass_photons, background_freespace_photons)
filter2080_background = subtract_dicts(filter2080_photons, background_freespace_photons)
signalfreespace_background = subtract_dicts(signal_freespace_photons, background_freespace_photons)

siliconeFilter_normalization_factor = sum(background_photons.values())/sum(background_siliconeFilter_photons.values())
multi30dB_normalization_factor = sum(background_photons.values())/sum(background_multimode_30dB_photons.values())
filter1550shortpass_normalization_factor = sum(background_photons.values())/sum(background_freespace_photons.values())
filter2080_normalization_factor = sum(background_photons.values())/sum(background_freespace_photons.values())
signalfreespace_normalization_factor = sum(background_photons.values())/sum(background_freespace_photons.values())

siliconeFilter_normalized = scale_dict(siliconeFilter_background, siliconeFilter_normalization_factor)
multi30dB_normalized = scale_dict(multi30dB_background, multi30dB_normalization_factor)
filter1550shortpass_normalized = scale_dict(filter1550shortpass_background, filter1550shortpass_normalization_factor)
filter2080_normalized = scale_dict(filter2080_background, filter2080_normalization_factor)
signalfreespace_normalized = scale_dict(signalfreespace_background, signalfreespace_normalization_factor)

siliconeFilter_transmission = get_transmission(signal_background, siliconeFilter_normalized)
freeSpace_transmission = get_transmission(signal_background, signalfreespace_normalized)
multi30dB_transmission = get_transmission(signalfreespace_normalized, multi30dB_normalized)
filter1550shortpass_transmission = get_transmission(signalfreespace_normalized, filter1550shortpass_normalized)
filter2080_transmission = get_transmission(signalfreespace_normalized, filter2080_normalized)

transmission_1550 = multiply_dicts([siliconeFilter_transmission, freeSpace_transmission, multi30dB_transmission, filter1550shortpass_transmission])
transmission_2080 = multiply_dicts([siliconeFilter_transmission, freeSpace_transmission, filter2080_transmission])
total_photons_1550_unnormalized = sum(transmission_1550.values())
total_photons_2080_unnormalized = sum(transmission_2080.values())

logger.info(f" total_photons_1550_unnormalized: {total_photons_1550_unnormalized}")
logger.info(f" total_photons_2080_unnormalized: {total_photons_2080_unnormalized}")


# logger.info(f"photon_dict: {photon_dict}")

# photon_1566 = photon_dict[wavelength_nm]
# logger.info(f"photon_1566: {photon_1566}")

# photon_dict = {key: value / photon_1566 for key, value in photon_dict.items()}


# # Print the resulting dictionary
# logger.info(f"power_dict: {power_dict}")

# logger.info(f"scaled photon_dict: {photon_dict}")
