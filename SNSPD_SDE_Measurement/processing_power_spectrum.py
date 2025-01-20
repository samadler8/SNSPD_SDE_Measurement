import os
import pickle
import logging
import math

import numpy as np
import pandas as pd
import scipy.constants as codata

import matplotlib.pyplot as plt

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
data_1610_path = os.path.join(current_file_dir, "data_ps", "saaed2um_silicone_1610_30dB_counts_data__20250116-180627.pkl")
data_1693_path = os.path.join(current_file_dir, "data_ps", "saaed2um_silicone_1693_50dB_counts_data__20250116-181445.pkl")
data_1775_path = os.path.join(current_file_dir, "data_ps", "saaed2um_silicone_1775_50dB_counts_data__20250116-181722.pkl")
data_1865_path = os.path.join(current_file_dir, "data_ps", "saaed2um_silicone_1865_30dB_counts_data__20250116-182319.pkl")
data_1965_path = os.path.join(current_file_dir, "data_ps", "saaed2um_silicone_1965_8dB_counts_data__20250116-182514.pkl")
data_2080_path = os.path.join(current_file_dir, "data_ps", "saaed2um_silicone_2080_0dB_counts_data__20250116-180221.pkl")


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
multimode_2dB_power = open_dat_file(multimode_2dB_path)
background_multimode_2dB_power = open_dat_file(background_multimode_2dB_path)
multimode_30dB_power = open_dat_file(multimode_30dB_path)
background_multimode_30dB_power = open_dat_file(background_multimode_30dB_path)
multimode_40dB_power = open_dat_file(multimode_40dB_path)
background_multimode_40dB_power = open_dat_file(background_multimode_40dB_path)
multimode_50dB_power = open_dat_file(multimode_50dB_path)
background_multimode_50dB_power = open_dat_file(background_multimode_50dB_path)
filter1550shortpass_power = open_dat_file(filter1550shortpass_path)
filter1610_power = open_dat_file(filter1610_path)
filter1693_power = open_dat_file(filter1693_path)
filter1775_power = open_dat_file(filter1775_path)
filter1865_power = open_dat_file(filter1865_path)
filter1965_power = open_dat_file(filter1965_path)
filter2080_power = open_dat_file(filter2080_path)
signal_freespace_power = open_dat_file(signal_freespace_path)
background_freespace_power = open_dat_file(background_freespace_path)

signal_photons = power_to_photons(signal_power)
background_photons = power_to_photons(background_power)
siliconeFilter_photons = power_to_photons(siliconeFilter_power)
background_siliconeFilter_photons = power_to_photons(background_siliconeFilter_power)

multimode_2dB_photons = power_to_photons(multimode_2dB_power)
background_multimode_2dB_photons = power_to_photons(background_multimode_2dB_power)
multimode_30dB_photons = power_to_photons(multimode_30dB_power)
background_multimode_30dB_photons = power_to_photons(background_multimode_30dB_power)
multimode_40dB_photons = power_to_photons(multimode_40dB_power)
background_multimode_40dB_photons = power_to_photons(background_multimode_40dB_power)
multimode_50dB_photons = power_to_photons(multimode_50dB_power)
background_multimode_50dB_photons = power_to_photons(background_multimode_50dB_power)

filter1550shortpass_photons = power_to_photons(filter1550shortpass_power)
filter1610_photons = power_to_photons(filter1610_power)
filter1693_photons = power_to_photons(filter1693_power)
filter1775_photons = power_to_photons(filter1775_power)
filter1865_photons = power_to_photons(filter1865_power)
filter1965_photons = power_to_photons(filter1965_power)
filter2080_photons = power_to_photons(filter2080_power)

background_freespace_photons = power_to_photons(background_freespace_power)
signal_freespace_photons = power_to_photons(signal_freespace_power)

signal_background = subtract_dicts(signal_photons, background_photons)
siliconeFilter_background = subtract_dicts(siliconeFilter_photons, background_siliconeFilter_photons)
multi2dB_background = subtract_dicts(multimode_2dB_photons, background_multimode_2dB_photons)
multi30dB_background = subtract_dicts(multimode_30dB_photons, background_multimode_30dB_photons)
multi40dB_background = subtract_dicts(multimode_40dB_photons, background_multimode_40dB_photons)
multi50dB_background = subtract_dicts(multimode_50dB_photons, background_multimode_50dB_photons)
filter1550shortpass_background = subtract_dicts(filter1550shortpass_photons, background_freespace_photons)
filter1610_background = subtract_dicts(filter1610_photons, background_freespace_photons)
filter1693_background = subtract_dicts(filter1693_photons, background_freespace_photons)
filter1775_background = subtract_dicts(filter1775_photons, background_freespace_photons)
filter1865_background = subtract_dicts(filter1865_photons, background_freespace_photons)
filter1965_background = subtract_dicts(filter1965_photons, background_freespace_photons)
filter2080_background = subtract_dicts(filter2080_photons, background_freespace_photons)
signalfreespace_background = subtract_dicts(signal_freespace_photons, background_freespace_photons)

siliconeFilter_normalization_factor = sum(background_photons.values())/sum(background_siliconeFilter_photons.values())
multi2dB_normalization_factor = sum(background_photons.values())/sum(background_multimode_2dB_photons.values())
multi30dB_normalization_factor = sum(background_photons.values())/sum(background_multimode_30dB_photons.values())
multi40dB_normalization_factor = sum(background_photons.values())/sum(background_multimode_40dB_photons.values())
multi50dB_normalization_factor = sum(background_photons.values())/sum(background_multimode_50dB_photons.values())
filter_normalization_factor = sum(background_photons.values())/sum(background_freespace_photons.values())
signalfreespace_normalization_factor = sum(background_photons.values())/sum(background_freespace_photons.values())

siliconeFilter_normalized = scale_dict(siliconeFilter_background, siliconeFilter_normalization_factor)
multi2dB_normalized = scale_dict(multi2dB_background, multi2dB_normalization_factor)
multi30dB_normalized = scale_dict(multi30dB_background, multi30dB_normalization_factor)
multi40dB_normalized = scale_dict(multi40dB_background, multi40dB_normalization_factor)
multi50dB_normalized = scale_dict(multi50dB_background, multi50dB_normalization_factor)
filter1550shortpass_normalized = scale_dict(filter1550shortpass_background, filter_normalization_factor)
filter1610_normalized = scale_dict(filter1610_background, filter_normalization_factor)
filter1693_normalized = scale_dict(filter1693_background, filter_normalization_factor)
filter1775_normalized = scale_dict(filter1775_background, filter_normalization_factor)
filter1865_normalized = scale_dict(filter1865_background, filter_normalization_factor)
filter1965_normalized = scale_dict(filter1965_background, filter_normalization_factor)
filter2080_normalized = scale_dict(filter2080_background, filter_normalization_factor)
signalfreespace_normalized = scale_dict(signalfreespace_background, signalfreespace_normalization_factor)

siliconeFilter_transmission = get_transmission(signal_background, siliconeFilter_normalized)
freeSpace_transmission = get_transmission(signal_background, signalfreespace_normalized)
multi2dB_transmission = get_transmission(signalfreespace_normalized, multi2dB_normalized)
multi30dB_transmission = get_transmission(signalfreespace_normalized, multi30dB_normalized)
multi40dB_transmission = get_transmission(signalfreespace_normalized, multi40dB_normalized)
multi50dB_transmission = get_transmission(signalfreespace_normalized, multi50dB_normalized)
filter1550shortpass_transmission = get_transmission(signalfreespace_normalized, filter1550shortpass_normalized)
filter1610_transmission = get_transmission(signalfreespace_normalized, filter1610_normalized)
filter1693_transmission = get_transmission(signalfreespace_normalized, filter1693_normalized)
filter1775_transmission = get_transmission(signalfreespace_normalized, filter1775_normalized)
filter1865_transmission = get_transmission(signalfreespace_normalized, filter1865_normalized)
filter1965_transmission = get_transmission(signalfreespace_normalized, filter1965_normalized)
filter2080_transmission = get_transmission(signalfreespace_normalized, filter2080_normalized)

transmission_1550 = multiply_dicts([signal_background, siliconeFilter_transmission, freeSpace_transmission, multi30dB_transmission, filter1550shortpass_transmission])
transmission_1610 = multiply_dicts([signal_background, siliconeFilter_transmission, freeSpace_transmission, multi30dB_transmission, filter1610_transmission])
transmission_1693 = multiply_dicts([signal_background, siliconeFilter_transmission, freeSpace_transmission, multi50dB_transmission, filter1693_transmission])
transmission_1775 = multiply_dicts([signal_background, siliconeFilter_transmission, freeSpace_transmission, multi50dB_transmission, filter1775_transmission])
transmission_1865 = multiply_dicts([signal_background, siliconeFilter_transmission, freeSpace_transmission, multi30dB_transmission, filter1865_transmission])
transmission_1965 = multiply_dicts([signal_background, siliconeFilter_transmission, freeSpace_transmission, multi2dB_transmission, multi2dB_transmission, multi2dB_transmission, multi2dB_transmission, filter1965_transmission])
transmission_2080 = multiply_dicts([signal_background, siliconeFilter_transmission, freeSpace_transmission, filter2080_transmission])
total_photons_1550_unnormalized = sum(transmission_1550.values())
total_photons_1610_unnormalized = sum(transmission_1610.values())
total_photons_1693_unnormalized = sum(transmission_1693.values())
total_photons_1775_unnormalized = sum(transmission_1775.values())
total_photons_1865_unnormalized = sum(transmission_1865.values())
total_photons_1965_unnormalized = sum(transmission_1965.values())
total_photons_2080_unnormalized = sum(transmission_2080.values())

logger.info(f" total_photons_1550_unnormalized: {total_photons_1550_unnormalized}")
logger.info(f" total_photons_1610_unnormalized: {total_photons_1610_unnormalized}")
logger.info(f" total_photons_1693_unnormalized: {total_photons_1693_unnormalized}")
logger.info(f" total_photons_1775_unnormalized: {total_photons_1775_unnormalized}")
logger.info(f" total_photons_1865_unnormalized: {total_photons_1865_unnormalized}")
logger.info(f" total_photons_1965_unnormalized: {total_photons_1965_unnormalized}")
logger.info(f" total_photons_2080_unnormalized: {total_photons_2080_unnormalized}")

def plot_spectrum(transmission_dict, name=''):
    plt.figure(figsize=(8, 5))
    plt.plot(transmission_dict.keys(), transmission_dict.values(), marker='o')  # Line plot with markers
    plt.title(name)
    plt.xlabel("Wavelength")
    plt.ylabel("Unnormalized photons transmitted")
    plt.grid(True)  # Optional, to add a grid
    figname = f"plot_spectrum_{name}.png"
    figdir = os.path.join(current_file_dir, "figs_ps")
    os.makedirs(figdir, exist_ok=True)
    figpath = os.path.join(figdir, figname)
    plt.savefig(figpath)
    return

plot_spectrum(transmission_1550, name='1550')
plot_spectrum(transmission_1610, name='1610')
plot_spectrum(transmission_1693, name='1693')
plot_spectrum(transmission_1775, name='1775')
plot_spectrum(transmission_1865, name='1865')
plot_spectrum(transmission_1965, name='1965')
plot_spectrum(transmission_2080, name='2080')

def plot_efficiency(data_filepath, transmission_dict, name='', photons_1550_nunormalized=1, counts_1550=1, eff_1550=1):
    photons_1550_normalized = counts_1550/eff_1550
    normalization_factor = photons_1550_normalized/photons_1550_nunormalized
    
    with open(data_filepath, 'rb') as file:
        data_dict = pickle.load(file)
    count_array = np.array(data_dict["Count_Array"])
    dark_count_array = np.array(data_dict["Dark_Count_Array"])
    counts = count_array - dark_count_array
    total_photons_unnormalized = sum(transmission_dict.values())
    total_photons_normalized = total_photons_unnormalized * normalization_factor

    eff = counts / total_photons_normalized
    plt.figure(figsize=(20, 10))
    plt.plot(data_dict["Cur_Array"], eff, marker='o')  # Line plot with markers
    plt.title(name)
    plt.xlabel("Current")
    plt.ylabel("Efficiency")
    plt.grid(True)  # Optional, to add a grid
    figname = f"plot_efficiency_{name}.png"
    figdir = os.path.join(current_file_dir, "figs_ps")
    os.makedirs(figdir, exist_ok=True)
    figpath = os.path.join(figdir, figname)
    plt.savefig(figpath)
    return

photons_1550_nunormalized = sum(transmission_1550.values())
counts_1550 = 150000
eff_1550 = 0.15
plot_efficiency(data_1550_path, transmission_1550, name='1550', photons_1550_nunormalized=photons_1550_nunormalized, counts_1550=counts_1550, eff_1550=eff_1550)
plot_efficiency(data_1610_path, transmission_1693, name='1610', photons_1550_nunormalized=photons_1550_nunormalized, counts_1550=counts_1550, eff_1550=eff_1550)
plot_efficiency(data_1693_path, transmission_1693, name='1693', photons_1550_nunormalized=photons_1550_nunormalized, counts_1550=counts_1550, eff_1550=eff_1550)
plot_efficiency(data_1775_path, transmission_1775, name='1775', photons_1550_nunormalized=photons_1550_nunormalized, counts_1550=counts_1550, eff_1550=eff_1550)
plot_efficiency(data_1865_path, transmission_1865, name='1865', photons_1550_nunormalized=photons_1550_nunormalized, counts_1550=counts_1550, eff_1550=eff_1550)
plot_efficiency(data_1965_path, transmission_1965, name='1965', photons_1550_nunormalized=photons_1550_nunormalized, counts_1550=counts_1550, eff_1550=eff_1550)
plot_efficiency(data_2080_path, transmission_2080, name='2080', photons_1550_nunormalized=photons_1550_nunormalized, counts_1550=counts_1550, eff_1550=eff_1550)