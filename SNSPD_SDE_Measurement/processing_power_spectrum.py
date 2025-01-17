import os
import pickle
import logging

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

background_path = os.path.join(current_file_dir, "data_ps", "background.dat")
light_path = os.path.join(current_file_dir, "data_ps", "light.dat")
att_path = os.path.join(current_file_dir, "data_ps", "att.dat")

filter1550_path = os.path.join(current_file_dir, "data_ps", "filter1550.dat")
filter1610_path = os.path.join(current_file_dir, "data_ps", "filter1610.dat")
filter1693_path = os.path.join(current_file_dir, "data_ps", "filter1693.dat")
filter1775_path = os.path.join(current_file_dir, "data_ps", "filter1775.dat")
filter1865_path = os.path.join(current_file_dir, "data_ps", "filter1865.dat")
filter1965_path = os.path.join(current_file_dir, "data_ps", "filter1965.dat")
filter2080_path = os.path.join(current_file_dir, "data_ps", "filter2080.dat")

light_path = os.path.join(current_file_dir, "data_ps", "light.dat")
att_path = os.path.join(current_file_dir, "data_ps", "att.dat")

# Initialize an empty dictionary
power_dict = {}

# Open the file and read its contents
with open(file_path, "r") as file:
    for line in file:
        # Split each line into key and value
        key, value = line.split()
        # Add the key-value pair to the dictionary
        power_dict[int(key)] = float(value)

# Print the resulting dictionary
logger.info(f"power_dict: {power_dict}")

photon_dict = {key: value/(codata.h * codata.c / (key*1e-9)) for key, value in power_dict.items()}
logger.info(f"photon_dict: {photon_dict}")

photon_1566 = photon_dict[wavelength_nm]
logger.info(f"photon_1566: {photon_1566}")

photon_dict = {key: value / photon_1566 for key, value in photon_dict.items()}
logger.info(f"scaled photon_dict: {photon_dict}")
