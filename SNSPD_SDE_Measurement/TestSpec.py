import subprocess
import time
from amcc.instruments.horiba_ihr320 import HoribaIHR320
from amcc.instruments.agilent_8164a import Agilent8164A

pm=Agilent8164A('GPIB0::23::INSTR')

print(pm.get_meter_pow(4))