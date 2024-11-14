# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:02:38 2024

@author: snk27
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%% load and plot data
file = open(".\\data\\"+file_name+'.pkl', "rb")
data_dict=pickle.load(file)
file.close()
Cur_Array=np.array(data_dict['Cur_Array'])
Volt_Array=np.array(data_dict['Volt_Array'])

plt.plot(Volt_Array,Cur_Array,'-*')
plt.ylabel('Bias current [uA]')
plt.xlabel('Voltage [V]')
plt.savefig(".\\pics\\"+file_name+'.pdf')
# plt.yscale('log')
