import numpy as np
import glob
import re
import matplotlib.pyplot as plt

def calculate_switch_correction(filename, correction=None):
    #  Calibrate switch ratio
    wl = re.search('_(.{4})nm', filename).group(1)
    switchdata = np.loadtxt(filename)
    n_avg = np.where(np.diff(switchdata[:, 1]))[0][0]+1
    pm1_power = switchdata[:, -3]
    pmcal_power = switchdata[:, -2]
    #  Reshape array so each row is a measurement
    pm1_power.shape = (-1, n_avg)
    pmcal_power.shape = (-1, n_avg)
    pm1_power = pm1_power.mean(axis=1)
    pmcal_power = pmcal_power.mean(axis=1)
    pm1_power = pm1_power[0::2]
    if correction is not None:
        for i in range(len(pm1_power)):
            pm1_power[i] = correction.power(pm1_power[i], -10)
            #  Assume pmcal_power is ~100uW so it does not need correction
    try:
        ratio = pm1_power/pmcal_power[1::2]
        #  print(ratio)
    except:
        pm1_power = pm1_power[:-1]
        ratio = pm1_power/pmcal_power[1::2]
    switch_correction = ratio.mean()
    return [int(wl), switch_correction]


fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run1\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result1 = np.array([calculate_switch_correction(f) for f in filelist])
result1 = result1[result1[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run2\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result2 = np.array([calculate_switch_correction(f) for f in filelist])
result2 = result2[result2[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run3\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result3 = np.array([calculate_switch_correction(f) for f in filelist])
result3 = result3[result3[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run4\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result4 = np.array([calculate_switch_correction(f) for f in filelist])
result4 = result4[result4[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run5\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result5 = np.array([calculate_switch_correction(f) for f in filelist])
result5 = result5[result5[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run6\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result6 = np.array([calculate_switch_correction(f) for f in filelist])
result6 = result6[result6[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run7\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result7 = np.array([calculate_switch_correction(f) for f in filelist])
result7 = result7[result7[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run8\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result8 = np.array([calculate_switch_correction(f) for f in filelist])
result8 = result8[result8[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run9\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result9 = np.array([calculate_switch_correction(f) for f in filelist])
result9 = result9[result9[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run10\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result10 = np.array([calculate_switch_correction(f) for f in filelist])
result10 = result10[result10[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run12\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result12 = np.array([calculate_switch_correction(f) for f in filelist])
result12 = result12[result12[:, 0].argsort()]
fpath =  'C:\\Users\\qittlab\\Documents\\686DR2\\switch_calib\\neo_run14\\*console_switch_cal*.dat'
filelist = glob.glob(fpath)
result14 = np.array([calculate_switch_correction(f) for f in filelist])
result14 = result14[result14[:, 0].argsort()]


plt.figure(1)
# plt.plot(result1[:,0], result1[:,1], marker='o', label='run1')
# plt.plot(result2[:,0], result2[:,1], marker='o', label='run2')
# plt.plot(result3[:,0], result3[:,1], marker='o', label='run3')
# plt.plot(result4[:,0], result4[:,1], marker='o', label='run4')
# plt.plot(result5[:,0], result5[:,1], marker='o', label='run5')
# plt.plot(result6[:,0], result6[:,1], marker='o', label='run6')
# plt.plot(result7[:,0], result7[:,1], marker='o', label='run7')
# plt.plot(result8[:,0], result8[:,1], marker='o', label='run8')
# plt.plot(result9[:,0], result9[:,1], marker='o', label='run9')
plt.plot(result10[:,0], result10[:,1], marker='o', label='run10')

plt.plot(result12[:,0], result12[:,1], marker='o', label='run12')

plt.plot(result14[:,0], result14[:,1], marker='o', label='run14')
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Switching ratio')
plt.gca().legend(loc='upper left')
plt.show()
