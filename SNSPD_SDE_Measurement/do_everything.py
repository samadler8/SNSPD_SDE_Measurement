Algorithm S1. Nonlinearity factor raw power meaurements
1: 2: 3: 4: 5: 6: 7: 8: 9:
10:
11: 12: 13: 14: 15: 16: 17: 18: 19: 20: 21:
procedureCONSOLE_NONLINEARITY(pm,att1,att2,out) N ← 10
xlist ← [20, 15]
xlist.append(arange(10, 0.9, −0.5)) xlist.append(arange(0.95, 0.5, −0.5)) base_array ← round(10 − 10 ∗ log10(xlist)) base_array ← base_array − min(base_array) att_setting ← {}
for rng ∈ [−10, −20, −30, −40, −50, −60] do
att_setting[rng] ← base_array − (rng + 10) − 3
for rng ∈ [−10, −20, −30, −40, −50, −60] do pm.set_range(rng)
pm.zero()
for α ∈ att_setting[rng] do
att1.set_att(α)
for att_step ∈ [0, 3] do
att2.set_att(att_step) for i ← 1 to N do
power ← pm.get_power() out.write(α, att_step, rng, power)
◃pm≡MPM,att(1,2)≡attenuators,out≡file ◃ Number of reads at each setting
◃ 10 to 0.9 in steps of -0.5 ◃ 0.95 to 0.5 in steps of -0.5
◃ initialize empty dictionary ◃ rng ≡ range setting ◃ 3 is an offset
◃ Set MPM range setting to rng ◃ Zero the MPM
◃ Set attenuation setting on att1 to α ◃ Set attenation setting on att2 to att_step
◃ Iterate N times ◃ Record settings and power readings from MPM into file

Algorithm S2. Attenuator calibration
1:
2: 3: 4: 5:
6: 7: 8: 9:
10: 11: 12: 13: 14:
15: 16: 17: 18: 19: 20: 21: 22: 23:
24: 25:
procedureCONSOLE_CAL(pm,att_list,attval,rngval,out) ◃pm≡MPM,att_list≡attenuators,attval≡attenuator value, rngval ≡ range value, out ≡ file
N = 5
init_rng = −10
for att ∈ att_list do
att.set_att(0)
for att ∈ att_list do
pm.set_range(init_rng)
att_list.disable()
pm.zero()
att_list.enable()
powers ← {}
fori←1toNdo ◃IterateNtimes
powers.append(pm.get_power()) out.write(att_list.get_att(), powers, init_rng)
att.set_att(attval)
pm.set_range(rngval)
att_list.disable()
pm.zero()
att_list.enable()
powers ← {}
fori←1toNdo ◃IterateNtimes
powers.append(pm.get_power())
out.write(att_list.get_att(), powers, rngval) ◃ Write to file
att.set_att(0)
◃ Number of measurements ◃ Initial MPM range setting
◃ Set all attenuators to 0 dB
◃ Disable all attenuators ◃ Zero the MPM ◃ Enable all attenuators
◃ Write to file ◃ Set attenuator att to attval ◃ Change range of MPM ◃ Disable all attenuators ◃ Zero the MPM ◃ Enable all attenuators


Algorithm S3. SDE counts measurement
1: 2: 3: 4: 5: 6: 7: 8: 9:
10: 11: 12: 13: 14: 15: 16: 17: 18: 19: 20: 21:
22: 23: 24: 25: 26: 27: 28: 29: 30: 31: 32: 33: 34: 35: 36: 37:
38: 39: 40: 41: 42: 43: 44: 45: 46:
47: 48: 49: 50:
procedureCONSOLE_DE(device_list,params,out) att_list ← device_list[0, 1, 2]
[pm, sw, pc] ← device_list[3, 4, 5]
[vsrc, counter] ← device_list[6, 7]
◃params≡parameters,out≡files ◃ List of attenuators ◃ pm ≡ MPM, sw ≡ optical switch, pc ≡ polarization controller ◃ vsrc ≡ voltage source, counter ≡ pulse counter
[attval, rngval] ← params[0, 1]
[vstop, vstep, vpol] ← params[2, 3, 4] [out_att, out_maxpol, out_minpol] ← out
N = 10 att_list.set_att(attval) vsrc.set_volt(0)
[out_maxpol, out_minpol].write(’# Dark Counts’) sw.set_route(’monitor_port’)
att_list.disable()
for vval ← 0 to vstop by vstep do
vsrc.set_volt(vval) for i ← 1 to N do
counts = counter.get_counts()
[out_maxpol, out_minpol].write(vval, counts)
vsrc.set_volt(0)
att_list.enable()
sw.set_route(’detector_port’)
vsrc.set_volt(vpol)
maxpol_settings ← maximize_counts(pc, counter, out_maxpol) minpol_settings ← minimize_counts(pc, counter, out_minpol) vsrc.set_volt(0)
out_maxpol.write(’# Maxpol light counts’) pc.set(maxpol_settings)
for vval ← 0 to vstop by vstep do
vsrc.set_volt(vval) for i ← 1 to N do
counts = counter.get_counts() out_maxpol.write(vval, counts)
vsrc.set_volt(0)
out_minpol.write(’# Minpol light counts’) pc.set(minpol_settings)
for vval ← 0 to vstop by vstep do
vsrc.set_volt(vval) for i ← 1 to N do
counts = counter.get_counts() out_minpol.write(vval, counts)
vsrc.set_volt(0) sw.set_route(’monitor_port’)
CONSOLE_CAL(pm, att_list, attval, rngval, out_att) 3. POLARIZATION SENSITIVITY MEASUREMENT
◃ attval ≡ attenuator value, rngval ≡ MPM range value ◃ Voltage bias end-stop, step, and polarization optimization settings ◃ Output files
◃ Number of counter readings ◃ Set attenuators to attval ◃ Set voltage bias to 0 V
◃ Start recording dark counts
◃ Iterate N times ◃ Count for 1 sec., write to files
◃ Set voltage bias for polarization optimization
◃ Start recording ’maxpol’ counts
◃ Iterate N times ◃ Count for 1 sec., write to files
◃ Start recording ’minpol’ counts
◃ Iterate N times ◃ Count for 1 sec., write to files
◃ Attenuator calibration (see algorithm S2)