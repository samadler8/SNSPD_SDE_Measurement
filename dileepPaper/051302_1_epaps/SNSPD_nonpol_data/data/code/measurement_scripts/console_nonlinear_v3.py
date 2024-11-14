# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\saewoo\.spyder2\.temp.py

"""
from __future__ import print_function
import os, sys, time, socket, datetime
import numpy as np
import threading
import time
import math
from dark_iv import createfilename, out2file
import argparse
import set_laser_latl_v3

parser = argparse.ArgumentParser()
#parser.add_argument("-d", type=int, choices = [3,5,6,8], help='detector choice')
#parser.add_argument("-r", type = float,default = 100e3, help="photon flux")
parser.add_argument("-t", action="store_true" , help="test, no zero")
parser.add_argument("-wl", type = int , help="wavelength in nm",default=1550)
parser.add_argument("-pm2auto", type=bool, help="set auto range pm2", default=False)
parser.add_argument("-initial_rng", type=int, help="initial_rng", default=-10)
args = parser.parse_args()
print(args)
from light import *
def main(args):
    global laser,pm1,pm2,  att1,att2,att3, switch
    #pm = pm2
    fname = createfilename()
    print(fname)
    fname = fname.strip('.dat')+'_%s_%dnm.dat'%((sys.argv[0].strip('.py')).strip('C:\\Python\\de_v3\\'),args.wl)
    print(fname)
    out = out2file(fname)
    #devices = [laser, pm, att1, att2, att0, switch]
    devices = [laser, pm1, att1, att2, att3, switch]
    devices2 = [pm1, att1, att2, att3, switch]
    pm_route = 1
    switch.route(pm_route) # go to internal power meter, sw:1 pm, sw:2 nano detector
    pm1.set_unit(1)
    # pm2.set_unit(1)
    for pm in [pm1]:  # [pm1, pm2]:
        if 'ag' in pm.__module__:
            pm.set_atim(1)
        elif 'aq' in pm.__module__:
            pm.set_atim(10)

    # if args.pm2auto:
    #     pm2.set_range('A')
    # else:
    #     pm2.set_range(0)
    #att1, att2 = att2, att1  # swap attenuator functions
    print('BEFORE', att1.identify(),att2.identify(),att3.identify())
    att3, att1, att2 = att1, att2, att3 #attenuator one always set to 0, att2 does all the work and att3 is always 0 or 3
    print('AFTER', att1.identify(), att2.identify(), att3.identify())
    # pm1, pm2 = pm2, pm1
    # pm2 = att3
    print('pm2', pm2.get_power())
    print('setting wavelengths to: %.3f'%args.wl)
    for inst in devices2:
        if hasattr(inst, 'set_lambda'):
            inst.set_lambda(args.wl)
    laser.set_lambda(args.wl+0.5) # The AQ8201-13 wavelength has shifted
    set_laser_latl_v3.find_latl(pm1, 100e-6)
    print('Dwelling at wavelength for 5 mins ...')
    time.sleep(300)

    if args.t:
        NOZERO=True
        wait_for_att = 0
        out.writeln('# no zero, testing')
    else:
        NOZERO = False
        wait_for_att = 3

    if 'ag' in pm1.__module__:
        MINRNG = -60
        overlap_stepsize = 1.0;
    elif 'aq' in pm1.__module__:
        MINRNG = -60
        overlap_stepsize = 0.5;

    if '81642a' in laser.__module__:
        slot = 0
    else:
        pass

    alpha = 0
    att1.set_att(alpha)
    att2.set_att(0)
    att3.set_att(0)
    print('att3 set to 0')
    att1.enable()
    att2.enable()
    att3.enable()
    rng = args.initial_rng  # 0 or  -10
    if rng != pm1.get_range():
        print('Setting range',rng)
        pm1.set_range(rng)
        time.sleep(3)
    initial_power = pm1.get_power()
    timestimeout = 0
    while(math.isnan(initial_power) and timestimeout < 3):
        print('Read NaN for power. Retrying ...')
        time.sleep(3)
        initial_power = pm1.get_power()
        timestimeout = timestimeout+1
    initial_rng = rng

    initial_target = 10.**(rng/10) * 2e-3
    start_offset = 0
    if initial_target>initial_power:
        initial_target = 10**(rng/10) * 1e-3
        start_offset = np.around( 10. * np.log10(1./2), decimals=2)
        # start_power = 2e-3;
        # start_offset = np.around(10.*np.log10(start_power / 2e-3), decimals=2)

    #set_laser_latl_v3.find_latl(pm1, target = initial_target)

    #set_laser_pwr.find_latl(pm1, target = initial_target)

    #  Should check range here and whether power is 1x range or 2xrange
    #
    #  This program assumes nearly 1mW or 2mW power is starting point
    #  step through roughly linearly spaced

    out.writeln('# Time: %s'% str(datetime.datetime.now()))
    out.writeln('# Computer: %s'%(socket.gethostname()))
    out.writeln('# wait for att: %d'%wait_for_att)
    out.writeln('# '+repr(sys.argv))
    out.writeln('#')
    out.writeln('# Laser ')
    laser.writeconfig(out.fp)
    out.writeln('#')
    out.writeln('# Power meter')
    out.writeln('# monitor power meter on attenuator')
    pm2.writeconfig(out.fp)
    out.writeln('#')
    out.writeln('# Power meter')
    pm1.writeconfig(out.fp)
    out.writeln('#')
    out.writeln('# power meter on output: %d'%pm_route)
    out.writeln('#')
    out.writeln('# Switch')
    switch.writeconfig(out.fp)
    out.writeln('#')
    out.writeln('# Att 1')
    att1.writeconfig(out.fp)
    out.writeln('#')
    out.writeln('# Att 2')
    att2.writeconfig(out.fp)
    out.writeln('#')
    out.writeln('# Att 3')
    att3.writeconfig(out.fp)
    out.writeln('#')
    out.writeln('#')
    out.writeln('# time, route, powermonitor, att1, att2, range, pow')
    #out.writeln('# Att power control')
    #att0.writeconfig(out.fp)

    N = 10

    pm_list = [pm1, pm2]
    pm_zero_list = [pm1]
    def zero():
        for att in att_list:
            print('disable')
            att.disable()
        for pm in pm_list:
            print('init power meeting',pm)
            pm.init_pwm_log(N)
            # don't zero if testing... this takes too much time
        if not NOZERO:
            for pm in pm_zero_list:
                print('zeroing', pm)
                pm.zero()
        for att in att_list:
            print('enable')
            att.enable()

    def wait():
        #print 'done waiting'
        return
    def get_powers(out,switch,rng,att1val, att2val):
        initial_power = pm1.get_power()
        timestimeout = 0
        while(math.isnan(initial_power) and timestimeout < 3):
            print('Read NaN for power. Retrying ...')
            time.sleep(3)
            initial_power = pm1.get_power()
            timestimeout = timestimeout+1
        start=time.time()
        if 'ag' in pm1.__module__:  # start pwm log as we get data from others
            print('start agilent log')
            pm1.start_pwm_log()
            # wait for data
            # w = threading.Timer(N,wait)
            # w.start()
        switch_out = switch.get_route()
        #att0_power = att0.get_power()
        att0_power = 0
        #print 'before att1 dt: %.2f'%(time.time()-start)
        att1_val = att1.get_att()
        # att1_val = att1val
        #print 'att1 dt: %.2f'%(time.time()-start)
        att2_val = att2.get_att()
        # att2_val = att2val
        #print 'att2 dt: %.2f'%(time.time()-start)

        data = [np.arange(N)+start]
        data.append(np.ones(N)*switch_out)
        data.append(np.ones(N)*att0_power)
        data.append(np.ones(N)*att1_val)
        data.append(np.ones(N)*att2_val)
        data.append(np.ones(N)*rng)
        #print len(data)
        #print 'append dt: %.2f'%(time.time()-start)
        if 'ag' in pm1.__module__:
            pass
        else:
            #print 'start logging'
            for pm in pm_list:
                pm.start_pwm_log()
            # wait for data
            #w = threading.Timer(N,wait)
            #w.start()
        pm2.start_pwm_log()
        #w.join()
        #print 'join dt: %.2f'%(time.time()-start)
        power = []
#        for pm in pm_list:
#            d = pm.read_pwm_log()
#            power.append(  d.mean())
#            #print data
#            #print 'dt: %.2f'%(time.time()-start)
#            #print d
#            data.append(d)

        d = pm1.read_pwm_log()
        power.append(  d.mean())
        #print data
        #print 'dt: %.2f'%(time.time()-start)
        #print d
        data.append(d)
        d = pm2.read_pwm_log()
        data[2] = d[0:N]
        data = np.array(data).T
#        data[:,2]=data[:,-1]
#        data = np.delete(data,-1,1)

        #print(data.shape)
        #print(data)
        #print('%.2f\t%5.0f\t%.8e'+'\t%7.2f'*2+'\t%5.0f'+1*'\t%.8e' % data)

        np.savetxt(out.fp, data,
                fmt='%.2f\t%5.0f\t%.8e'+'\t%7.2f'*2+'\t%5.0f'+1*'\t%.8e')
        out.fp.flush()
        print('dt: %.2f'%(time.time()-start))

        return power
    #  pre-calculate all the attenuator settings for each range
    att_setting={}
    if False:  #AQ820121:
        xlist = np.array([])
    else:
        xlist = np.array([20,15])
    xlist = np.append(xlist,np.arange(10,0.9,-0.5))
    xlist = np.append(xlist,np.arange(0.95,0.5,-0.05))


    base_att = np.around(10-10*np.log10(xlist),decimals=2)
    base_att = base_att - min(base_att)
    print('base_att',base_att)
    for rng in np.arange(initial_rng,initial_rng-60,-10):
        att_setting[rng]=base_att - (rng-initial_rng) + start_offset
        att_setting[rng]=att_setting[rng][att_setting[rng]<=60]
        att_setting[rng]=att_setting[rng][att_setting[rng]>=0]
    for rng in np.arange(initial_rng,initial_rng-60,-10): # step through ranges, max set by maxatt
        pm1.set_range(rng)
        readrng = pm1.get_range()
        timestimeout = 0
        while(math.isnan(readrng) and timestimeout < 3):
            print('Read NaN for range. Retrying ...')
            pm1.set_range(rng)
            time.sleep(3)
            readrng = pm1.get_range()
            timestimeout = timestimeout+1
        zero()
        for alpha in att_setting[rng]: #  measure a range, use a
            print('rng: %d  alpha: %.2f'%(rng, alpha))
            att1.set_att(alpha)
            power={}
            for att_step in [0, 3]:
                att2.set_att(att_step)
                time.sleep(wait_for_att)
                power[att_step] = get_powers(out, switch, readrng,alpha,att_step)

            out.writeln('#')
            print('power:',power)
        #pm.stop_pwm_log()
    out.writeln('# read 0 dB power again to look at drift')
    rng = initial_rng
    pm1.set_range(rng)
    # pm2.set_range(rng)
    readrng = pm1.get_range()
    att1.set_att(0)
    att2.set_att(0)
    att3.set_att(0)
    alpha = 0
    zero()
    att_step = -1
    time.sleep(wait_for_att)
    power[att_step] = get_powers(out, switch, readrng,alpha,att_step)

    out.writeln('#')

    for device in devices:
        if hasattr(device,'close'):
            device.close()
        else:
            print(device,' has not close method')

if __name__ == '__main__':
    #wl_list = [1537]
    #wl_list.extend(range(1510,1640,10))
    #wl_list = [1470]
    #wl_list.extend(range(1470,1581,10))
    #wl_list = range(1590,1621,10)
    #wl_list = [1566.314]
    #wl_list = [1546.95]
    # wl_list = [ 1490, 1500, 1530, 1550]
    #wl_list = [1525, 1549.6]
    #wl_list = range(1525, 1585, 5)
    ##wl_list = [1561, 1497]
    #wl_list = [1550, 1550, 1550, 1550]
    #wl_list.extend([1549.6])
    #wl_list = [1549.72]
    #wl_list = [1550]
    wl_list = [args.wl]
    #wl_list = [1495, 1497, 1553, 1568]
    # wl_list = [1555, 1555]
    #wl_list = range(1575,1469,-1)
    #wl_list = [1575, 1573, 1572, 1571, 1569, 1567, 1564, 1561, 1558, 1529, 1526,
    #1525, 1523, 1519, 1515, 1514, 1513, 1508, 1500, 1498, 1496, 1492, 1491,
    #1485, 1484, 1478]
    #wl_list = range(1549,1524,-1)
    #wl_list.extend(range(1575,1549,-1))
    #wl_list = range(1470,1576,1)
    # print('wl_list', wl_list)
    #wl_list = [1540]
    for wl in wl_list:
        args.wl = wl
        main(args)
    # main(args)
    #devices2 = [pm1, att1, att2, att3, switch]
    #for inst in devices2:
    #    if hasattr(inst, 'set_lambda'):
    #        inst.set_lambda(args.wl+5)
    #laser.set_lambda(args.wl+5.5) # The AQ8201-13 wavelength has shifted
    #set_laser_latl_v3.find_latl(pmcal, 100e-6)
    """
    for count in range(5):
      main(args)
    """
