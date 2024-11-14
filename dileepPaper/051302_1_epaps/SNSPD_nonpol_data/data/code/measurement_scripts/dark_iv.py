from __future__ import print_function
# import logging
import datetime
import time
import socket
import os
import sys
import numpy as np
TIMING = False
RUNNING = True
hostname = socket.gethostname()
v_jump = 0.020  # set the vswitch thrshhold


def createfolder(subfolder='', suffix=''):
    d = datetime.datetime.now()  # current date and time for filename
    if len(subfolder) == 0:
        localdatafolder = '~/Documents/%s/' % (hostname) + \
                          d.strftime('%Y_%m_%d')
    else:
        localdatafolder = '~/Documents/%s/%s/' % (subfolder, hostname) + \
                d.strftime('%Y_%m_%d')
    # expand ~ in OS dependent way
    localdatafolder = os.path.expanduser(localdatafolder)
    localdatafolder += suffix
    if not localdatafolder.endswith('/'):
        localdatafolder += '/'
    #  Make the folder if it does not exist
    if not os.path.isdir(localdatafolder):
        print('Creating: %s' % (localdatafolder))
        os.makedirs(localdatafolder)
    return localdatafolder


def createHMname():
    d = datetime.datetime.now()
    return d.strftime('%H_%M.dat')


def createfilename(subfolder=''):
    d = datetime.datetime.now()  # current date and time for filename
    if len(subfolder) == 0:
        localdatafolder = '~/Documents/%s/' % (hostname)+d.strftime('%Y_%m_%d')
    else:
        localdatafolder = '~/Documents/%s/%s/' % (subfolder,
                                                  hostname) + \
                          d.strftime('%Y_%m_%d')
    localdatafolder = os.path.expanduser(localdatafolder)
    if not os.path.isdir(localdatafolder):
        print('Creating: %s' % (localdatafolder))
        os.makedirs(localdatafolder)
    fileName = localdatafolder + '/'+d.strftime('%H_%M')+'.dat'
    return fileName


class out2file():
    def __init__(self, filename):
        self.filename = filename
        self.fp = open(filename, 'w')
        self.writeln('# Time: %s' % str(datetime.datetime.now()))
        self.writeln('# Computer: %s' % (socket.gethostname()))
        self.writeln('# Program commandline: '+repr(sys.argv))
        self.writeln('# cwd: '+os.path.dirname(os.path.realpath(__file__)))
        self.writeln('#')

    def writeln(self, msgout):
        self.fp.write(msgout+'\n')
        print('writeln: ', msgout)
        self.fp.flush()

    def write(self, msgout):
        self.fp.write(msgout)
        print('write: ', msgout)
        self.fp.flush()

    def close(self):
        self.fp.close()


def get_dmm_counter(dmm, counter):
    global TIMING
    #start = time.time()
    if counter.__module__ == 'hp53131':
        counter.meter.write('READ?')
    else:
        counter.meter.write('CR;CS')
    start = time.time()
    if dmm.__module__ == 'ag34401A':
        dmm.meter.write('READ?')
        time.sleep(0.12)
        msg = dmm.meter.read(100).strip()
        try:
            v_dmm = float(msg)
        except:
            print('problem conversion: ', repr(msg))
            v_dmm = float('NaN')
    else:
        v_dmm = dmm.get_volt()

    dt = time.time()-start - 1
    if TIMING:
        print('dt: %.2f' % dt)
    if dt < -0.1:
        time.sleep(-dt)
    if counter.__module__ == 'sr400':
        time.sleep(0.2)
        while True:
            # rsp = self.meter.query('SS')
            # print(repr(rsp))
            rsp = counter.meter.rsp()
            if int(rsp) & 6 == 6:
                break
        if counter.channel == 0:
            counter.meter.write('QA')
        else:
            counter.meter.write('QB')
        time.sleep(0.2)
    if TIMING:
        print('%.2f  %.2f' % (time.time(), time.time()-start))
    #msg = counter.meter.read(100).strip()
    msg = counter.meter.readline().strip()
    # print('counter: ',repr(msg))
    if TIMING:
        print('%.2f  %.2f' % (time.time(), time.time()-start))
    try:
        counts = float(msg)
    except:
        print('problem with converting count: %s' % repr(msg))
        counts = -1
    return v_dmm, counts


def reset():
    global RUNNING
    RUNNING = True


def stop():
    global RUNNING
    RUNNING = False


def case10(out,k213,dmm,counter,vbias,vstep,vstop, daq_loop,update=None ):
        global RUNNING
        global v_jump
        case = 10
        out.writeln('#'*10)
        out.writeln('# scan bias')
        out.writeln('# case: %d'%case)
        out.writeln('#'*10)
        out.writeln('# Voltage source')
        k213.writeconfig(out.fp)
        out.writeln('#'*10)
        out.writeln('# Voltage meter')
        dmm.writeconfig(out.fp)
        out.writeln('#'*10)
        out.writeln('# Counter')
        counter.writeconfig(out.fp)
        """
        k213.writeconfig(out.fp)
        out.writeln('#')
        dmm.writeconfig(out.fp)
        out.writeln('#')
        counter.writeconfig(out.fp)
        out.writeln('#')
        """
        out.writeln('#####')
        out.writeln('#  time, vbias, vdmm, thresh, counts')
        # msgin = counter.get_threshold()
        # print repr(msgin)
        try:
            thresh = float(counter.get_threshold())
        except:
            thresh = -1
        while vbias<vstop and RUNNING:
            k213.set_volt(vbias)
            #v_set = k213.get_volt()
            v_set = vbias
            for loop in range(daq_loop):
                v_dmm,counts = get_dmm_counter(dmm, counter)
                msgout =  '%.2f %.3f %.3e %.3f %10d'%(time.time(), v_set, v_dmm, thresh, counts)
                #msgout =  '%.2f %.3f  %.3e %10d'%(time.time(), v_set, v_dmm, counts)
                out.writeln( msgout)
                if update is not None:
                    update(msgout)
            if abs(v_dmm) > v_jump:
                break;
            vbias =  vbias + vstep
        return vbias
def case90(out,k213,dmm,counter,vbias,vstep,vstop,attval,daq_loop,update=None):
        global RUNNING
        global v_jump
        case = 90
        """
        k213.writeconfig(out.fp)
        out.writeln('#')
        dmm.writeconfig(out.fp)
        out.writeln('#')
        counter.writeconfig(out.fp)
        out.writeln('#')
        """
        out.writeln('##### Att3: %f' % attval)
        out.writeln('#  time, vbias, vdmm, att3, counts')

        while vbias<vstop and RUNNING:
            k213.set_volt(vbias)
            #v_set = k213.get_volt()
            v_set = vbias
            for loop in range(daq_loop):
                v_dmm,counts = get_dmm_counter(dmm, counter)
                msgout =  '%.2f %.3f %.3e %.1f %10d'%(time.time(), v_set, v_dmm, attval, counts)
                #msgout =  '%.2f %.3f  %.3e %10d'%(time.time(), v_set, v_dmm, counts)
                out.writeln( msgout)
                if update is not None:
                    update(msgout)
            if abs(v_dmm) > v_jump:
                break;
            vbias =  vbias + vstep
        return vbias

def case40(out,k213,dmm,vbias,vstep,vstop, daq_loop ):
    case = 40
    out.writeln('# iv only')
    out.writeln('# case: %d'%case)
    k213.writeconfig(out.fp)
    out.writeln('#')
    dmm.writeconfig(out.fp)
    out.writeln('#')

    while abs(vbias)<abs(vstop):
        k213.set_volt(vbias)
        #v_set = k213.get_volt()
        v_set = vbias
        for loop in range(daq_loop):
            v_dmm =dmm.get_volt()
            msgout =  '%.2f %.3f %.3e'%(time.time(), v_set, v_dmm)
            out.writeln( msgout)
        time.sleep(0.2)
        if abs(v_dmm) > 0.1:
            break;
        vbias =  vbias + vstep

if __name__ == '__main__':
    fname = createfilename()
    print(fname)
    #logging.basicConfig(filename=fname,format='%(message)s',level=logging.WARNING)
    out = out2file(fname)
    from light import *
    k213 = vsrc
    vbias = 0.0
    vstep = 0.01
    daq_loop = 1
    #daq_loop = 18  # change because dwell because of MLS daq problems
    case = 20 # sweep threshold
    case = 10 # sweep bias
    #case = 40 # iv_only
    if case ==1 :
        while True:
            out.writeln('# scan bias')
            k213.set_volt(vbias)
            v_set = k213.get_volt()
            #thresh = float(counter.get_threshold())
            v_dmm = dmm.get_volt()
            for loop in range(daq_loop):
                counts = counter.get_tot_count()
                msgout =  '%.2f %.3f %.3e %.3f %10d'%(time.time(), v_set, v_dmm, thresh, counts)
                #msgout =  '%.2f %.3f  %.3e %10d'%(time.time(), v_set, v_dmm, counts)
                out.writeln( msgout)
            if abs(v_dmm) > 0.01:  # 0.1
                break;
            vbias =  vbias + vstep
    elif case ==10 :
        vstop=10
        case10(out,k213,dmm,counter,vbias,vstep,vstop,daq_loop)

    elif case==2:
        out.writeln('# scan threshold')
        vbias = 0.3
        k213.set_volt(vbias)
        for thresh in np.arange(0,0.2,0.005):
            v_dmm = dmm.get_volt()
            counter.set_threshold(thresh)
            counts = counter.get_tot_count()
            v_set = k213.get_volt()
            msgout =  '%.2f %.3f %.3e %.3f %10d'%(time.time(), v_set, v_dmm, thresh, counts)
            out.warning(msgout)
    elif case==20:
        out.writeln('# scan threshold')
        vbias = 0.25
        k213.set_volt(vbias)
        v_set = vbias
        for thresh in np.arange(0,1,0.005):
            counter.set_threshold(thresh)
            v_dmm,counts = get_dmm_counter(dmm, counter)
            msgout =  '%.2f %.3f %.3e %.3f %10d'%(time.time(), v_set, v_dmm, thresh, counts)
            #msgout =  '%.2f %.3f  %.3e %10d'%(time.time(), v_set, v_dmm, counts)
            out.writeln( msgout)
            if counts==0 and v_dmm<0.1:
                break
            if abs(v_dmm) > 0.1:
                k213.set_volt(0)
                time.sleep(1)
                k213.set_volt(vbias)

    elif case==3:  # log dark
        out.writeln('# take dark / shutter close')
        v_set = k213.get_volt()
        thresh = float(counter.get_threshold())
        while True:
            v_dmm = dmm.get_volt()
            counts = counter.get_tot_count()
            msgout =  '%.2f %.3f %.3e %.3f %10d'%(time.time(), v_set, v_dmm, thresh, counts)
            out.writeln(msgout)
    elif case==31:
        out.writeln('#  take_data indefinitely')
        out.writeln('# case: %d'%case)
        k213.writeconfig(out.fp)
        out.writeln('#')
        dmm.writeconfig(out.fp)
        out.writeln('#')
        counter.writeconfig(out.fp)
        out.writeln('#')

        vbias = 0.3
        k213.set_volt(vbias)
        v_set = k213.get_volt()
        thresh = float(counter.get_threshold())
        while True:
            for loop in range(daq_loop):
                v_dmm,counts = get_dmm_counter(dmm, counter)
                msgout =  '%.2f %.3f %.3e %.3f %10d'%(time.time(), v_set, v_dmm, thresh, counts)
                out.writeln( msgout)
            if abs(v_dmm) > 0.1:
                print('reset bias')
                k213.set_volt(0)
                time.sleep(1)
                k213.set_volt(vbias)
                time.sleep(1)
    elif case==40:
        vstop=10
        case40(out,k213,dmm,vbias,vstep,vstop,daq_loop )

    k213.set_volt(0)
    dmm.meter.loc()
    k213.meter.loc()
    #counter.set_continuous()
    #counter.close()
