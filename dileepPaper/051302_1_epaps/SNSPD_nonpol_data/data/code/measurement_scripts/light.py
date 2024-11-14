import time
from load_instr_v2 import create_instances

instruments = create_instances('instruments_config.yaml')
laser = instruments['laser2']['dev']
#laser0 = instruments['laser0']['dev']
#laser2 = instruments['laser2']['dev']
pc = instruments['pc']['dev']
pm1 = instruments['pm1']['dev']
pm2 = instruments['pm2']['dev']
pmcal = instruments['pmcal']['dev']
switch = instruments['switch']['dev']
sw12 = instruments['sw12']['dev']
att1 = instruments['att1']['dev']
att2 = instruments['att2']['dev']
att3 = instruments['att3']['dev']
dmm = instruments['dmm']['dev']
vsrc = instruments['vsrc']['dev']
counter = instruments['counter']['dev']

att_list = [att1, att2, att3]
#laser_list = [laser,laser2]
devices2 = [pm1, pmcal, att1, att2, att3, switch]
devices = [laser, pm1, pmcal, att1, att2, att3, switch]

class nodev:
    def __init__(self, devname='switch'):
        self.devname = devname

    def route(self, *args):
        print('no %s' % (self.devname))

    def get_route(self, *args):
        return 0

    def writeconfig(self, fp):
        fp.write('# no %s' % self.devname)

# switch=nodev()
# laser = nodev('laser')

def find_switching_ratio(num):
     switch.set_route(1);
     time.sleep(1);
     pm1_avg = 0;
     print("");
     for i in range(0,num):
         pm1_read = pm1.get_power();
         print("Route 1 trial %d of %d: %f uW" % (i+1, num, pm1_read*1e6));
         pm1_avg = pm1_avg + pm1_read;
         time.sleep(1);
     if(num > 1):
         pm1_avg = pm1_avg/(1.0*num);
         print("Route 1 average: %f uW\n" % (pm1_avg*1e6));
     switch.set_route(2);
     time.sleep(1);
     pmcal_avg = 0;
     for i in range(0,num):
         pmcal_read = pmcal.get_pow(2);
         print("Route 2 trial %d of %d: %f uW" % (i+1, num, pmcal_read*1e6));
         pmcal_avg = pmcal_avg + pmcal_read;
         time.sleep(1);
     if(num > 1):
         pmcal_avg = pmcal_avg/(1.0*num);
         print("Route 2 average: %f uW\n" % (pmcal_avg*1e6));
     switch_cal_fac = pm1_avg/pmcal_avg;
     return switch_cal_fac;


def find_avg_sw_ratio(num):
    myr = 0.0;
    for i in range(1, num+1):
         myr = myr + find_switching_ratio(1);
    myr = myr/(1.0*num);
    return myr

def att_enable():
    for att in att_list:
        att.enable()


def att_disable():
    for att in att_list:
        att.disable()


def att_set(value):
    for att in att_list:
        att.set_att(value)


def on():
    att_enable()
    switch.route(2)


def off():
    att_disable()
    switch.route(1)


def reset():
    vbias = vsrc.get_volt()
    vsrc.set_volt(0)
    time.sleep(0.5)
    vsrc.set_volt(vbias)
