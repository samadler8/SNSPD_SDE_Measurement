import light
import console_cal
import nloptim
import time
import argparse
import numpy as np
import yaml
import logging
import dark_iv
#  from dark_iv import createfolder, createHMname, out2file, case10

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

fp = open('config.yaml')
config = yaml.load(fp)
fp.close()

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=int, choices=xrange(1, 11),
                    help='detector choice', default=0)
parser.add_argument("-r", type=float, default=config['target_rate'],
                    help="photon flux")
parser.add_argument("-wl", type=int, default=config['wavelength'],
                    help="wavelength in nm")
parser.add_argument("-daqloop", type=int, default=config['daqloop'],
                    help="number of readings per bias")
# for console_optim
parser.add_argument("-ch", type=int, help="sr400 channel", default=0)
parser.add_argument("-wait", type=int, help="wait time before reading counter",
                    default=3)
parser.set_defaults(nopol=False, nocal=False, comparator=False, min_pol=False)
parser.add_argument("--no-pol", dest='nopol', action="store_true",
                    help="no polarization optimization")
parser.add_argument("--no-cal", dest='nocal', action="store_true",
                    help="no att cal")
parser.add_argument("--use-comparator", dest='comparator',
                    action="store_true",
                    help="use comparator being used before counter")
parser.add_argument("--min_pol", dest='min_pol', action="store_true",
                    help="opitmize on min polarization")

args = parser.parse_args()
global bias_dict
bias_dict = config['bias_dict']
NIST = 0  # setting for daq to measure NIST detectors
suffix = ''
NOPOL = args.nopol  # False
NOATTCAL = args.nocal  # False
USE_COMPARATOR = args.comparator
if 'use_comparator' in config:
    USE_COMPARATOR = config['use_comparator']

config['nopol'] = NOPOL
config['nocal'] = NOATTCAL
config['d'] = args.d
config['wait_for_att'] = args.wait
config['ch'] = args.ch
config['take_darks'] = True
config['skip_zero'] = False
config['max_pol'] = not args.min_pol
config['target_rate'] = args.r
# config['target_att']=47
# config['measure_cmax'] = False
optics = None


def main(config, update=None):
    global NOPOL, NOATTCAL, suffix, bias_dict, optics
    suffix = '_%dnm' % config['wavelength']
    foldername = dark_iv.createfolder()
    basename = dark_iv.createHMname()
    bname = basename
    if config['nopol'] == True:
        basename = basename.strip('.dat') + '_no_pol.dat'
    elif config['max_pol'] == True:
        basename = basename.strip('.dat') + '_maxpol.dat'
    else:
        basename = basename.strip('.dat') + '_minpol.dat'
    fname = foldername + basename.strip('.dat')+'%s.dat' % suffix
    logger.debug('filename: %s' % fname)
    config['basefilename'] = foldername + basename
    out = dark_iv.out2file(fname)
    # setup optics
    optics = console_cal.daq(config)
    optics.running = True
    # optics.setup_dark()
    if config['skip_zero']:
        optics.set_SKIPZERO()
    else:
        optics.set_SKIPZERO(False)
    out.writeln('#  NIST Detector:  SMA%d' % config['d'])
    out.writeln('#')
    out.writeln('# Config')
    out.writeln('# %s' % repr(config))
    out.writeln('#')
    vsrc = light.vsrc
    counter = light.counter
    dmm = light.dmm

    def cleanup():
        optics.cleanup()
        vsrc.close()
        dmm.close()
        counter.close()

    #vsrc.set_power_on()
    # Set up bias and triggering thresholds
    vsrc.set_volt(1,0)
    if USE_COMPARATOR:
        vsrc.set_volt(config['thresh_ch'], config['thresh'])
        #print("Setting threshold. Channel: %d, value: %f" %
        #        (config['thresh_ch'], config['thresh']))
        vsrc.set_volt(config['hysteresis_ch'], config['hysteresis'])
        #print("Setting Hysteresis. Channel: %d, value: %f" %
        #        (config['hysteresis_ch'], config['hysteresis']))
    counter.set_time(config['counter_time'])

    vbias, vstop, vstep, vpol = bias_dict
    daq_loop = config['daqloop']
    # counter.meter.port.timeout=0.1
    if config['take_darks']:
        optics.setup_dark()
        out.writeln('#  Dark counts')
        # start = time.time()
        dark_iv.v_jump = 0.015
        vswitch = dark_iv.case10(out, vsrc, dmm, counter,
                                 vbias, vstep, vstop, daq_loop,
                                 update=update)
        if not dark_iv.RUNNING:
            cleanup()
            return
        vsrc.set_volt(1,0)
        # vsrc.set_power_off()
    #  Need to figure out attenuation for tuning polarization
    #     Use initial target_att
    # sweep_time = time.time()-start
    if (not config['nopol']) and dark_iv.RUNNING:
        optics.setup_light()
        logger.info('Start polarization optimization')
        # fname_pol = createfilename()
        fname_pol = config['basefilename']
        fname_pol = fname_pol.strip('.dat')+'_optim_pol.dat'
        logger.debug('filename polarization: %s' % fname_pol)
        out_pol = dark_iv.out2file(fname_pol)

        nloptim.output_file = out_pol  # pass file pointer into nloptim
        #vsrc.set_power_on()
        vsrc.set_volt(1,0)
        vsrc.set_volt(config['thresh_ch'], config['thresh'])
        vsrc.set_volt(config['hysteresis_ch'], config['hysteresis'])
        if vpol > vswitch:
            # if True:
            vpol = vswitch * 0.7
        logger.info('Set vpol to %f' % vpol)
        vsrc.set_volt(1,vpol)
        nloptim.PM = False
        nloptim.MAX = config['max_pol']
        nloptim.fname_pol_opt = fname_pol
        while dark_iv.RUNNING:  # True:
            logging.info('Starting optimize')
            pol_data = nloptim.optimize()
            print("Done optimize")
            logger.info(pol_data)
            if pol_data is None:
                logger.debug('counts were zero...')
                test_voltage = dmm.get_volt()
                logger.debug('voltage on device: %f' % test_voltage)
                if test_voltage < 100e-3:
                    logger.debug('Need more light')
                    config['target_att'] -= 5
                else:  # The detectors latched, need to try a lower bias
                    logger.debug('reset the detectors, try lower vbias')
                    logger.debug(' could be too much light???')
                    vsrc.set_volt(1,0)
                    time.sleep(1)
                    vpol = vpol*0.9
                    vsrc.set_volt(1,vpol)
                optics.setup_light()
            else:
                break

        np.savetxt(out_pol.fp, pol_data)
        out_pol.close()

        #  Turn off the detector after polarization optimization
        vsrc.set_volt(1,0)

    if not config['nocal']:
        vsrc.set_volt(1,0)
        logger.info('Start to calibrate attenuators')
        # optics.running = True
        # optics.start_monitor()
        # optics.set_SKIPZERO()
        # print 'SKIPZERO: ',console_cal.SKIPZERO
        optics.cal()
    # print switch.__name__
    # raw_input('Hit enter after moving the fiber')
    out.writeln('#  Light counts')
    optics.setup_light()
    vsrc.set_volt(1,0)
    time.sleep(1)
    #vsrc.set_power_on()

    dark_iv.case10(out, vsrc, dmm, counter,
                   vbias, vstep, vstop, daq_loop, update)
    out.close()
    vsrc.set_volt(1,0)

    if config['measure_cmax'] and config['max_pol'] and config['wavelength']==1550:

        fname_cmax = bname.strip('.dat')+'_cmax.dat'
        fname_cmax = foldername+fname_cmax.strip('.dat')+'%s.dat' % suffix
        out_cmax = dark_iv.out2file(fname_cmax)
        logger.debug('filename: %s' % fname_cmax)
        print('Cmax filename: %s' % fname_cmax)
        att_start = config['cmax_start']
        att_end = config['cmax_stop']
        att_val = att_start
        for att in light.att_list:
            att.set_att(att_val)
        out_cmax.writeln('#'*10)
        out_cmax.writeln('# scan bias')
        out_cmax.writeln('# case: 90')
        out_cmax.writeln('#'*10)
        out_cmax.writeln('# Voltage source')
        vsrc.writeconfig(out_cmax.fp)
        out_cmax.writeln('#'*10)
        out_cmax.writeln('# Voltage meter')
        dmm.writeconfig(out_cmax.fp)
        out_cmax.writeln('#'*10)
        out_cmax.writeln('# Counter')
        counter.writeconfig(out_cmax.fp)
        out_cmax.writeln('#'*10)
        out_cmax.writeln('# Attenuators 1 & 2 set to %f' % att_val)


        while(att_val >= att_end):
            for nn in range(0, config['cmax_num']):
                out_cmax.writeln('#')
                out_cmax.writeln('# Cmax Att: %f, trial: %d of %d' % (att_val, nn+1, config['cmax_num']))
                out_cmax.writeln('#')
                light.att3.set_att(att_val)
                dark_iv.get_dmm_counter(dmm, counter)
                vsw = dark_iv.case90(out_cmax, vsrc, dmm, counter,
                               config['cmax_vstart'], vstep, vstop,
                               att_val, daq_loop, update)
                print('Cmax Att: %f, Vswitch: %f' % (att_val, vsw))
                if not dark_iv.RUNNING:
                    cleanup()
                    return
                light.att3.set_att(att_start)
                vsrc.set_volt(1,0.0)
                dark_iv.get_dmm_counter(dmm, counter)
                time.sleep(10)
                dark_iv.get_dmm_counter(dmm, counter)
            att_val = att_val - 3.0

        for att in light.att_list:
            att.set_att(34)
            att.disable()
        out_cmax.close()


    if not dark_iv.RUNNING:
        cleanup()
        return
    #vsrc.set_power_off()
    optics.setup_dark()
    optics.post_0dB()
    cleanup()
    # counter.set_continuous()
    # counter.close()


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    #wl_list = range(1470,1576,2)
    #wl_list = range(1575,1470,-5)
    #wl_list = range(1502,1470,-2)
    #wl_list = range(1505,1465,-5)
    #wl_list = range(1535,1498,-1)
    #wl_list = [1550]
    wl_list = [args.wl]
    #wl_list = [1510, 1515, 1570, 1575]
    for wl in wl_list:
        config['wavelength'] = wl
        print('Setting wavelength to: %d' % wl)
        suffix = '_%dnm' % config['wavelength']
        main(config)
