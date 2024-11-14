# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:01:36 2021

@author: snk27
"""

def counter_setup(self):
     
    self.write('*RST')
    self.write('*CLS')
    self.write(':EVEN:LEV:AUTO OFF') # Turn off auto trigger level
    self.write(':EVEN:LEV -0.200V')  # Set trigger level
    self.write(':EVEN:SLOP NEG') # Or POS. Trigger on negative slope
    self.write(':EVEN:HYST:REL 0') # Set hysteresis (?)
    self.write(':INP:COUP AC') # Or DC.  Input coupling
    self.write(':INP:IMP 50') # Set input impedance to 50ohms
    self.write(':INP:FILT OFF') # Turn off 100kHz lowpass filter
    self.write(':FUNC "TOT 1"') # Totalize on channel 1
    self.write(':TOT:ARM:STAR:SOUR IMM') # Set start source to immediate (run on command)
    self.write(':TOT:ARM:STOP:SOUR TIM') # Set stop source to time (wait certain time)
    self.write(':TOT:ARM:STOP:TIM 0.1') # Set stop time to 100 ms
    self.write(':INP:ATT 1') # Or 10. Set attenuation factor
    
def counter_set_hysteresis(self,hysteresis):
    self.write(':EVEN:HYST:REL %s' % hysteresis) # Set hysteresis (?)

def counter_impedance(self, ohms = 50, channel = 1):
    self.write(':INP%s:IMP %s' % (channel, ohms)) # Set input impedance to 50 ohms
        
def counter_setup_timed_count(self, channel = 1):
    self.write(':INP:FILT OFF') # Turn off 100kHz lowpass filter
    self.write(':FUNC "TOT %s"' % channel) # Totalize on channel 1
    self.write(':TOT:ARM:STAR:SOUR IMM') # Set start source to immediate (run on command)
    self.write(':TOT:ARM:STOP:SOUR TIM') # Set stop source to time (wait certain time)
    
def counter_set_trigger(self, trigger_voltage = 0.500, slope_positive = True, channel = 1, trigger_level_auto = False):
    if slope_positive is True:
        trigger_slope = 'POS'
    else:
        trigger_slope = 'NEG'
    self.write(':EVEN%s:SLOP %s' %  (channel, trigger_slope)) # Or POS. Trigger on negative slope
    
    if trigger_level_auto is False:
        self.write(':EVEN:LEV:AUTO OFF') # Turn off auto trigger level
        self.write(':EVEN%s:LEV %0.3fV' % (channel, trigger_voltage)) # Set trigger level
        
def timed_count(self, counting_time = 0.1):
    self.write(':TOT:ARM:STOP:TIM %0.3f' % counting_time) # Set stop time to # of seconds
    dcr = self.query(':READ?')
    dcr = float(dcr)/counting_time
    # time.sleep(counting_time + 0.1)
    return dcr

def Func_set_pulse(self, freq=1000, vlow=0.0, vhigh=1.0, width = 100e-6, edge_time = 1e-6):
        vpp = vhigh-vlow
        voffset = vpp/2
        self.write('APPL:PULS %0.6e HZ, %0.6e VPP, %0.6e V' % (freq,vpp,voffset))
        self.write('PULS:WIDT %0.6e' % (width))
        self.write('PULS:TRAN %0.6e' % (edge_time))

def Func_set_output(self,output=False):
        if output is True:  self.write('OUTPUT ON')
        else:               self.write('OUTPUT OFF')