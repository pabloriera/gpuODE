# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 00:20:39 2016

@author: miles
"""

# -*- coding: utf-8 -*-

from __future__ import division
from gpuODE import ode, param_grid
import pylab as pl
import numpy as np
            
hopf = ode("hopf")

PARAMETERS = ["e","w","a","b","F","g"]
INPUTS = ["I1","I2"]
FORMULA = {"x1": "w*y1 + w*e*x1 - (x1*x1+y1*y1)*x1 + F*I1 + w*a*x2",
           "y1": "-w*x1 + w*e*y1 - (x1*x1+y1*y1)*y1 + F*I2 + w*a*y2",
           "x2": "g*w*y2 + g*w*e*x2 - (x2*x2+y2*y2)*x2 + g*w*b*a*x1",
           "y2": "-g*w*x2 + g*w*e*y2 - (x2*x2+y2*y2)*y2 + g*w*b*a*y1"}

hopf.setup(FORMULA, PARAMETERS, INPUTS)
hopf.generate_code(debug=False, stochastic = False, gpu = True, dtype = np.float32)
hopf.compile()

x0 = {'x1':0,'y1':0,'x2':0,'y2':0}

M1 = 64

ff = np.logspace(np.log10(100.0),np.log10(5000.0),M1)
w = 2*np.pi*ff

e = -0.01
amps = 1

c = logspace(log10(0.01),log10(2),8)

M2 = len(c)



params = {'e': e,
          'w': w,
          'F':amps,
          'a':c,
          'b':-1,
          'g':1.2}


Tterm = 0
T = 0.5 + Tterm
fs = 100000
dt = 1/fs
N = int(T*fs)
Nterm = int(Tterm*fs)
t = np.arange(N-Nterm)/fs


tramp = 0.05
ramp = np.ones(t.size)
ix = t<tramp
ramp[ix] = (1-np.cos( t[ix]/tramp*np.pi ))*0.5

f0 = 1000
inputs = np.empty((N,2))
inputs[:,0] = ramp*np.sin(2*np.pi*f0*t)
inputs[:,1] = ramp*np.cos(2*np.pi*f0*t)

pgrid = param_grid(**params)


time, out = hopf.run(x0, pgrid ,dt, inputs = 4*inputs, Nterm = Nterm)

x1 = out['x1']
x2 = out['x2']

#%%
pl.clf()
pl.subplot(211)
x_mean = np.sqrt(np.mean(x1**2,0))
pl.cla()
pl.loglog(ff,x_mean.reshape((M1,M2)),'.-')

pl.subplot(212)

x_mean = np.sqrt(np.mean(x2**2,0))
pl.cla()
pl.loglog(ff,x_mean.reshape((M1,M2)),'.-')
