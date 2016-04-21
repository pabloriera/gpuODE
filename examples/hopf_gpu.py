# -*- coding: utf-8 -*-

from __future__ import division
from gpuODE import ode, param_grid
import pylab as pl
import numpy as np
            
hopf = ode("hopf")

PARAMETERS = ["e","w"]
INPUTS = ["I1","I2"]
FORMULA = {"x": "w*y + w*e*x - (x*x+y*y)*x + I1",
           "y": "-w*x + w*e*y - (x*x+y*y)*y + I2"}

hopf.setup(FORMULA, PARAMETERS, INPUTS)
hopf.generate_code(debug=False, stochastic = False, gpu = True, dtype = np.float32)
hopf.compile()

x0 = {'x':0,'y':0}

M1 = 128


ff = np.logspace(np.log10(100.0),np.log10(5000.0),M1)
w = 2*np.pi*ff

es = np.logspace(0,-2,5)
e = np.r_[-es,0,0.0005]

M2 = len(e)

params = {'e': e,
          'w': w}


Tterm = 0
T = 1.0 + Tterm
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

x = out['x']

#%%
x_mean = np.sqrt(np.mean(x**2,0))
pl.cla()
pl.loglog(ff,x_mean.reshape((M1,M2)),'.-')
pl.legend(np.around(e,2),loc='best',title='$\epsilon$')
pl.text(110,1e-1,"$\\frac{dx}{dt} = \omega y + \omega \epsilon x - (x^2+y^2)x + I_1$ \n $\\frac{dy}{dt} = -\omega x + \omega \epsilon y - (x^2+y^2)y + I_2$",fontsize=16)
pl.ylim(1e-4,1e0)
pl.xlabel("Frequency (Hz)" )
pl.ylabel("RMS amplitude" )