# -*- coding: utf-8 -*-

from __future__ import division
from gpuODE import ode, param_grid
import pylab as pl
import numpy as np
            
vdp = ode("vdp")

PARAMETERS = ["mu"]
INPUTS = ["I"]
FORMULA = {"x": "mu*(x-x*x*x/3 -y)",
           "y": "x/mu"}

vdp.setup(FORMULA, PARAMETERS, INPUTS)
vdp.generate_code(debug=False, stochastic = False, gpu = True, dtype = np.float32)
vdp.compile()

x0 = {'x':1,'y':0}

M = 32

mu = np.linspace(-1,10,M)

params = {'mu': mu}

Tterm = 0
T = 100 + Tterm
fs = 100
dt = 1/fs
N = int(T*fs)
Nterm = int(Tterm*fs)
t = np.arange(N-Nterm)/fs

pgrid = param_grid(**params)

time, out = vdp.run(x0, pgrid ,dt, N = N, Nterm = Nterm)

x = out['x']
print x.dtype

pl.figure(56)
pl.clf()
pl.plot( x[:,-1] ,'o-',label="float32")

vdp.generate_code(debug=False, stochastic = False, gpu = True, dtype = np.float64)
vdp.compile()
time, out = vdp.run(x0, pgrid ,dt, N = N, Nterm = Nterm)

x = out['x']
print x.dtype

pl.plot( x[:,-1] ,'.-',label="float64")
pl.legend()


pl.figure(46)
pl.clf()
pl.plot(x + np.ones((N-Nterm,M))*np.arange(M))