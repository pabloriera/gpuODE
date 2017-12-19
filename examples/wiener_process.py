# -*- coding: utf-8 -*-
from __future__ import division

from gpuODE import run_ode
import numpy as np
import pylab as pl

        
FUNCTION = { }

INPUTS = []
FORMULA = {"w": "noise"}
           
x0 = {'w':0}
M = 32
params = {'d':np.ones(M)}

T = 100
fs = 1000

stochastic = True
gpu = False

if gpu:
    time, outs = run_ode( FORMULA, FUNCTION, INPUTS,  x0, params, T , fs, inputs = None, stochastic = stochastic, Tterm = 0, gpu = gpu , dtype = np.float32)

else:
    
    t,outs = run_ode( FORMULA, FUNCTION, INPUTS,  x0, params, T , fs, inputs = None, stochastic = stochastic, nthreads= 16, Tterm = 0, gpu = gpu , dtype = np.float32,debug=False,seed=1234)
    
w = outs['w']


pl.figure()
pl.plot(w)

pl.figure()
ax = pl.axes()
ax.hist(w[-1,:]);