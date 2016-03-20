# -*- coding: utf-8 -*-
from __future__ import division

from gpuODE import run_ode
import numpy as np
import pylab as pl

        
FUNCTION = { }

INPUTS = []
FORMULA = {"w": "noise"}
           
x0 = {'w':0}
M = 2
params = {'d':np.ones(M)}

T = 0.001
fs = 1000

stochastic = True
gpu = False

if gpu:
    time, outs = run_ode( FORMULA, FUNCTION, INPUTS,  x0, params, T , fs, inputs = None, stochastic = stochastic, Tterm = 0, gpu = gpu , dtype = np.float32)
    w = outs['w']

else:
    
    outs = run_ode( FORMULA, FUNCTION, INPUTS,  x0, params, T , fs, inputs = None, stochastic = stochastic, nthreads= 1, Tterm = 0, gpu = gpu , dtype = np.float32,debug=True,seed=1234)
    
    w = np.zeros((int(T*fs),len(outs)))
    for i,o in enumerate(outs):
        w[:,i] = o['out'][1]['w'].T
    

pl.figure()
pl.plot(w)

pl.figure()
ax = pl.axes()
ax.hist(w[-1,:]);