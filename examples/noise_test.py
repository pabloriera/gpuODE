# -*- coding: utf-8 -*-
from __future__ import division

from gpuODE import run_ode
import numpy as np
import pylab as pl

        
FUNCTION = { }

INPUTS = ["I"]
FORMULA = {"w": "d*noise"}
           
x0 = {'w':0}
M = 128
params = {'d':np.ones(M)}

T = 1.0
fs = 1000

stochastic = True
gpu = True

if gpu:
    time, outs = run_ode( FORMULA, FUNCTION, INPUTS,  x0, params, T , fs, inputs = None, stochastic = stochastic, Tterm = 0, gpu = gpu , dtype = np.float32)
    w = outs['w']

else:
    
    outs = run_ode( FORMULA, FUNCTION, INPUTS,  x0, params, T , fs, inputs = None, stochastic = stochastic, nthreads= 1, Tterm = 0, gpu = gpu , dtype = np.float32)
    
    w = np.zeros((int(T*fs),len(outs)))
    for i,o in enumerate(outs):
        w[:,i] = o['out'][1]['w'].T
    

dw= np.diff(w,axis=0)*fs
print (dw**2).mean(0).mean()
#hist(dw.flatten())
#%%

from scipy import signal

a = dw[:,M/2]

c = []
for i,b in enumerate(dw.T):
    cc = signal.fftconvolve(b, a[::-1], mode='full')
    c.append( (cc**2).mean()  )

c = np.array(c)    
plot(c/c.max())
