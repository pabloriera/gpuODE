# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 07:23:23 2015

@author: miles
"""
from __future__ import division
from gpuODE import run_ode, param_grid
import pylab as pl
import numpy as np
            

PARAMETERS = ["e","w"]
INPUTS = ["I1","I2"]
FORMULA = {"x": "w*y + w*e*x - (x*x+y*y)*x + I1 ",
           "y": "-w*x + w*e*y - (x*x+y*y)*y + I2"}

x0 = {'x':0,'y':1}

M = 32

params = {'e': np.linspace(-0.1,0.1,M),
          'w': 2*np.pi*100}

Tterm = 0
T = .5 + Tterm
fs = 100000
dt = 1/fs
N = int(T*fs)
Nterm = int(Tterm*fs)
t = np.arange(N-Nterm)/fs

inputs = np.zeros((N,2))

pgrid = param_grid(**params)

gpu = True

tictoc, out1 = run_ode( FORMULA, INPUTS,  x0, params, T , fs, inputs = inputs, decimate=1 ,variable_length = False, stochastic = False, Tterm = Tterm, gpu = gpu, nthreads = 4 , dtype = np.float64)

gpu = False

tictoc, out2 = run_ode( FORMULA, INPUTS,  x0, params, T , fs, inputs = inputs, decimate=1 ,variable_length = False, stochastic = False, Tterm = Tterm, gpu = gpu, nthreads = 4 , dtype = np.float64)
#%%

for i,r in out1.sort('e').iterrows():
    plot(r['x'])