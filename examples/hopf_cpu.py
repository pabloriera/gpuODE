# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 07:23:23 2015

@author: miles
"""
from __future__ import division
from gpuODE import ode, param_grid
import pylab as pl
import numpy as np
            
hopf = ode("hopf")

PARAMETERS = ["e","w"]
INPUTS = ["I1","I2"]
FORMULA = {"x": "w*y + w*e*x - (x*x+y*y)*x + I1 ",
           "y": "-w*x + w*e*y - (x*x+y*y)*y + I2"}

hopf.setup(FORMULA, PARAMETERS, INPUTS)
hopf.generate_code(debug=False, stochastic = False, gpu = False, dtype = np.float32)
hopf.compile()

import os
print os.curdir

x0 = {'x':0,'y':1}

M = 2

params = {'e': np.array([-0.01,0.01]),
          'w': 2*np.pi*100}


Tterm = 0
T = 0.5 + Tterm
fs = 100000
dt = 1/fs
N = int(T*fs)
Nterm = int(Tterm*fs)
t = np.arange(N-Nterm)/fs

inputs = np.zeros((N,2))

pgrid = param_grid(**params)

time, out = hopf.run(x0, pgrid ,dt, N = N, Nterm = Nterm)

x = out['x']

#%%

pl.figure(546)
pl.clf()
pl.plot(t,x[:,0])
pl.plot(t,x[:,1])