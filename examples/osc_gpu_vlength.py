# -*- coding: utf-8 -*-

from __future__ import division

from gpuODE import ode, param_grid
import pylab as pl
import numpy as np

#from utils import *   

osc = ode("osc")

INPUTS = []
PARAMETERS = ["e","w"]
FORMULA = {"x": "w*y + w*e*x",
           "y": "-w*x + w*e*y"}
                         

osc.setup(FORMULA, PARAMETERS, INPUTS)
osc.generate_code(debug=False, stochastic = False, gpu = True, variable_length = True, dtype = np.float32)
osc.compile()

x0 = {'x':1,'y':0}

M = 32
                    
freq_list = np.linspace(10,1000,M)
w_list = 2*np.pi*freq_list 

params = {'w':w_list,
          'e':-0.1}

Tterm = 0
T = 10/freq_list
fs = freq_list*10
dt = 1/fs
N = np.int32(T*fs)
Nterm = 0

pgrid = param_grid(**params)

time, Y = osc.run(x0, pgrid ,dt, N = N, Nterm = Nterm)

#%%

cN1 = np.r_[0,np.cumsum(N)[:-1]]
cN2 = np.cumsum(N)

y = [ Y[n1*2:n2*2:2] for n1,n2 in zip(cN1,cN2) ]
x = [ Y[n1*2+1:n2*2:2] for n1,n2 in zip(cN1,cN2) ]

pl.cla()
for i,xx in enumerate(x):
    
    pl.plot(xx+i*2)
