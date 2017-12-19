# -*- coding: utf-8 -*-
from __future__ import division

from gpuODE import run_ode
import numpy as np
import pylab as pl
from gpuODE import ode, param_grid, funcs2code

        
FUNCTION = { }

INPUTS = []
FORMULA = {"w": "noise"}
           
x0 = {'w':0}
M = 32
params = {'d':np.ones(M)}

T = 0.001
fs = 1000

stochastic = True
gpu = True
           
PARAMETERS = params.keys()
   
odeRK4 = ode("ode")
odeRK4.setup(FORMULA, PARAMETERS, INPUTS)

Tterm = 0
decimate = 1
inputs = None
debug = True
nthreads = 1

T = T + Tterm
dt = 1.0/float(fs)
N = np.int32(T*fs)

Nterm = np.int32(Tterm*fs).min()
dtype = np.float32
variable_length = False

extra_funcs = funcs2code(FUNCTION, gpu = gpu)
odeRK4.extra_func = extra_funcs
odeRK4.generate_code(debug=debug, gpu = gpu,variable_length = variable_length, stochastic=stochastic, dtype = dtype )
odeRK4.compile()

if gpu==True:

    time, out = odeRK4.run(x0, param_grid(**params) , dt, decimate = decimate, inputs=inputs, N = N, Nterm = Nterm)

    w = out['w']
    
else:

    time,out = odeRK4.run(x0, param_grid(**params) , dt,decimate=decimate, inputs=inputs, N = N , Nterm = Nterm,seed = 1234) 
    
    w = out['w']
        
        
#    
#
#pl.figure()
#pl.plot(w)
#
#pl.figure()
#ax = pl.axes()
#ax.hist(w[-1,:]);