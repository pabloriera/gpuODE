from __future__ import division

from gpuODE import ode, param_grid
import pylab as pl
import numpy as np

#from utils import *   

hr = ode("hr")

extra_func="""
__device__ float phi( float x,float a, float b)
    {
        return -a*x*x*x+b*x*x;
    }
__device__ float psi( float x,float c, float d)
    {
        return  c-d*x*x;
    }
"""

PARAMETERS = ["a","b","c","d","r","s","xr","i"]
INPUTS = ["I"]
FORMULA = {'x':'y + phi( x ,a,b)  - z + i + I', 
            'y':'psi( x , c ,d )- y',
            'z':'r*(s*(x - xr) - z )'}
                         

hr.setup(FORMULA, PARAMETERS, INPUTS)
hr.extra_func = extra_func
hr.generate_code(debug=False, stochastic = False, gpu = True, dtype = np.float32)
hr.compile()

x0 = {'x':-1.6,'y':0,'z':2.5}

M1 = 8
M2 = 8
M = M1*M2
                    
i_list = np.linspace(1.5,3,M1)
r_list = np.logspace(-4,-2,M2)

params = {'a':1,
            'b':3,
            'c':1,
            'd':5,
            'r':r_list,
            's':4,
            'xr':-1.6,
            'i':i_list
            }

Tterm = 1000
T = 4000.0 + Tterm
fs = 100.0
dt = 1/fs
N = int(T*fs)
Nterm = int(Tterm*fs)
t = np.arange(N-Nterm)/fs

pgrid = param_grid(**params)

time, out = hr.run(x0, pgrid ,dt, N = N, Nterm = Nterm)

#%%
x = out['x']

pl.figure(35)
pl.clf()
pl.plot(t, x + np.ones((t.size,M))*np.arange(M)*4)