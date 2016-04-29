# -*- coding: utf-8 -*-

from __future__ import division

from gpuODE import ode, param_grid
import pylab as pl
import numpy as np

#from utils import *   

IF = ode("IF")

PARAMETERS = ["tau","x0","d","th","i"]
INPUTS = ["I"]
FORMULA = {'x':'-x/tau + i + I + d*noise; if(x>th) x=x0;'}
                         

IF.setup(FORMULA, PARAMETERS, INPUTS)
#IF.extra_func = extra_func
IF.generate_code(debug=False, stochastic = True, gpu = True, dtype = np.float32)
IF.compile()


M1 = 64
M2 = 8
M = M1*M2
i_list = np.logspace(np.log10(1),np.log10(2),M2)

params = {'tau':[1]*M1,
            'd':0.8,
            'th':1,
            'x0':0,
            'i':i_list}

init = {'x':0}


Tterm = 0
T0 = 100.0
T = T0+Tterm
fs = 100.0
dt = 1/fs
N = int(T*fs)
Nterm = int(Tterm*fs)

decimate = 4

pgrid = param_grid(**params)

time, out = IF.run(init, pgrid ,dt, decimate = decimate, N = N, Nterm = Nterm)

#%%
t = np.arange(N-Nterm)/fs
td = t[::decimate]

x = out['x']

pl.figure(35)
pl.clf()
pl.plot(td[:-1], -diff(x,axis=0) + np.ones((td.size-1,M))*np.arange(M)*1)

#spi = {}
import pandas as pd

df = pd.DataFrame()
dx = -diff(x,axis=0)

for i in xrange(M):
    
    spikes = np.where(dx[:,i]>0.8)[0]/fs*decimate
    D = {'i':pgrid['i'][i],'spikes':spikes,'fs':fs/decimate}
    df = df.append(D,ignore_index=True)
    
Ns = df.spikes.apply(np.size).min()

sts = np.zeros((M,Ns))

for i in xrange(M):
    
    sts[i,:] = df.spikes.iloc[i][:Ns]

pl.figure(36)
pl.clf()
    
for k,i in enumerate(i_list):
    mask = np.array(df.i == i)
#    subplot(len(i_list),1,k+1)
    loglog(sts[mask,:].mean(0),sts[mask,:].std(0),'o-',label=np.around(i,2))

pl.legend(loc='best')
    
