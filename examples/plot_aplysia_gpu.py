# -*- coding: utf-8 -*-

import pylab as pl
import numpy as np

import pickle

with open("aplysia_32.pickle","rb") as fi:
    datas = pickle.load(fi)

out32 = datas['out']
    
M = datas['M']
fs = datas['fs']
decimate = datas['decimate']
out64 = datas['out']
T = datas['T']
Tterm = datas['Tterm']

with open("aplysia_64.pickle","rb") as fi:
    datas = pickle.load(fi)

out64 = datas['out']


#%%
N = int(T*fs)
Nterm = int(Tterm*fs)
    
t = np.arange(N-Nterm)/fs
td = t[::decimate]

x32 = out32['V']
x64 = out64['V']

pl.figure(35)
pl.clf()
pl.plot(x32[:,-1],'.-')
pl.plot(x64[:,-1],'.-')

pl.figure(36)
pl.clf()
pl.plot(x64[:,-1]-x32[:,-1],'.-')
