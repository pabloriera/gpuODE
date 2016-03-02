# -*- coding: utf-8 -*-

from __future__ import division

from gpuODE import run_ode
import numpy as np
import pylab as pl

        
FUNCTION = { "func" : ( ["x"], "sin(x)" )}

INPUTS = []
FORMULA = {"ph": "w",
           "w": "-g*func(ph)-e*w "}
           
x0 = {'ph':np.pi-0.01,'w':-40.}
params = {"e":[1.0],"g":[1000]}

T = 10.0
fs = 10000

outs = run_ode( FORMULA, FUNCTION, INPUTS,  x0, params, T , fs, inputs = None, stochastic = False, Tterm = 0, gpu = False, nthreads = 4 , dtype = np.float32)

ph = outs[0]['out'][1]['ph']
w = outs[0]['out'][1]['w']

t = np.arange(T*fs)/fs

pl.plot(t,ph)
pl.xlabel("Time (s)")
pl.ylabel("Phase")