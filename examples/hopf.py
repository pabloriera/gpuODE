# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:17:16 2015

@author: miles
"""

from gpuODE import run_ode, param_grid
import numpy as np

FUNCTION = { "func" : ( ["x"], "sin(x)" )}

INPUTS = []
FORMULA = {"th": "w",
           "w": "-g*sin(th)-e*w"}
           
x0 = {'th':0.3,'w':0.1}
params = {"e":[1],"g":[10]}

T = 1.0
fs = 100000

outs = run_ode( FORMULA, FUNCTION, INPUTS,  x0, params, T , fs, inputs = None, stochastic = False, Tterm = 0, gpu = False, nthreads = 4 , dtype = np.float32)

th = outs[0]['out'][1]['th']
w = outs[0]['out'][1]['w']

plot(th)