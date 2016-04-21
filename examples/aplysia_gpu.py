# -*- coding: utf-8 -*-

from __future__ import division

from gpuODE import ode, param_grid, funcs2code
import pylab as pl
import numpy as np

aplysia = ode("aplysia")

Vhna = 115.0
Vhk = -12.0
V_i = 30.0
V_k = -75.0

a = (Vhna-Vhk)/(V_i-V_k)
b = -(Vhna*V_k-V_i*Vhk )/(V_i-V_k)

params = {'a'  :a,
        'b'  :b,
        'g_i':4.0,
        'g_t':0.01,
        'g_k':0.3,
        'g_p':0.03,
        'g_l':0.003,
        'K_p':0.5,
        'K_c':0.008,
        'rho':0.0001,
        'V_i':V_i,
        'V_k':V_k,
        'V_l':-40.0,
        'V_ca':140.0,
        'tau_xi':10.0,
        'tau_xt':70.0,
        'tau':0.018,
        'noise_amp':0.002}   

inits = {'V':-60.0,
         'x_t': 0.2,#y
         'x_k': 0.9,#n
         'y_i': 0.1,#h
         'c': 0.6}
         
fnspecs = {'a_m': (['V'], '0.1*(50.0-V)/(expf( (50.0-V)/10.0 ) - 1.0)' ),
           'b_m': (['V'], '4.0*expf((25.0-V)/18.0)' ),
           's_i': (['V'], 'a_m(V)/( a_m(V)+b_m(V) )'),

           'a_h': (['V'], '0.07*expf( (25.0-V)/20.0 )' ),
           'b_h': (['V'], '1.0/(expf( (55.0-V)/10.0 ) + 1.0)' ),
           'z_i': (['V'], 'a_h(V)/( a_h(V)+b_h(V) )'),
           'tau_yi': (['V'], '12.5/(a_h(V)+b_h(V))'),
 
           'a_n': (['V'], '0.01*(55.0-V)/( expf( (55.0-V)/10.0 ) - 1.0)' ),
           'b_n': (['V'], '0.125*expf((45.0-V)/80.0)' ),
           's_k': (['V'], 'a_n(V)/( a_n(V)+b_n(V) )'),
           'tau_xk': (['V'], '12.5/(a_n(V)+b_n(V))'),

           'Vs' : (['V','a','b'], 'a*V+b'),
          
           's_t': (['V'], '1.0/( expf(0.15*(-50.0-V))+1.0)')}    


INPUTS = ["I"]

stochastic = False

if stochastic:
    cond = "+ noise_amp*noise"
else:
    cond = ""
    
FORMULA = {"c"  : "rho*(K_c*x_t*(V_ca - V) - c) /tau",
           "x_t": "( ( s_t(V) - x_t)/tau_xt {}) / tau ".format(cond),
           "x_k": "( s_k(Vs(V,a,b)) - x_k)/tau_xk(Vs(V,a,b))/tau  ",
           "y_i": "( z_i(Vs(V,a,b)) - y_i)/tau_yi(Vs(V,a,b))/tau",
           "V": "( (g_i*powf(s_i(Vs(V,a,b)),3.0)*y_i+g_t*x_t)*(V_i-V) + (g_k*powf(x_k,4.0) + g_p*c/(K_p+c) )*(V_k-V)  + g_l*(V_l-V) + I  )/tau "}
           
PARAMETERS = params.keys()

aplysia.setup(FORMULA, PARAMETERS, INPUTS)
aplysia.extra_func = funcs2code(fnspecs,gpu=True)
aplysia.generate_code(debug=False, stochastic = stochastic, gpu = True, dtype = np.float32)
aplysia.compile()

M = 32

params['tau_xt'] = np.linspace(70,400,M)

Tterm = 1000
T = 1000.0 + Tterm
fs = 400.0
dt = 1/fs
N = int(T*fs)
Nterm = int(Tterm*fs)

decimate = 4

pgrid = param_grid(**params)

time, out = aplysia.run(inits, pgrid ,dt, decimate = decimate, N = N, Nterm = Nterm)

#%%
import pickle 
data = {'params':params,'M':M, 'inits':inits,'T':T,'Tterm':Tterm,'fs':fs,'out':out,'decimate':decimate}

print "Saving to disk"
with open("aplysia_32.pickle","wb") as fi:
    pickle.dump(data,fi)   
    
aplysia.generate_code(debug=False, stochastic = stochastic, gpu = True, dtype = np.float64)
aplysia.compile()
time, out = aplysia.run(inits, pgrid ,dt, decimate = decimate, N = N, Nterm = Nterm)    
    
data = {'params':params,'M':M, 'inits':inits,'T':T,'Tterm':Tterm,'fs':fs,'out':out,'decimate':decimate}
print "Saving to disk"
with open("aplysia_64.pickle","wb") as fi:
    pickle.dump(data,fi)   