# -*- coding: utf-8 -*-
"""
Simulation of dynamical systems with gpu
"""

import numpy as np
from utils import *
import pandas as pd
from ode import ode

def run_ode( formula, inits, params, T , fs, external_forces_names = None, extra_functions = None, external_forces = None, decimate=1 ,variable_length = False, stochastic = False, Tterm = 0, gpu = False, nthreads = 4 , dtype = np.float32, debug = False,seed =None,threads_per_block=32,pandas_output = True, do_param_grid = True ):
    from gpuODE import ode, param_grid, funcs2code
           
    parameters = params.keys()
       
    odeRK4 = ode("ode")
    odeRK4.setup(formula, parameters, external_forces_names)
    
    Tt = T + Tterm
    dt = 1.0/np.array(fs)
    N = np.int32(Tt*fs)
    
    Nterm = np.int32(Tterm*fs)
    
    if extra_functions:
        extra_funcs = funcs2code(extra_functions, gpu = gpu)
        odeRK4.extra_func = extra_funcs

    odeRK4.generate_code(debug=debug, gpu = gpu, variable_length = variable_length, stochastic=stochastic, dtype = dtype)
    odeRK4.compile()

    if gpu==True:

        import pycuda.driver as drv

        if do_param_grid: 
            pgrid = param_grid(**params)
        else:
            pgrid = params
            
        pgrids = [pgrid]

        if not variable_length:
            M = pgrid.items()[0][1].size
            size = (N-Nterm)*odeRK4.K/decimate*M
            mem = np.dtype(dtype).itemsize*size

            (free,total)=drv.mem_get_info()
            if(mem>free):
                print 'Total Memory: {} MB, Free: {} MB, Needed: {} MB'.format(total/1024**2, free/1024**2,mem/1024**2)
                
#                raise MemoryError('Memory not available') 

            
            fitsize = int(np.ceil(float(mem)/float(free)))
            
            chl = int(M/fitsize)+1
            chl = chl-np.mod(chl,32)
            fitsize = int(np.ceil(M/float(chl)))

            if(mem>free):
                print "Number of GPU runs", fitsize
            

            pgrids = [{} for i in range(fitsize)]
            for k,v in pgrid.iteritems():
                for i,ch in enumerate(chunks(v,chl)):
                    pgrids[i][k] = ch
                    
            
        else:
            assert True, "Variable length Pandas output no yet implemented"
            pass
        
        outs = []
        times = []

        for pgrid in pgrids:

            time, out = odeRK4.run(inits, pgrid , dt, decimate = decimate, external_forces=external_forces, N = N, Nterm = Nterm,seed = seed,threads_per_block=threads_per_block)         
            times.append( time)
            
            if pandas_output:

                dfa = pd.DataFrame(pgrid)

                D = {}
                for k,v in out.iteritems():
                    D[k] = list(v.T)

                dfb = pd.DataFrame(D)
                    
                outs.append( pd.concat([dfa, dfb], axis=1) )

            else:

                outs.append( out )

        if len(outs)==1:
            return times[0],outs[0] 
        else:
            if pandas_output:
                return times, pd.concat(outs,axis=0)
            else:
                return times, outs

    else:

        import distributed_exploration as de
        import timeit
        from random import randint

        if stochastic:
            #Modifications are needed to change the seed on differnet threads
            def func(**args):
                seedr = seed + randint(0,10000)
                time,out = odeRK4.run(inits, args, dt ,decimate=decimate, external_forces=external_forces, N = N , Nterm = Nterm,seed = seedr)
                return time, out
        else:

            def func(**args):
                time,out = odeRK4.run(inits, args, dt ,decimate=decimate, external_forces=external_forces, N = N , Nterm = Nterm)
                return time, out


        tic = timeit.default_timer()
        out0 = de.explore_thread(func, param_grid(**params) ,nthreads=nthreads)
        toc = timeit.default_timer()

        Nm = int(T*fs)/decimate
        M = len(out0)

        if pandas_output:
            L = []

            for i,o in enumerate(out0):
                D = {}
                D.update(o['args'])
                for j,k in enumerate(formula.keys()):
                    D.update({k:o['out'][1][k].flatten()})

                L.append(D)

            df = pd.DataFrame(L)

            return toc-tic, df

        else:

            out = {}
            for j,k in enumerate(formula.keys()):   

                out[k] = np.zeros((Nm,M))
                
            for i,o in enumerate(out0):
                for j,k in enumerate(formula.keys()):
                    out[k][:,i] = o['out'][1][k].T[0,:N]
        
        return toc-tic,out

