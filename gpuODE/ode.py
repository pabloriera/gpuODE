# -*- coding: utf-8 -*-
import numpy as np
from utils import *
import pandas as pd

class ode():

    name = None

    extra_func = ""
    ode_func = None
    code = None

    dynvars = None
    parameters = None
    variables = None
    external_forces_names = None
    noise_sources = None

    odeRK4 = None
    odeRK4_gpu = None

    K = 0
    P = 0
    I = 0
    S = 0

    def __init__(self,name):
        self.name = name

    def setup(self,formula,parameters,external_forces_names=None,noise_sources=None):
           
        assert type(formula)==dict, "Formula must be a dictionary with dynamic variables as keys and equations as items"
    
        self.dynvars, self.ode_func, self.K, self.P, self.I = eqparser2(formula, parameters, external_forces_names,noise_sources)
        self.parameters = parameters
        self.external_forces_names = external_forces_names
        self.noise_sources = noise_sources
        
        self.K = np.int32(self.K)    #number dynamic variables
        self.P = np.int32(self.P)    #number parameters
        self.I = np.int32(self.I)    #number of external forces
        if self.I == 0:
            self.I = 1
        
        if noise_sources:
            self.S = np.int32(len(self.noise_sources))    #number noise sources

    def generate_code(self,debug=False,gpu = False, stochastic=False, variable_length = False, dtype = np.float32, seed = 1234):

        from string import Template

        assert self.ode_func!=None, "Run the setup first"

        if self.noise_sources:
            stochastic = True

        self.variable_length = variable_length
        self.gpu = gpu
        self.stochastic = stochastic

        subds_dict={}
        
        if debug:
            subds_dict["debug"]=""
        else:
            subds_dict["debug"]="//"

        subds_dict["stochastic0"] = ""
        subds_dict["stochastic1"] = ""
        subds_dict["stochastic2"] = ""
        subds_dict["stochastic3"] = ""
        subds_dict["stochastic4"] = ""
        subds_dict["stochastic5"] = ""
        subds_dict["stochastic6"] = ""
                    
        subds_dict["cpu0"] =""
        subds_dict["cpu1"] =""
        subds_dict["cpu2"] =""
        
        subds_dict["device0"] =""
        subds_dict["device1"] =""
        subds_dict["device2"] =""
        subds_dict["device3"] =""
        subds_dict["device4"] =""
            
        if variable_length:
        
            subds_dict["save0"] = "int * N, int * cumsumN"
            subds_dict["save1"] ="Y[ ( cumsumN[m] + c )*"+`self.K`+"+ k ] = X[k];"
            subds_dict["save2"] ="Nm = N[m];"
            subds_dict["save3"] ="0;"
            subds_dict["save4"] ="float *dt"
            subds_dict["save5"] ="dtm = dt[m];"
            
        else:
            subds_dict["save0"] ="int N"
            subds_dict["save1"] ="Y[ ( c*"+`self.K`+"+k)*M + m ] = X[k];"
            subds_dict["save2"] ="Nm = N;"
            subds_dict["save3"] ="_input[n+i*Nm];"
            subds_dict["save4"] ="float dt"
            subds_dict["save5"] ="dtm = dt;"

        if gpu:
            
            if stochastic:
                subds_dict["stochastic0"] = "#include <curand_kernel.h>" 
                subds_dict["stochastic1"] =", curandState *state" 
                subds_dict["stochastic2"] ="".join(["float {} = curand_normal(state);".format(s) for s in self.noise_sources]) 
                subds_dict["stochastic3"] = ", state" 
                subds_dict["stochastic4"] ="""curandState state;
                            curand_init(seed, m, 0, &state);"""
                subds_dict["stochastic5"] =",&state"
                subds_dict["stochastic6"] =",int seed"

            subds_dict["device0"] ="""extern "C"{"""
            subds_dict["device1"] ="__device__"
            subds_dict["device2"] ="__global__"
            subds_dict["device3"] ="}"
            subds_dict["device4"] ="m = blockIdx.x*blockDim.x + threadIdx.x;"

        else:
            
            if stochastic:
                subds_dict["stochastic0"] = """#include <random>

                                              std::normal_distribution<float> distribution(0.0,1.0);"""

                subds_dict["stochastic1"] =",std::default_random_engine *generator" 
                subds_dict["stochastic2"] ="".join(["float {} = distribution(*generator);".format(s) for s in self.noise_sources]) 
                subds_dict["stochastic3"] = ",generator" 
                subds_dict["stochastic4"] = "std::default_random_engine generator(seed+m);"

                subds_dict["stochastic5"] =",&generator"    
                subds_dict["stochastic6"] =",int seed"

            subds_dict["device0"] ="""extern "C"{"""
            subds_dict["device3"] ="}"
            
            subds_dict["cpu0"] ="#include <math.h>"
            subds_dict["cpu1"] ="for(m=0;m<M;m++){"
            subds_dict["cpu2"] ="}"


        subds_dict["K"] = self.K
        subds_dict["P"] = self.P
        subds_dict["I"] = self.I

    
        subds_dict["extra_funcs"] = self.extra_func
        subds_dict["ode_func"] = self.ode_func

#      $debug for(int k=0;k<n_eq;k++)
#      $debug   printf("F1[%d]=%g\\t",k,F1[k] );
           
        self.code_template = Template("""

                    $debug #include <stdio.h>

                    $cpu0

                    $stochastic0

                    $device0

                    #define PI 3.141592653589793                    
                    

                    $extra_funcs
                        

                    $device1 void equations(float t, float *X,float *dX, float *param, float * input $stochastic1)
                    {
                    
                        $stochastic2
    
                        $ode_func
    
                    }
    
                    $device1 void rk4(float &t, float *X, float *param, float * input, float dt $stochastic1)
                    {
    
                        // Runge-Kutta integrator (4th order)
                        // Inputs
                        //   X        Current value of dependent variable
                        //   t        Independent variable (usually time)
                        //   dt       Step size (usually time step)
                        //   equations   Right hand side of the ODE; equations is the
                        //            name of the function which returns dX/dt
                        //            Calling format equations(t,X,dXdt,param,input,seed).
                        //   param    Extra parameters passed to equations
                        //   input    Extra input passed to equations

                        // Output
                        //   X        New value of X after a step of size dt
    
                        float F1[$K], F2[$K], F3[$K], F4[$K], Xtemp[$K];
                        int n_eq = $K;
    
                        //* Evaluate F1 = f(X,t).
                        equations(t,  X, F1, param, input $stochastic3);
    
                        
    
                        //* Evaluate F2 = f( X+dt*F1/2, t+dt/2 ).
                        float half_dt = 0.5*dt;
                        float t_half = t + half_dt;
                        int i;
    
                        for( i=0; i<n_eq; i++ )
                            Xtemp[i] = X[i] + half_dt*F1[i];
    
                        equations(t_half, Xtemp, F2, param, input $stochastic3);
    
                        //* Evaluate F3 = f( X+dt*F2/2, t+dt/2 ).
                        for( i=0; i<n_eq; i++ )
                            Xtemp[i] = X[i] + half_dt*F2[i];
    
                        equations(t_half, Xtemp, F3, param, input $stochastic3);
    
                        //* Evaluate F4 = f( X+dt*F3, t+dt ).
                        float t_full = t + dt;
    					t = t_full;
    					
                        for( i=0; i<n_eq; i++ )
                            Xtemp[i] = X[i] + dt*F3[i];
    
                        equations(t_full, Xtemp, F4, param, input $stochastic3);
    
                        //* Return X(t+dt) computed from fourth-order R-K.
                        for( i=0; i<n_eq; i++ )
                            X[i] += dt/6.*(F1[i] + F4[i] + 2.*(F2[i]+F3[i]));


                    }
    
                    $device2 void odeRK4(float *Y,float * _param, float * _input, float *x0,  $save4, $save0, int Nterm, int decimate, int M $stochastic6)
                    {
                        // Y is the ouput must have N*K x M size
                        // the input must have N size
                        // X is the vector state
                        // dX is the vector state derivative
                        // K is the size of the vector state
                        // dt is the temporal step

                        int n,m,k,p,i,Nm,c;
                        float dtm;
                        
                        $device4
    
                        $save2
                        
                        $save5
                        
                        // define the vector state
                        float X[$K];
    
                        // define the parameters state
                        float param[$P];
    
                        // define the input array
                        float input[$I];

                        float t=0;
    
                        $debug printf("dt=%g\\n",dtm );

                        $cpu1
                        
                        $stochastic4  
                        //$debug printf("seed=%d\\n",seed );

                        for(k=0;k<$K;k++)
                        {
                            X[k] = x0[k];
    
                        $debug printf("m=%d, X0[%d]=%g\\t",m,k,X[k] );
    
                        }
    
                        $debug printf("\\n");
    
                        for(p=0;p<$P;p++)
                        {
                            param[p] = _param[m+p*M];
                            $debug printf("m=%d, p=%d, param=%g\\n", m, p, _param[m+p*M] );
                        }
    
                        $debug printf("input=%g\\n",_input[0] );
    
                        for(n=0;n<Nterm;n++)
                        {
                            for(i=0;i<$I;i++)
                            {
                                input[i] = $save3
                            }
    
                            rk4(t,X,param,input,dtm $stochastic5);    
                        }
                        
                        c = 0;
                        
                        for(n=Nterm;n<Nm;n++)
                        {
                            for(i=0;i<$I;i++)
                            {
                                input[i] = $save3
                            }
    
                            rk4(t,X,param,input,dtm $stochastic5);
                            
                            if(n % decimate == 0)
                            {
                                for(k=0;k<$K;k++)
                                {

                                    $save1
        
                                    $debug printf("m=%d, n=%d, c=%d, X[%d]=%g\\t",m,n,c,k,X[k] );
                                               
                                }
                                
                                c++;
                           }
                              
                        }
                        
                        $cpu2
                    }

                $device3 """)
                
        

        self.dtype = dtype
        
        if dtype == np.float32:
            
            flo = "float"
            
        elif dtype == np.float64:
            
            flo = "double"
        
        self.code = self.code_template.substitute(subds_dict).replace("float",flo)

        
    def compile(self):
        try:
            import pycuda.autoinit
        except:
            print "Pycuda not found"
            pass

        assert self.code!=None, "Generate the code first"

        if self.gpu:
            
            from pycuda.compiler import SourceModule

            # mod = SourceModule(self.code)
            mod = SourceModule(self.code,no_extern_c=True)
            self.odeRK4_gpu = mod.get_function("odeRK4")

        else:           

            import tempfile, os

            fh = tempfile.NamedTemporaryFile(mode='w',suffix='.cpp')
            fh.write(self.code)
            fh.seek(0)
            
            os.environ["CC"]="g++"

            setup_script = \
"""from distutils.core import setup, Extension
module1 = Extension('odeRK4', sources = ["%(filename)s"], libraries = ['m'], extra_compile_args = ['-std=c++11'] )
setup(name = 'odeRK4',version = '1.0', ext_modules = [module1])""" % {"filename":fh.name}
            
            fh2 = tempfile.NamedTemporaryFile(mode='w',suffix='.py')
            fh2.write(setup_script)
            fh2.seek(0)
            
            from distutils.core import run_setup
            
            self.dist = run_setup(fh2.name)
            self.dist.run_command("build")
            
            import ctypes
            self.odeRK4 = ctypes.cdll.LoadLibrary("./build/lib.linux-x86_64-2.7/odeRK4.so").odeRK4      


    def run(self, d_x0, d_params, dt, decimate = 1, external_forces=None, N = None, Nterm = 0, threads_per_block = 32,seed=None):

    
        import timeit

        assert self.code != None, "Source code not generated"

        dtype = self.dtype

        K = self.K
        P = self.P
        I = self.I
        decimate = np.int32(decimate)
        self.Nterm = Nterm =  np.int32(Nterm)   
        
        if type(external_forces) != type(None):
            external_forces = external_forces.astype(dtype, order='F')
            assert external_forces.shape[1]==I, "Inputs shape should be n_external_forces, n_timesteps"
            
        elif type(N) != type(None):
            
            if type(N) in [list,np.ndarray]:
                
                assert self.variable_length, "Variable length not assigned"
            
            elif type(N) in [int,np.int,np.int32]:
                
                external_forces = np.zeros((N,I)).astype(dtype, order='F')
                
                assert Nterm >= 0 and Nterm < N, "Nterm (termalization steps) should be between 0 and N (simulation samples)" 
                
        else:
            assert N!=None or external_forces!=None, "A input signal or a number samples (N) is required"
       
        params = np.vstack(tuple([d_params[p] for p in self.parameters])).T
        params = params.astype(dtype,order='F')

        assert params.shape[1]==P, "Params shape should be n_params=%s n_threads" % P       
        
        M = self.M = np.int32(params.shape[0]) 
        
        
        x0 = [d_x0[k] for k in self.dynvars]        
        x0 =  np.array(x0).astype(dtype)

        assert x0.size==K, "x0 size must be number of variables"       

        

        
        if self.variable_length:
            
            N = np.array(N).astype(np.int32)
            cumsumN = np.r_[0,np.cumsum(N/decimate)[:-1]]
            cumsumN = cumsumN.astype(np.int32)
            Y = np.zeros( np.sum( (N-Nterm)/decimate ) *K ).astype(dtype)
            
            dt = np.array(dt).astype(dtype)
            
            external_forces = np.zeros((  np.sum( (N-Nterm)/decimate ) ,I)).astype(dtype, order='F')
            
        else:
            N = self.N = np.int32(external_forces.shape[0]) 
            Y = np.zeros( ((N-Nterm)*K/decimate,M) ).astype(dtype)
            dt = np.array(dt).astype(dtype) 

        if self.stochastic:
            import random,sys
            if seed==None:
                seed = np.int32(random.randint(0, sys.maxint))
            else:
                seed = np.int32(seed) 



        if self.gpu:

            assert self.odeRK4_gpu != None, "Compile the code first"

            import pycuda.driver as drv
         
            gridShape = ( M / threads_per_block, 1, 1)
            blockShape = (threads_per_block, 1, 1)
            
            mem = np.dtype(dtype).itemsize*Y.size

            (free,total)=drv.mem_get_info()
            if(mem>free):
                print 'Total Memory: {} MB, Free: {} MB, Needed: {} MB'.format(total/1024**2, free/1024**2,mem/1024**2)
                raise MemoryError('Memory not available') 

            if self.variable_length:
                
                if self.stochastic:

                    tic = timeit.default_timer()
                    self.odeRK4_gpu( drv.Out(Y), drv.In(params), drv.In(external_forces), drv.In(x0), drv.In(dt), drv.In(N), drv.In(cumsumN), Nterm, decimate, M ,seed, block=blockShape, grid=gridShape )
                    toc = timeit.default_timer()

                else:

                    tic = timeit.default_timer()
                    self.odeRK4_gpu( drv.Out(Y), drv.In(params), drv.In(external_forces), drv.In(x0), drv.In(dt), drv.In(N), drv.In(cumsumN), Nterm, decimate, M , block=blockShape, grid=gridShape )
                    toc = timeit.default_timer()
    
                return toc-tic,Y
                
            else:

                if self.stochastic:
                    
                    tic = timeit.default_timer()
                    self.odeRK4_gpu( drv.Out(Y), drv.In(params), drv.In(external_forces), drv.In(x0), dt, N, Nterm,decimate, M , seed, block=blockShape, grid=gridShape )
                    toc = timeit.default_timer()

                else:
                    
                    tic = timeit.default_timer()
                    self.odeRK4_gpu( drv.Out(Y), drv.In(params), drv.In(external_forces), drv.In(x0), dt, N, Nterm,decimate, M ,  block=blockShape, grid=gridShape )
                    toc = timeit.default_timer()

                d_Y = {k:Y[i::K] for i,k in enumerate(self.dynvars)}
    
                return toc-tic,d_Y

        else:

            assert self.odeRK4 != None, "Compile the code first"

            from numpy.ctypeslib import ndpointer
            import ctypes
                
            dtype = self.dtype 

            if dtype == np.float32:
                flo = ctypes.c_float
            elif dtype == np.float64:
                flo = ctypes.c_double
            
            self.odeRK4.restype = None
            
            if self.variable_length:

                if self.stochastic:
                
                    self.odeRK4.argtypes = [ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ndpointer(flo, flags="F_CONTIGUOUS"),
                                            ndpointer(flo, flags="F_CONTIGUOUS"),
                                            ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int]
                else:

                    self.odeRK4.argtypes = [ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ndpointer(flo, flags="F_CONTIGUOUS"),
                                            ndpointer(flo, flags="F_CONTIGUOUS"),
                                            ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int]

            else:

                if self.stochastic:

                    self.odeRK4.argtypes = [ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ndpointer(flo, flags="F_CONTIGUOUS"),
                                            ndpointer(flo, flags="F_CONTIGUOUS"),
                                            ndpointer(flo, flags="C_CONTIGUOUS"),
                                            flo,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int]
                
                else:
                
                    self.odeRK4.argtypes = [ndpointer(flo, flags="C_CONTIGUOUS"),
                                            ndpointer(flo, flags="F_CONTIGUOUS"),
                                            ndpointer(flo, flags="F_CONTIGUOUS"),
                                            ndpointer(flo, flags="C_CONTIGUOUS"),
                                            flo,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int]
                
            
            if self.stochastic:
                tic = timeit.default_timer()
                self.odeRK4( Y, params, external_forces, x0, dt, N, Nterm, decimate, M ,seed )
                toc = timeit.default_timer()

            else:

                tic = timeit.default_timer()
                self.odeRK4( Y, params, external_forces, x0, dt, N, Nterm, decimate, M )
                toc = timeit.default_timer()

            d_Y = {k:Y[i::K] for i,k in enumerate(self.dynvars)}
            
            return toc-tic,d_Y   