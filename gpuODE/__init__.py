# -*- coding: utf-8 -*-
"""
Simulation of dynamical systems with gpu
"""


import numpy as np
from utils import *


def run_ode( FORMULA, FUNCTIONS, INPUTS,  inits, params, T , fs, inputs = None, stochastic = False, Tterm = 0, gpu = False, nthreads = 4 , dtype = np.float32):
    
    from gpuODE import ode, param_grid, devicefuncs
           
    PARAMETERS = params.keys()
       
    odeRK4 = ode("ode")
    odeRK4.setup(FORMULA, PARAMETERS, INPUTS)
    
    T = T + Tterm
    dt = 1/fs
    N = int(T*fs)
    
    Nterm = int(Tterm*fs)
    
    extra_funcs = devicefuncs(FUNCTIONS, gpu = gpu)
    odeRK4.extra_func = extra_funcs
    odeRK4.generate_code(debug=False, gpu = gpu, stochastic=stochastic, dtype = dtype )
    odeRK4.compile()

    if gpu==True:

        time, out = odeRK4.run(inits, param_grid(**params) , dt, inputs=inputs, N = N, Nterm = Nterm)

        return time, out
        
    else:
        import distributed_exploration as de

        def func(**args):
            time,out = odeRK4.run(inits, args, dt , inputs=inputs, N = N , Nterm = Nterm)
            return time, out
        
        outs = de.explore_thread(func, param_grid(**params) ,nthreads=4)
        
        return outs

class ode():

#    import pycuda.driver as drv
    name = None

    extra_func = None
    ode_func = None
    code = None

    dynvars = None
    parameters = None
    variables = None
    inputs = None

    odeRK4 = None
    odeRK4_gpu = None

    K = 0
    P = 0
    I = 0

    def __init__(self,name):
        self.name = name

    def setup(self,formula,parameters,inputs):

        if inputs == []:
            inputs = ["I"]
            
        if(type(formula)==dict):
            self.dynvars, self.ode_func, self.K, self.P, self.I = eqparser2(formula,  parameters, inputs)
            self.parameters = parameters
            self.inputs = inputs
            
#        else:
#            self.dynvars, self.ode_func, self.K,self.P,self.I = eqparser(formula,  parameters, inputs)
#            self.parameters = parameters.split(' ')
#            self.inputs = inputs.split(' ')

        self.K = np.int32(self.K)     #numero de variables dinámicas
        self.P = np.int32(self.P)     #numero de parámetros
        self.I = np.int32(self.I)     #numero de inputs

    def generate_code(self,debug=False,gpu = False, stochastic=False, dtype = np.float32):

        assert self.ode_func!=None, "Run the setup first"

        self.gpu = gpu

        if debug:
            debug_str=""
        else:
            debug_str="//"
          
        self.dtype = dtype
        
        if dtype == np.float32:
            
            flo = "float"
            
        elif dtype == np.float64:
            
            flo = "double"

        device_str = [""]*5
        stochastic_str = [""]*6
        cpu_str = [""]*3

        if gpu:
            
            if stochastic:
                stochastic_str[0] =  "#include <curand_kernel.h>" 
                stochastic_str[1] = ", curandState *state" 
                stochastic_str[2] = "float noise = curand_normal(state);" 
                stochastic_str[3] =  ", state" 
                stochastic_str[4] = """curandState state;
                            curand_init(1234, m, 0, &state);"""
                stochastic_str[5] = ",&state"
    

            device_str[0] = ""#"""extern "C"{"""
            device_str[1] = "__device__"
            device_str[2] = "__global__"
            device_str[3] = ""#"}"
            device_str[4] = "m = blockIdx.x*blockDim.x + threadIdx.x;"

        else:

			cpu_str[0] = "#include <math.h>"
			cpu_str[1] = "for(m=0;m<M;m++){"
			cpu_str[2] = "}"

        self.code = """

                    """+debug_str+""" #include <stdio.h>

                    """+cpu_str[0]+"""

                    """+stochastic_str[0]+"""                    
    
                    """+device_str[0]+"""


                    #define PI 3.141592653589793                    
                    

                        """+(self.extra_func,"")[self.extra_func==None]+"""
                        
                        
                        """+device_str[1]+""" void equations( float *X,float *dX, float *param, float * input """+stochastic_str[1]+""")
                        {
                        
                           """+stochastic_str[2]+"""
        
                            """+self.ode_func+"""
        
                        }
        
                        """+device_str[1]+""" void rk4(float *X, float *param, float * input, float dt """+stochastic_str[1]+""")
                        {
        
                            // Runge-Kutta integrator (4th order)
                            // Inputs
                            //   X          Current value of dependent variable
                            //   n_eq         Number of elements in dependent variable X
                            //   t          Independent variable (usually time)
                            //   dt        Step size (usually time step)
                            //   derivsRK   Right hand side of the ODE; derivsRK is the
                            //              name of the function which returns dX/dt
                            //              Calling format derivsRK(X,t,param,dXdt).
                            //   param      Extra parameters passed to derivsRK
                            // Output
                            //   X          New value of X after a step of size dt
        
                            float F1[%(K)s], F2[%(K)s], F3[%(K)s], F4[%(K)s], Xtemp[%(K)s];
                            int n_eq = %(K)s;
        
                            //* Evaluate F1 = f(X,t).
                            equations( X, F1, param, input"""+stochastic_str[3]+""");
        
              """+debug_str+""" for(int k=0;k<n_eq;k++)
              """+debug_str+"""     printf("F1[%%d]=%%g\\t",k,F1[k] );
        
                            //* Evaluate F2 = f( X+dt*F1/2, t+dt/2 ).
                            float half_dt = 0.5*dt;
                            //float t_half = t + half_dt;
                            int i;
        
                            for( i=0; i<n_eq; i++ )
                                Xtemp[i] = X[i] + half_dt*F1[i];
        
                            equations( Xtemp, F2, param, input"""+stochastic_str[3]+""");
        
                            //* Evaluate F3 = f( X+dt*F2/2, t+dt/2 ).
                            for( i=0; i<n_eq; i++ )
                                Xtemp[i] = X[i] + half_dt*F2[i];
        
                            equations( Xtemp, F3, param, input"""+stochastic_str[3]+""");
        
                            //* Evaluate F4 = f( X+dt*F3, t+dt ).
                            //float t_full = t + dt;
        
                            for( i=0; i<n_eq; i++ )
                                Xtemp[i] = X[i] + dt*F3[i];
        
                            equations( Xtemp, F4, param, input"""+stochastic_str[3]+""");
        
                            //* Return X(t+dt) computed from fourth-order R-K.
                            for( i=0; i<n_eq; i++ )
                                X[i] += dt/6.*(F1[i] + F4[i] + 2.*(F2[i]+F3[i]));
                        }
        
                        """+device_str[2]+""" void odeRK4(float *Y,float * _param, float * _input, float *x0, int N, int Nterm, int M, int K, float dt)
                        {
                            // Y is the ouput must have N*K x M size
                            // the input must have N size
                            // X is the vector state
                            // dX is the vector state derivative
                            // K is the size of the vector state
                            // dt is the temporal step

                            int n,m,k,p,i;
                            """+device_str[4]+"""
        
                            // define the vector state
                            float X[%(K)s];
        
                            // define the parameters state
                            float param[%(P)s];
        
                            // define the input array
                            float input[%(I)s];
        
                            """+stochastic_str[4]+"""                            
        
              """+debug_str+""" printf("dt=%%g\\n",dt );

              				"""+cpu_str[1]+""" 
        
                            for(k=0;k<K;k++)
                            {
                                X[k] = x0[k];
        
              """+debug_str+""" printf("m=%%d, X0[%%d]=%%g\\t",m,k,X[k] );
        
                            }
        
              """+debug_str+""" printf("\\n");
        
                            for(p=0;p<%(P)s;p++)
                            {
                                param[p] = _param[m+p*M];
              """+debug_str+""" printf("m=%%d, p=%%d, param=%%g\\n", m, p, _param[m+p*M] );
                            }
        
              """+debug_str+""" printf("input=%%g\\n",_input[0] );
        
                            for(n=0;n<Nterm;n++)
                            {
                                for(i=0;i<%(I)s;i++)
                                {
                                    input[i] = _input[n+i*N];
                                }
        
                                rk4(X,param,input,dt"""+stochastic_str[5]+""");
        
                            }
                            
                            for(n=Nterm;n<N;n++)
                            {
                                for(i=0;i<%(I)s;i++)
                                {
                                    input[i] = _input[n+i*N];
                                }
        
                                rk4(X,param,input,dt"""+stochastic_str[5]+""");
                                
                                for(k=0;k<K;k++)
                                {
                                    Y[ ( (n-Nterm)*K+k)*M + m ] = X[k];
        
              """+debug_str+""" printf("m=%%d, X[%%d]=%%g\\t",m,k,X[k] );
        
                                }
              """+debug_str+""" printf("\\n");
                            }
            				
            				"""+cpu_str[2]+""" 
                        }

                    """+device_str[3]

        K = self.K
        P = self.P
        I = self.I
    
        self.code = self.code % {'K':K,'P':P,'I':I}
        
        self.code = self.code.replace("float",flo)

    def compile(self):
        import pycuda.autoinit
        assert self.code!=None, "Generate the code first"

        if self.gpu:

            from pycuda.compiler import SourceModule

            mod = SourceModule(self.code)
            # mod = SourceModule(self.code,no_extern_c=True)
            self.odeRK4_gpu = mod.get_function("odeRK4")

        else:            

            import tempfile

            fh = tempfile.NamedTemporaryFile(mode='w',suffix='.c')
            fh.write(self.code)
            fh.seek(0)
            
            setup_script = \
"""from distutils.core import setup, Extension
module1 = Extension('odeRK4', sources = ["%(filename)s"], libraries = ['m'])
setup(name = 'odeRK4',version = '1.0', ext_modules = [module1])""" % {"filename":fh.name}
            
            fh2 = tempfile.NamedTemporaryFile(mode='w',suffix='.py')
            fh2.write(setup_script)
            fh2.seek(0)
            
            from distutils.core import run_setup
            
            self.dist = run_setup(fh2.name)
            self.dist.run_command("build")
            
            import ctypes
            self.odeRK4 = ctypes.cdll.LoadLibrary("./build/lib.linux-x86_64-2.7/odeRK4.so").odeRK4        


    def run(self, d_x0, d_params, dt, inputs=None, N = None, Nterm = 0, THREAD_NUM = 32):
    
        import timeit

        assert self.code != None, "source code not generated"

        dtype = self.dtype

        K = self.K
        P = self.P
        I = self.I
        self.Nterm = Nterm =  np.int32(Nterm)   
        
        if type(inputs) != type(None):
            inputs = inputs.astype(dtype, order='F')
            assert inputs.shape[1]==I, "inputs shape should be n_inputs, n_timesteps"
            
        elif type(N) != type(None):
            inputs = np.zeros((N,I)).astype(dtype, order='F')
                
        else:
            assert N!=None or inputs!=None, "A input signal or a number samples (N) is required"
       
        N = self.N = np.int32(inputs.shape[0]) 

        assert Nterm >= 0 and Nterm < N, "Nterm (termalization steps) should be between 0 and N (simulation samples)" 
        
        params = np.vstack(tuple([d_params[p] for p in self.parameters])).T
        params = params.astype(dtype,order='F')

        assert params.shape[1]==P, "params shape should be n_params=%s n_threads" % P        
        
        M = self.M = np.int32(params.shape[0]) 
        
        x0 = [d_x0[k] for k in self.dynvars]        
        x0 =  np.array(x0).astype(dtype)

        assert x0.size==K, "x0 size must be number of variables"       

        dt = np.array(dt).astype(dtype) 
        Y = np.zeros( ((N-Nterm)*K,M) ).astype(dtype)


        if self.gpu:

            assert self.odeRK4_gpu != None, "Compile the code first"

            import pycuda.driver as drv
         
            gridShape = ( M / THREAD_NUM, 1, 1)
            blockShape = (THREAD_NUM, 1, 1)
            
            mem = np.dtype(dtype).itemsize

            (free,total)=drv.mem_get_info()
            if(mem>free):
                print 'Free/Needed/Total Memory', free/1024**3,mem/1024**3, total/1024**3
                raise MemoryError('Memory not available') 

            tic = timeit.default_timer()
            self.odeRK4_gpu( drv.Out(Y), drv.In(params), drv.In(inputs), drv.In(x0), N, Nterm, M , K, dt, block=blockShape, grid=gridShape )
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
            self.odeRK4.argtypes = [ndpointer(flo, flags="C_CONTIGUOUS"),
                                    ndpointer(flo, flags="F_CONTIGUOUS"),
                                    ndpointer(flo, flags="F_CONTIGUOUS"),
                                    ndpointer(flo, flags="C_CONTIGUOUS"),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    flo]
                
            
            tic = timeit.default_timer()
            self.odeRK4( Y, params, inputs, x0, N, Nterm, M , K, dt )
            toc = timeit.default_timer()

            d_Y = {k:Y[i::K] for i,k in enumerate(self.dynvars)}
            
            return toc-tic,d_Y   