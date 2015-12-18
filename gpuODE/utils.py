import numpy as np

def eqparser2(FORMULA, PARAMETERS=[], INPUTS=[]):

    assert type(FORMULA)==dict,"FORMULA must be a dict"
    assert type(PARAMETERS)==list,"PARAMETERS must be a list"
    assert type(INPUTS)==list,"INPUTS must be a list"
    
    """
    EXAMPLE HOPF NORMAL FORM\n
    PARAMETERS = ["e", "w"]\n
    INPUTS = ["I1", "I2"]\n
    FORMULA = {'x': '2*PI*w*y + e*x - (x*x+y*y)*x + I1','y': '-2*PI*w*x + e*y - (x*x+y*y)*y + I2'}
    """

    import re

    #    eq = FORMULA.__repr__()
    eq = ""

    dynvars = FORMULA.keys()
    
    for i,k in enumerate(dynvars):
               
        aux = FORMULA[k]

        for j,k2 in enumerate(dynvars):
            aux = re.sub('\\b'+ k2 + '\\b', "X["+`j`+"]", aux)  
            
        eq = eq+"dX["+`i`+"]=" + aux + ";\n"       
        
    
    params = PARAMETERS
    inputs = INPUTS

    for i,p in enumerate(params):
        eq = re.sub('\\b'+ p + '\\b', "param["+`i`+"]", eq)

    for i,p in enumerate(inputs):
        eq = re.sub('\\b'+ p + '\\b', "input["+`i`+"]", eq)


    N_INPUTS = len(inputs)
    N_EQ = len(dynvars)
    N_PARAMETERS = len(params)

    return dynvars, eq, N_EQ, N_PARAMETERS, N_INPUTS
    
    

def eqparser(FORMULA, PARAMETERS, INPUTS):

    """
    EXAMPLE HOPF NORMAL FORM\n
    PARAMETERS = "e w"\n
    INPUTS = "I1 I2"\n
    FORMULA = "dx = 2*PI*w*y + e*x - (x*x+y*y)*x + I1; dy = -2*PI*w*x + e*y - (x*x+y*y)*y + I2;"\n
    """

    import re

    eq = FORMULA
    dynvars = re.findall("d([a-zA-Z])\ *=",FORMULA)
    params = PARAMETERS.split(" ")

    if INPUTS.count(" ")==len(INPUTS):
        inputs = []
    else:
        inputs = INPUTS.split(" ")

    for p,i in zip(params,range(len(params))):
        eq = re.sub('\\b'+ p + '\\b', "param["+`i`+"]", eq)

    for p,i in zip(inputs,range(len(inputs))):
        eq = re.sub('\\b'+ p + '\\b', "input["+`i`+"]", eq)

    for dv,i in zip(dynvars,range(len(dynvars))):
        eq = re.sub('\\b'+ dv + '\\b', "X["+`i`+"]", eq)

    for dv,i in zip(dynvars,range(len(dynvars))):
        eq = re.sub('\\bd'+ dv + '\\b', "dX["+`i`+"]", eq)

    for dv,i in zip(dynvars,range(len(dynvars))):
        eq = eq.replace(';',';\n ')

    N_INPUTS = len(inputs)
    N_EQ = len(dynvars)
    N_PARAMETERS = len(params)

    return dynvars, eq, N_EQ, N_PARAMETERS, N_INPUTS

def param_grid(**kwargs):

    assert len(kwargs)>0

    fixed = []
    explore = []

    for k,a in kwargs.iteritems():
        assert isinstance(a,(list,int,float,np.ndarray))
        if np.array(a).size == 1:
            fixed.append(k)
        else:
            explore.append(k)

    if len(explore)==1:
        
        flats = []
        flats.append(kwargs[explore[0]].flatten())

        M = flats[0].size

        d_fixed = {k:kwargs[k]*np.ones(M) for k in fixed}
        d_explore = {k:flats[i] for i,k in enumerate(explore)}
        d_params = d_fixed.copy()
        d_params.update(d_explore)

        return d_params
    
    
    elif len(explore)>1:
        togrid = [kwargs[k] for k in explore]
        grids = np.meshgrid(*togrid)
        flats = []
        for g in grids:
            flats.append(g.flatten())

        M = flats[0].size

        d_fixed = {k:kwargs[k]*np.ones(M) for k in fixed}
        d_explore = {k:flats[i] for i,k in enumerate(explore)}
        d_params = d_fixed.copy()
        d_params.update(d_explore)

        return d_params

    else:
        return {k:kwargs[k] for k in fixed}

def funcs2code(fnspecs,gpu=False):

    """
    example:

    fnspecs = {'phi': (['x','a','b'], '-a*x*x*x+b*x*x' ) }

    __device__ float phi( float x,float a, float b);

    __device__ float phi( float x,float a, float b)
    {
        return -a*x*x*x+b*x*x;
    }
    """
    
    if gpu == True:
        s1 = "__device__ float "
    else:
        s1 = "float "
        
    s2 = ")\n{\n"
    s3 = "return "

    string = ""

    for fkey,fspec in fnspecs.iteritems():

        args = ",".join(['float '+aux for aux in  fspec[0] ])

        func = fspec[1]

        string+= s1 + fkey + "(" + args + ");\n"

    string+="\n\n"

    for fkey,fspec in fnspecs.iteritems():

        args = ",".join(['float '+aux for aux in  fspec[0] ])

        func = fspec[1]

        string+= s1 + fkey + "(" + args + s2 + s3 + func + ";\n}\n "

    return string
