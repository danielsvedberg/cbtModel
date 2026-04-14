
def dDSdt(dsEx, dsInh, dsLeak):
    return dsEx - dsInh - dsLeak

def dISdt(isEx, isInh, isLeak):
    return isEx - isInh - isLeak

def dTHALdt(thalEx, thalInh, thalLeak):
    return thalEx - thalInh - thalLeak

def dCTXdt(input, ctxEx, ctxInh, ctxLeak):
    return input + ctxEx - ctxInh - ctxLeak

def network():
    #set up the initial conditions
    rDS = 0.01
    rIS = 0.01
    rTHAL = 0.01
    rCTX = 0.01
    
    #set up the time steps
    dt = 0.01
    time = jnp.arange(0, 1000, dt)

    #set up storage for the different populations
    rDS_storage = jnp.zeros((len(time),))
    rIS_storage = jnp.zeros((len(time),))
    rTHAL_storage = jnp.zeros((len(time),))
    rCTX_storage = jnp.zeros((len(time),))

    for t in range(1000):
    rDS = dDSdt(dsEx = rCTX, )
    rIS = dISdt()
    rTHAL = dTHALdt()
    rCTX = dCTXdt()
    
    

