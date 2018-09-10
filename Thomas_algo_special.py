def trid_known(b,f):
    """Function that solves [Ax=b'] for A beeing a tridiagonal matrix (a,b,c).
    Vectors a,b,c represents A as such:
        [a: lower diagonal],[b: Diagonal],[c: Upper diagonal],
    Vector f represents b'
    The values in resp. a,b,c vector are identical and known as (-1,2-1)
    """
    n = len(f)                            # Draws nr of eq:s

    bp = list(b)                          # Copy arrays => no change in inputs.
    fp = list(f)
    u = np.zeros(n)                       # Create unknown vector
    
    start = time.clock()
    for i in range(1,n):                # Decomposition
        bp[i] = (i+1)/i                   # Flops: [3*(n-1)]
        fp[i] = fp[i] + fp[i-1]/bp[i-1]
    
    u[n-1] = fp[n-1]/bp[n-1]              # Sub last u value (u[n-1]) [1 flop]

    for i in range(n-2,-1,-1):            # Sub rest of u values
        u[i] = (fp[i] + u[i+1])/bp[i]     # Flops: [2*(n-1)]
        
    end = time.clock()
    print ('CPU time (Special case):',end-start)
    return(u)                             # Returns solution vector x as u