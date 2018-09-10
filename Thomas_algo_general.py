def trid(a, b, c, f):
    """Function that solves [Ax=b'] for A beeing a tridiagonal matrix (a,b,c).
    Vectors a,b,c represents A as such:
        [a: lower diagonal],[b: Diagonal],[c: Upper diagonal],
    Input vector f represents b'
    The values in a,b,c are unknown.
    """
    n = len(f)                            # Draws nr of eq:s
    
    ap = list(a)                          # Copy arrays => no change in main.
    bp = list(b)
    cp = list(c)
    fp = list(f)
    u = np.zeros(n)                       # Create unknown vector
    
    start = time.clock()
    for i in range(1,n):                 # Decomposition
        m = ap[i]/bp[i-1]                 # Flops: [5*(n-1)]
        bp[i] = bp[i] - m*cp[i-1]
        fp[i] = fp[i] - m*fp[i-1]

    u[n-1] = fp[n-1]/bp[n-1]              # Sub last u value (u[n-1]) [1 flop]
    for i in range(n-2, -1, -1):          # Sub rest of u values ()
        u[i] = (fp[i]-cp[i]*u[i+1])/bp[i] # Flops: [3*(n-1)]
    
    end = time.clock()
    print ('CPU time (General case):',end-start)
    return u                              # Returns solution vector x as u