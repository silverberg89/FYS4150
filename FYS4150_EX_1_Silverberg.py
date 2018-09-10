# Importing packages------------------------------------------------------
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.linalg as linalg

#-Functions---------------------------------------------------------------
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

def trid_known(b,f):
    """Function that solves [Ax=b] for A beeing a tridiagonal matrix (a,b,c).
    Vectors a,b,c represents A as such:
        [a: lower diagonal],[b: Diagonal],[c: Upper diagonal],
    Vector f represents b
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
    
def error(u,v):
    """Function that calculates relative error between to solutions.
    """
    n = len(u)
    E = np.zeros(n)
    
    for i in range(n):
        E[i] = math.log10(abs((v[i]-u[i])/u[i]))
    return(E)

def LUD(M,f):
    start = time.clock()
    LU = linalg.lu_factor(M)
    x = linalg.lu_solve(LU,f)
    end = time.clock()
    print ('CPU time (LUD case):',end-start)
    return(x)
    
#-Data--------------------------------------------------------------------
exponent = int(input("Enter prefered exponent for your square matrix, [n=10^(exponent)]: "))
n = 10**(exponent)                               # Grid Points
h = 1/(n+1)                                      # Step length

x = np.zeros(n+2)                                # Grid
for i in range(n+2):
    x[i]=i*h

Uknown = np.zeros(n+2)                           # Unknown vectors
Uunknown = np.zeros(n+2)                         # [0] and [n+1] holds B.C
Ulud = np.zeros(n+2)

av = float(input("Enter value of the upper diagonal: "))
bv = float(input("Enter value of the diagonal: "))
cv = float(input("Enter value of the lower diagonal: "))

a = np.ones(n)*(av)                              # Vectors for [Au=f]
b = np.ones(n)*(bv)                              # [A = a,b,c tridiagonaly]
c = np.ones(n)*(cv)
f = (100*(np.exp(-10*x)))*h*h                    # f(x)*step^2

# Solutions---------------------------------------------------------------
Uunknown[1:-1] += trid(a,b,c,f[1:-1])            # General solution
Uknown[1:-1] += trid_known(b,f[1:-1])            # Special solution
Uexact = 1 - (1 - np.exp(-10))*x - np.exp(-10*x) # Exact solution
Uexact[-1] = 0

# Relative error analysis-------------------------------------------------
# Er is just a list that hold the maximum errors for each n, avoiding loop.
# Er is calculated between the general solution and the exact solution.
Error = error(Uexact[1:-1],Uunknown[1:-1])       # Calculate error for n input
Emax = max(Error)                                # Maximum error for n input
Er = [-1.18,-3.09,-5.08,-7.08,-8.84,-6.08,-5.53] # Max errors for n[10,10^7]

h_range = np.zeros(7)                            # log10(h) for n[10,10^7]
for i in range(1,8,1):
    N = 10**i
    H = 1/(N+1)
    h_range[i-1] = np.log10(H)

# LU Decomposition--------------------------------------------------------
M = np.zeros((n, n))                            # Create tridiagonal matrix
M[np.arange(n), np.arange(n)] = b               # Diagonal
M[np.arange(n-1), np.arange(n-1) + 1] = c[0:-1] # Upper diagonal
M[np.arange(n-1) + 1, np.arange(n-1)] = a[0:-1] # Lower diagonal

Ulud[1:-1] += LUD(M,f[1:-1])                    # LUD solution

# Plots-------------------------------------------------------------------
fig2 = plt.figure(2)
plt.plot(h_range,Er)
plt.scatter(h_range,Er, label="Relative error points")
fig2.suptitle('Relative error (General vs Exact)', fontsize=16)
plt.xlabel('log10(h)', fontsize=12)
plt.ylabel('E', fontsize=12)
plt.legend()
plt.show()

fig1 = plt.figure(1)
plt.plot(x, Uexact, 'red', label="Exact solution")
plt.scatter(x, Ulud, marker="s", linewidth=3, label="LUD solution")
plt.scatter(x, Uunknown, marker="o", linewidth=1, label="General numerical solution")
plt.scatter(x, Uknown, marker="d", linewidth=0.5, label="Special numerical solution")
fig1.suptitle('Solutions for: [−u´´(x) = 100e^(-10x)]', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('u(x)', fontsize=12)
plt.legend()
plt.show()