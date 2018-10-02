# Importing packages------------------------------------------------------
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

#-Functions---------------------------------------------------------------
start_time = time.clock()

def unit_test_1(A,max_ov,myzero,I,J):
    ''' Evaluates if the largest non-diagonal value is found with np'''
    S = np.copy(A)
    np.fill_diagonal(S,0)
    S = abs(S.flatten())
    maxv = max(S)
    if abs(maxv - max_ov) > myzero:         # Difference should be zero
        raise Exception('Unit test 1: Failed at element:',I,J)
    else:
        print('Unit test 1: Passed')
    return()
    
def unit_test_2(E,myzero):
    '''Evaluates if the the rotation preserves orthogonality'''
    k = len(E)
    for i in range(0,k-1):
        dotp = np.dot(E[:,i],E[:,i+1])
        if dotp > myzero:                   # Dotproduct = 0 => Ortongoality
            raise Exception('Unit test 2: Failed at dimension:',k)
        else:
            print('Unit test 2: Passed')
    return()
    
def unit_test_3(Ev,Ev_np,myzero,n):
    '''Evaluates if the algorithm gives accurate eigenvalues against np'''
    Ev = np.sort(Ev)
    Ev_np = np.sort(Ev_np)
    for i in range(len(Ev)):
        if Ev[i]-Ev_np[i] > myzero:         # Dieffernece should be zero
            raise Exception('Unit test 3: Failed at dimension:',n)
        else:
            print('Unit test 3: Passed')
    return()  

def max_off(A,I,J):
    ''' Finds the maximum value of the off-diagonals in matrix A
        by observing the absolute value at each element on the right of
        the diagonal. I:E for a 3x3 matrix: [(0.1),(0,2),(1,2)].
        For each observation it evaluates the value against the standing
        maximum value 'max_ov'. If the observed element value is larger,
        then it overwrite 'max_ov' with the observed value.
        After all elements are evaluated it returns the indexs (I,J)
        where it was found in the matrix A, and the maximum value itself.
    '''
    max_ov = 0
    for i in range(len(A)):
        for j in range(i+1, len(A)): # Only search one side (i+1) due to sym.
            if i != j:
                Aij = abs(A[j][i])          # Since symmertry
                if Aij > max_ov:
                    max_ov = Aij
                    I = i
                    J = j
    return (max_ov,I,J)

def jacubi(A,E,k,l,n):
    ''' Calculates the eigenvectors of matrix A
        The Jacobi method eliminates the largest off-diagonal (symmetrical)
        elements, A[k, l] and A[l, k], found from the maxoffdiag function,
        one by one until A is our defined diagonal matrix holding the eigvals.
    '''
    # -----Calculate quantities in B for solving [B=S(k,l,θ)'*A*S(k,l,θ)]
    # Lecture slide:
    # http://compphysics.github.io/ComputationalPhysics/doc/pub/eigvalues/html/._eigvalues-bs020.html
    t = 0
    T = (A[l,l]-A[k,k])/(2*A[k,l])
    c = 1/(math.sqrt(1+t*t))
    s = c*t
    if (A[k,l] != 0.0):                       # See if element is already zero
        T = (A[l,l]-A[k,k])/(2*A[k,l])        # T in tan(theta)=-T(+/-)sqrt(1+T^2)
        if T >= 0:                            # Multiply by conjugate =>
            t = 1/(T+math.sqrt(1+T*T))        # No loss of num.prec when T => inf
        else:
            t = -1/(-T+math.sqrt(1+T*T))
        c = 1/(math.sqrt(1+t*t))              # C = cos(theta) = 1/sqrt(1+T^2)
        s = c*t                               # S = sin(theta) = t*C

    # -----Rotating matrix
    # Algorithm translated from lecture slide:
    # http://compphysics.github.io/ComputationalPhysics/doc/pub/eigvalues/html/._eigvalues-bs029.html
    a_kk = A[k,k]                             # Cementing initial diag values
    a_ll = A[l,l]                             # By symmetry
    A[k,k] = c*c*a_kk - 2.0*c*s*A[k,l] + s*s*a_ll # Assign new diag values
    A[l,l] = s*s*a_kk + 2.0*c*s*A[k,l] + c*c*a_ll
    A[k,l] = 0.0                              # Set largest max.off.d to 0
    A[l,k] = 0.0                              # By symmetry
    
    for i in range(n):                        # Transform rest of matrix
        if i != k and i != l:
            a_ik = A[i,k]                     # Cementing initial off.d values
            a_il = A[i,l]                     # By symmetry
            A[i,k] = c*a_ik - s*a_il          # Assign new off.d values
            A[k,i] = A[i,k]                   # By symmetry
            A[i,l] = c*a_il + s*a_ik
            A[l,i] = A[i,l]

        r_ik = E[i,k]                         # Rotating the E matrix
        r_il = E[i,l]                         # (Eigenvectors found in E)

        E[i,k] = c*r_ik - s*r_il
        E[i,l] = c*r_il + s*r_ik
        Ev = np.diag(A)                       # Draws eigenvals from diag(E)

    return(E,Ev)
    
def eignumpy(A):
    ''' Calculates eigenpairs with numpy'''
    ts = time.clock()
    Ev_np = np.linalg.eigvals(A)              # Eigenvalue with numpy
    E_np = np.linalg.eig(A)                   # Eigenvector with numpy
    E_np = E_np[1]
    tse = time.clock()
    timec = tse-ts
    return(timec,Ev_np,E_np)
    
# End of def--------------------------------------------------------------
N=100
wr = [0.01,0.5,1,5]                          # Strength of oscillator potential
for l in range(len(wr)):
    #-Data--------------------------------------------------------------------
    pmin = 0            # Start of interval for rho
    pmax = 10            # End of interval for rho
    n = N-2             # Mesh points
    h = (pmax-pmin)/n   # Step size
    
    w = wr[l]                                # Characteristic frequency
    d = np.zeros(n)                          # Diagonal vector
    V = np.zeros(n)                          # Potential vector
    p = np.zeros(n)                          # Rho (Dimensionless)
    for i in range(0,n):
        p[i] = pmin+(i+1)*h
        #V[i] = w*p[i]**2                     # Non interacting potential (Only Harmonic oscillator)
        V[i] = (w**2*p[i]**2)+(1/p[i])       # Interacting potential (Harmonic oscillator + Coloumb)
        d[i] = (2/h**2)+V[i]
    a = -1/(h**2)                            # Upper/Lower diag
    
    E = np.eye(n)                            # Eigenvector matrix
    A = np.zeros((n, n))                     # Create tridiagonal matrix A
    for i in range(len(d)-1):
        A[i,i] = d[i]
        A[i+1,i] = a
        A[i,i+1] = a
    A[-1,-1] = d[-1] 
    Acopy = A.copy()                         # Copy original A matrix
    
    # Numpy Method----------------------------------------------------------
    tid, Ev_np, E_np = eignumpy(A)           # Calculate eigenpairs with np
    
# =============================================================================
#     # Jacobis Method--------------------------------------------------------
#     ts = time.clock()
#     I = 0
#     J = 0
#     itt = 0                               # Nr of transformations
#     max_ov = 1
#     myzero = 10**(-8)                     # Chooses a tolerance (My zero def)
#     while max_ov >= myzero:
#         itt += 1 
#         max_ov,I,J = max_off(A,I,J)       # Find index of abs.max value of A
#         #unit_test_1(A,max_ov,myzero,I,J)  # Unit test 1 (Largest off diag)
#         E,Ev = jacubi(A,E,I,J,n)          # Calculate eigenvectors with Jacubi
#         #unit_test_2(E,myzero)             # Unit teset 2 (Ortogonality)
#     tse = time.clock()
#     #unit_test_3(Ev,Ev_np,myzero,n)        # Unit test 3 (Correct eigenvalues)
# =============================================================================
    
    # Boundary conditions--------------------------------------------------
    radius = p                        # p = (1/alpha)*r (Choose alpha = 1)
    radius = np.insert(radius,0,0,axis=0)
    radius = np.insert(radius,len(radius),pmax+h,axis=0)
    Eigvs = np.insert(Ev_np,0,0,axis=0)
    Eigvs = np.insert(Eigvs,len(Eigvs),0,axis=0)
    Eigve = np.insert(E_np, 0, 0, axis=0)
    Eigve = np.insert(Eigve, len(Eigve), 0, axis=0)
    
# =============================================================================
#     radius = p                        # p = (1/alpha)*r (Choose alpha = 1)
#     radius = np.insert(radius,0,0,axis=0)
#     radius = np.insert(radius,len(radius),pmax+h,axis=0)
#     Eigvs = np.insert(Ev,0,0,axis=0)
#     Eigvs = np.insert(Eigvs,len(Eigvs),0,axis=0)
#     Eigve = np.insert(E, 0, 0, axis=0)
#     Eigve = np.insert(Eigve, len(Eigve), 0, axis=0)
# =============================================================================
    
    # Plot results---------------------------------------------------------
    ''' Eigenvectors represents the radial probability.
        r represents the radius. As seen the probability decreases
        as the potential grows, choose rho such that information don´t
        get lost beyond a certain distance'''
    ind = Ev_np.argsort()                 # Sort low-high such that indexes remains
    Eigvs = Eigvs[ind]                 # Sort Eigenvalues
    Eigve = Eigve[:,ind]               # Shuffle Eigvecs corresponding to Eigenvals
    Eigve1 = Eigve[:,0]                # Draws the first eigenvector
    Eigve1 = Eigve1 / np.linalg.norm(Eigve1) # Normalizes
    fig1 = plt.figure(1)
    plt.plot(radius,Eigve1**2,label='%s Wr' % wr[l])
    plt.xlabel('rho')
    plt.ylabel('Radial probability [u(rho)^(2)]')
    plt.title('Probability distribution for ground state (Non-Interacting)')
    plt.legend()
    fig1.savefig('Probability distribution_N40_Pmax4.3_Np.png')
    
plt.show()
end_time = time.clock()
print(end_time-start_time)