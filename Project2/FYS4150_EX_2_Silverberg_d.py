# Importing packages------------------------------------------------------
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#-Functions---------------------------------------------------------------
start_time = time.clock()

def func(x, a,b,c):
    return a+b*x+x**c

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

loop = 10
M_itt = np.zeros(loop-2)
M_n = np.zeros(loop-2)
M_t = np.zeros(loop-2)
M_tnp = np.zeros(loop-2)

pmax_1 = np.zeros((loop-2,3))
pmax_2 = np.zeros((loop-2,3))
pmax_4 = np.zeros((loop-2,3))
pmax_6= np.zeros((loop-2,3))
pmax_8 = np.zeros((loop-2,3))

for k in range (3,loop):
    it = 0
    for r in (1,2,3,4,5):
        it +=1
        #-Data--------------------------------------------------------------------
        pmin = 0            # Start of interval for dimension variable p
        pmax = r          # End of interval for dimension variable p
        n = k               # Mesh points
        h = (pmax-pmin)/n   # Step size
        
        p = np.zeros(n)                          # Grid vector
        d = np.zeros(n)                          # Diag
        V = np.zeros(n)                          # Harmonic oscillator potential
        for i in range(n):
            p[i] = pmin+i*h
            V[i] = p[i]**2
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
        M_tnp[k-2], Ev_np, E_np = eignumpy(A)    # Calculate eigenpairs with np
        
        # Jacobis Method--------------------------------------------------------
        ts = time.clock()
        I = 0
        J = 0
        itt = 0                               # Nr of transformations
        max_ov = 1
        myzero = 10**(-8)                     # Chooses a tolerance (My zero def)
        while max_ov >= myzero:
            itt += 1 
            max_ov,I,J = max_off(A,I,J)       # Find index of abs.max value of A
            #unit_test_1(A,max_ov,myzero,I,J)  # Unit test 1 (Largest off diag)
            E,Ev = jacubi(A,E,I,J,n)          # Calculate eigenvectors with Jacubi
            #unit_test_2(E,myzero)             # Unit teset 2 (Ortogonality)
        tse = time.clock()
        
        #unit_test_3(Ev,Ev_np,myzero,n)       # Unit test 3 (Correct eigenvalues)
        
        Ev = np.sort(Ev)
        Ev_np = np.sort(Ev_np)
    
        # Comutational measurements---------------------------------------------
        M_itt[k-2] = itt                      # Comutational measurements
        M_n[k-2] = n
        M_t[k-2] = tse-ts
        #popt, pcov = curve_fit(func, M_n, M_itt) # Fit Transformations
        #popt1, pcov1 = curve_fit(func, M_n[15:-1], M_t[15:-1]) Fit Time
        
        # Measure how results behave with different Pmax and nr of N-----------
        Ev_an = [3,7,11,15]                   # Analytical solutions
        
        if it == 1:
            pmax_1[k-2,0] = Ev_an[0]-Ev_np[0]
            pmax_1[k-2,1] = Ev_an[1]-Ev_np[1]
            pmax_1[k-2,2] = Ev_an[2]-Ev_np[2]
        if it == 2:
            pmax_2[k-2,0] = Ev_an[0]-Ev_np[0]
            pmax_2[k-2,1] = Ev_an[1]-Ev_np[1]
            pmax_2[k-2,2] = Ev_an[2]-Ev_np[2]
        if it == 3:
            pmax_4[k-2,0] = Ev_an[0]-Ev_np[0]
            pmax_4[k-2,1] = Ev_an[1]-Ev_np[1]
            pmax_4[k-2,2] = Ev_an[2]-Ev_np[2]
        if it == 4:
            pmax_6[k-2,0] = Ev_an[0]-Ev_np[0]
            pmax_6[k-2,1] = Ev_an[1]-Ev_np[1]
            pmax_6[k-2,2] = Ev_an[2]-Ev_np[2]
        if it == 5:
            pmax_8[k-2,0] = Ev_an[0]-Ev_np[0]
            pmax_8[k-2,1] = Ev_an[1]-Ev_np[1]
            pmax_8[k-2,2] = Ev_an[2]-Ev_np[2]     

# Nr of transformations vs square matrix dimension----------------------------
''' Based on registred observations: itt, n and time
    the relationship is an exponential growth of needed transformations
    as N increases for NxN square matrices, hence computation time increases
    exponential too, so the method is not suitible for large matrices.'''

fig6 = plt.figure(6)
plt.scatter(M_n,M_itt, label="F(n) Jacobi Method", marker='o')
plt.xlabel('Square matrix dimension [n]', fontsize=12)
plt.ylabel('Transformationsn[nr]', fontsize=12)
plt.legend()
plt.title('Transformations vs Dimensions')
plt.show()
fig6.savefig('Nr of transformations behavior_T.png',figsize=(3.000, 2.000), dpi=150)

fig7 = plt.figure(7)
plt.scatter(M_n,M_t, label="F(n) Jacobi Method", marker='o')
plt.scatter(M_n,M_tnp, label="F(n) Numpy", marker='d')
plt.xlabel('Square matrix dimension [n]', fontsize=12)
plt.ylabel('Computational time [s]', fontsize=12)
plt.legend()
plt.tight_layout()
plt.title('Transformations vs Dimensions (Time)')
plt.show()
fig7.savefig('Nr of transformations behavior_Time.png',figsize=(3.000, 2.000), dpi=150)

# Error against eigenvalues--------------------------------------------------
''' For low values of Pmax, the number of int.points do not yield more
    precision, the error keeps increasing until a certain value.
    For higher values of Pmax the number of itterations matter, and gives
    succesive smaller errors as n grows. Also the errors generally
    increases for higher order eigenvalues. Meaning that, even with increasing
    int.points, we will never reach the exact analytical results. At best
    we could find the closest eigenvalues with the optimal Pmax.
    For Pmax larger than 3 ? the precision increases as a function of
    the resolution int.points but decreases as a function of Pmax.'''
    
x = np.zeros(n-1)

fig8 = plt.figure(8,figsize=(3.000, 2.000), dpi=100)
plt.plot(M_n,x, linestyle='--')
plt.plot(M_n,pmax_1[:,0], label="Eigenvalue 1")
plt.plot(M_n,pmax_1[:,1], label="Eigenvalue 2")
plt.plot(M_n,pmax_1[:,2], label="Eigenvalue 3")
plt.xlabel('Square matrix dimension [n]', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend()
plt.title('Pmax = 1')
plt.show()
fig8.savefig('Pmax_1-N50.png', dpi=100)

# =============================================================================
# fig9 = plt.figure(9)
# plt.plot(M_n,x, linestyle='--')
# plt.plot(M_n,pmax_2[:,0], label="Eigenvalue 1")
# plt.plot(M_n,pmax_2[:,1], label="Eigenvalue 2")
# plt.plot(M_n,pmax_2[:,2], label="Eigenvalue 3")
# plt.xlabel('Square matrix dimension [n]', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.legend()
# plt.title('Pmax = 2')
# plt.show()
# fig9.savefig('Nr of transformations behavior.png')
# =============================================================================

fig10 = plt.figure(10,figsize=(3.000, 2.000), dpi=100)
plt.plot(M_n,x, linestyle='--')
plt.plot(M_n,pmax_4[:,0], label="Eigenvalue 1")
plt.plot(M_n,pmax_4[:,1], label="Eigenvalue 2")
plt.plot(M_n,pmax_4[:,2], label="Eigenvalue 3")
plt.xlabel('Square matrix dimension [n]', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend()
plt.title('Pmax = 4')
plt.show()
fig10.savefig('Pmax_4-N50.png', dpi=100)

# =============================================================================
# fig11 = plt.figure(11)
# plt.plot(M_n,x, linestyle='--')
# plt.plot(M_n,pmax_6[:,0], label="Eigenvalue 1")
# plt.plot(M_n,pmax_6[:,1], label="Eigenvalue 2")
# plt.plot(M_n,pmax_6[:,2], label="Eigenvalue 3")
# plt.xlabel('Square matrix dimension [n]', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.legend()
# plt.title('Pmax = 6')
# plt.show()
# fig11.savefig('Nr of transformations behavior.png')
# =============================================================================

fig12 = plt.figure(12,figsize=(3.000, 2.000), dpi=100)
plt.plot(M_n,x, linestyle='--')
plt.plot(M_n,pmax_8[:,0], label="Eigenvalue 1")
plt.plot(M_n,pmax_8[:,1], label="Eigenvalue 2")
plt.plot(M_n,pmax_8[:,2], label="Eigenvalue 3")
plt.xlabel('Square matrix dimension [n]', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend()
plt.title('Pmax = 8')
plt.show()
fig12.savefig('Pmax_8-N50.png', dpi=100)
#----------------------------------------------------------------------------
end_time = time.clock()
print(end_time-start_time)