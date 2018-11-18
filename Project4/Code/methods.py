import numpy as np
np.random.seed(12)

def lattice(n):
    '''Create a nxn lattice with random spin configuration
        Output: A lattice as matrix'''
    lattice = np.random.choice([1, -1],size=(n,n))
    return (lattice)

def dE(old_config, new_config, J):
    '''Energy difference for a new configuration of spins
        Input old matrix, new matrix and coupling constant
        Outputs the energy difference between them'''
    f = 2 * old_config * (J * new_config)
    return (f)

def periodic_bc(i,limit,add):
    ''' Choose correct matrix index with periodic boundary conditions
        Input: i = Base index, limit = Highest index, add = Number to add or subtract from i'''
    f = (i+limit+add) % limit
    return (f)
    
def calc_energy(L,lattice,E):
    ''' Takes in size of lattice, the lattice and initial energy
        Outputs the energy for the lattice'''
    for j in range(L): 
        for i in range(L):
            E -= lattice.item(i,j) * (lattice.item(periodic_bc(i,L,-1),j) + lattice.item(i,periodic_bc(j,L,1)))
    return(E)

def monte_carlo(cycles,lattice,L,J,T,vec_vals):
    ''' Algorithm that simulate the behaviour of a magnetic phase change
        Input: Number of cycles, lattice with spins, size of lattice (L)
        ccoupling constant (J), temperature (T) and choice of return values
        Output: Mean energies, mean magnetisms as well as squared quantities'''
    E               = calc_energy(L,lattice,0)  # Initial energy
    M               = np.sum(lattice)           # Initial magnetism
    N               = L**2                      # Nr of spins
    E_average       = 0
    M_average       = 0
    E2_average      = 0
    M2_average      = 0
    M_absaverage    = 0
    counter         = 0
    accepted_spins  = 0
    energy_eq       = np.zeros(cycles)
    magnetic_eq     = np.zeros(cycles)
    acc_spin_vec    = np.zeros(cycles)
    # Start Metropolis algorithm
    for u in range(cycles):
        counter +=1
        for k in range(N):
            # Flip one spin at the time
            i = np.random.randint(L)
            j = np.random.randint(L)
            new_config = lattice[(i-1) % L,j] + lattice[(i+1) % L,j] + lattice[i,(j-1) % L] + lattice[i,(j+1) % L]
            # If the energy difference is negative or by probability pass, new config = ok
            delta_energy = int(dE(lattice[i,j],new_config,J))
            boltz        = np.exp(-delta_energy / T)
            if delta_energy < 0 or np.random.random() < boltz:
                accepted_spins +=1
                lattice[i,j]    = -lattice[i,j]
                E               = np.float64(E) + np.float64(delta_energy)
                M               = np.float64(M) + np.float64((2 * lattice[i,j]))
        # Update values
        E_average       = np.float64(E_average + np.float64(E))
        M_average       = np.float64(M_average + np.float64(M))
        E2_average      = np.float64(E2_average + np.float64(E**2))
        M2_average      = np.float64(M2_average + np.float64(M**2))
        M_absaverage    = np.float64(M_absaverage + np.float64(np.abs(M)))
        energy_eq[u]    = E_average / counter / N
        magnetic_eq[u]  = M_absaverage / counter / N
        acc_spin_vec[u] = accepted_spins
        
    if vec_vals == 1:
        return(energy_eq,magnetic_eq,acc_spin_vec)
        
    # Average the values
    E_average     /= cycles
    M_average     /= cycles
    E2_average    /= cycles
    M2_average    /= cycles
    M_absaverage  /= cycles
    #Calculate variances
    E_variance  = (E2_average-E_average**2)
    M_variance  = (M2_average-M_absaverage**2)
    heatcapacity   = (E_variance / (T**2*L**2))
    susceptibility = (M_variance / (T*L**2))
    #Normalize lattice values to point values (E/N), |M|/N)
    E_average     /= N
    M_average     /= N
    M_absaverage  /= N

    if vec_vals == 2:
        return(energy_eq,E_variance)
    
    return(E_average,M_average,E2_average,M2_average,E_variance,M_variance,M_absaverage,heatcapacity,susceptibility)
    
def analytical():
    ''' Take in parameters for a 2x2 lattice, output is the analytical
        values for the lattice:s properties.'''
    T = 1       # Temperature
    L = 2       # Matrix dimension
    J = 1       # Coupling constant
    B = 1       # Inverse temperature
    N = L**2    # Nr of Spins
    
    Z              = (np.cosh(8*J*B)+3)/4
    E_ord_analytic = (-2*J*np.sinh(8*J*B)) / Z
    M_abs_analytic = (0.5*np.exp(8*J*B)+0.5) / Z
    
    E_var_analytic = (16*J**2*np.cosh(8*J*B)) / Z -  E_ord_analytic**2
    M_var_analytic = (2*np.exp(8*J*B)+0.25) / Z
    
    Cv_analytic    = (E_var_analytic / B*T)
    X_analytic     = (M_var_analytic / B)
    X_abs_analytic = ((2*np.exp(8) +2) - ((np.exp(8)+2)**2)/(4*Z))/Z
    
    analytical_vals = np.array([E_ord_analytic/N,M_abs_analytic/N,Cv_analytic/N,X_abs_analytic/N])
    analytical_vals = analytical_vals[:, np.newaxis]
    return(analytical_vals.T)