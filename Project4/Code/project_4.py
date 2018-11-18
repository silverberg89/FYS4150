import numpy as np
import methods
import plots
import time
np.random.seed(12)

print('1: Task b (Test of cycles)')
print('2: Task c (Eq plot and accepted flips)')
print('3: Task d (Plot of P(E)')
print('4: Task e (Plot of E,|M|,Cv,X)')
print('5: Task e (Plots of E,|M|,Cv,X) with saved data')
print('6: Task f (Critical temperature)')
task    = input('Choose task:')
t0 = time.clock()

# Parameters:
T       = 2.4 # Initial temperature
J       = 1 # Coupling constant
B       = 1 # Inverse temperature
    
if int(task) == 6:
    ''' Task f '''
    ''' Calculates the estimated critical temperature in two ways. Tc
        holds the value calculated from theory equations, Tc1 is estimated from the
        average critical point temperature for each lattice.
        Use specific heat capacity values for this purpose.'''
    temps = np.load('temps.npy')
    a40   = np.load('vals40.npy')
    b60   = np.load('vals60.npy')
    c80   = np.load('vals80.npy')
    energy_vec          = [a40[:,0],b60[:,0],c80[:,0]]
    magnetism_vec       = [a40[:,6],b60[:,6],c80[:,6]]
    heatcapacity_vec    = [a40[:,7],b60[:,7],c80[:,7]]
    susceptibility_vec  = [a40[:,8],b60[:,8],c80[:,8]]
    N = [40,60,80]
    
    index = []
    Tc1 = 0
    for i in range(len(heatcapacity_vec)):
        maxi = max(heatcapacity_vec[i])
        index.append(heatcapacity_vec[i].tolist().index(maxi))
        Tc1 += temps[index[i]]
    # Tc1 by use of average critical point for each lattice
    Tc1 = Tc1/len(index)
    # Tc by use of equation (20) in theory section
    a = (temps[index[0]] - temps[index[-1]]) / (N[0]**(-1) - N[-1]**(-1))
    Tc = temps[index[-1]] - a*N[-1]**(-1)
    print('By equation:',Tc,'By critical point',Tc1)
    np.save('critical_temps',np.array([Tc1,Tc]))

if int(task) == 5:
    ''' Task e '''
    ''' Uses saved data from:
        Cycles: 10^5
        Temperatures: [2.15...2.33, step 0.03]
        Lattize sizes: [40,60,80,100]
        Plots E, |M|, Cv, X as function of temperature for different L'''
    plots.plot_task_e([],[],[],[],[],0,1)
    
if int(task) == 4:
    ''' Task e '''
    ''' Plots E, |M|, Cv, X as function of temperature for different L
        at the phase transision location. Serial version.'''
    cycles         = 10**2
    L_vec          = [4,6,8,10]
    Temps          = np.round(np.arange(2.15,2.331,0.03),decimals=2) # (1,7)
    #Temps          = Temps[np.newaxis,:]
    Dim            = len(Temps)
    energy         = np.zeros(Dim)
    magnetism      = np.zeros(Dim)
    heatcapacity   = np.zeros(Dim) 
    susceptibility = np.zeros(Dim)
    energy_vec         = []
    magnetism_vec      = []
    heatcapacity_vec   = []
    susceptibility_vec = []
    vec_vals = 0 # 0 = Gives the correct vectors back from the Monte Carlo.
    for i, L in enumerate(L_vec):
        for k, T in enumerate(Temps):
            print(T,L,k)
            print(time.clock()-t0)
            #lattice = methods.lattice(L)  # Random lattice
            lattice = np.ones((L,L))       # Ordered lattice
            E_average,M_average,E2_average,M2_average,E_variance,M_variance,M_absaverage,cv,X = methods.monte_carlo(cycles,lattice,L,J,T,vec_vals)
            energy[k]         = E_average
            magnetism[k]      = M_absaverage
            heatcapacity[k]   = cv
            susceptibility[k] = X
        energy_vec.append(np.copy(energy))
        magnetism_vec.append(np.copy(magnetism))
        heatcapacity_vec.append(np.copy(heatcapacity))
        susceptibility_vec.append(np.copy(susceptibility))
    np.save('e6',energy_vec)
    np.save('m6',magnetism_vec)
    np.save('h6',heatcapacity_vec)
    np.save('s6',susceptibility_vec)
        
    plots.plot_task_e(energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,Temps,Dim,0)

if int(task) == 3:
    ''' Task d '''
    ''' Plots probability of mean energy and collect variances'''
    L                   = 20
    cycles              = 10**6
    lattice = methods.lattice(L)    # Random lattice
    #lattice = np.ones((L,L))       # Ordered lattice
    temps               = [1,2.4]
    energy_collect = []
    energy_vec          = []
    energy_var_vec      = []
    vec_vals = 2 # 2 = Gives the correct vectors back from the Monte Carlo.
    stable_cut = int(10**4)
    if stable_cut >= cycles:
        print('Cycles must be larger than stable_cut')
    for i,T in enumerate(temps):
        energy_eq,E_variance = methods.monte_carlo(cycles,lattice,L,J,T,vec_vals)
        energy_collect.append(energy_eq)
        energy_vec.append(energy_eq[stable_cut:-1])
        energy_var_vec.append(E_variance)
        
    plots.plot_task_d(energy_vec[0],energy_vec[1])

if int(task) == 2:
    ''' Task c '''
    ''' Equlibrium plot for E and |M| as well as accepted flips)'''
    ''' 10**4 cycles = 150 sec computational time '''
    L                   = 20
    cycles              = 10**4
    temps               = [1,2.4]
    energy_vec          = []
    magnetic_vec        = []
    accepted_spins_vec  = []
    vec_vals = 1 # 1 = Gives the correct vectors back from the Monte Carlo.
    for i,T in enumerate(temps):
        for k in range(2):
            if k == 1:
                lattice = methods.lattice(L)    # Random lattice
            else:
                lattice = np.ones((L,L))        # Ordered lattice
            energy_eq,magnetic_eq,accepted_spins = methods.monte_carlo(cycles,lattice,L,J,T,vec_vals)
            energy_vec.append(energy_eq)
            magnetic_vec.append(magnetic_eq)
            accepted_spins_vec.append(accepted_spins)
    np.save('eq_acc_vals'+str(cycles),np.array([energy_vec,magnetic_vec,accepted_spins_vec]))
    plots.plot_task_c(energy_vec,magnetic_vec,accepted_spins_vec)
    
if int(task) == 1:
    ''' Task a and b'''
    ''' Test of cycles and comparisson of analytical values'''
    ''' 2*10**7 cycles as limit = 1236 seconds of computational time'''
    # Calculate analytical values
    # Output as: Energy, Magnetism, Heat Capacity, abs Susceptibility
    analytical_vals = methods.analytical()
    # Set up for different monte carlo cycles
    L              = 2                                        # LxL lattice
    cycles_vec     = [10**3,10**4,10**5,10**6,10**7,2*10**7]  # Monte Carlo cycles
    Dim            = len(cycles_vec)
    energy         = np.zeros(Dim)
    magnetism      = np.zeros(Dim)
    heatcapacity   = np.zeros(Dim) 
    susceptibility = np.zeros(Dim)
    calc_vals = np.zeros((Dim,5))
    vec_vals = 0
    for i, cycles in enumerate(cycles_vec):
        # Create random lattice configuration
        lattice = methods.lattice(L)
        # Activate monte carlo algorithm
        E_average,M_average,E2_average,M2_average,E_variance,M_variance,M_absaverage = methods.monte_carlo(cycles,lattice,L,J,T,vec_vals)
        # Calculated quantities
        energy[i]         = E_average
        magnetism[i]      = M_absaverage
        heatcapacity[i]   = (E_variance / (T**2*L**2))
        susceptibility[i] = (M_variance / (T*L**2))
        calc_vals[i,0] = E_average
        calc_vals[i,1] = M_absaverage
        calc_vals[i,2] = heatcapacity[i]
        calc_vals[i,3] = susceptibility[i]
    # Collect calculated values
    # Output as: Energy, Magnetism, Heat Capacity, Susceptibility, Cycles
    calc_vals[:,-1] = cycles_vec
    np.savetxt('analytical_values_2by2.txt',analytical_vals,fmt='%.10f',header='<E>\t\t<|M|>\t\tCv\t\tChi',delimiter='\t')
    np.savetxt('monte_carlo_cycles_2by2.txt',calc_vals,fmt='%.10f',header='<E>\t\t<|M|>\t\tCv\t\tChi\t\tCycles',delimiter='\t')

print(time.clock()-t0)