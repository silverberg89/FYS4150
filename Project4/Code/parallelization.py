import sys
import numpy as np
from multiprocessing import Pool
import time
import plots
from methods import monte_carlo
np.random.seed(12)
# --------------------------------------------------
# Begin fix
# are we running inside Blender?
bpy = sys.modules.get("bpy")
if bpy is not None:
    sys.executable = bpy.app.binary_path_python
    # get the text-block's filepath
    __file__ = bpy.data.texts[__file__[1:]].filepath
del bpy, sys
# end fix!
# --------------------------------------------------

def task_e():
    t0 = time.clock()
    ''' Task e '''
    ''' Plots E, |M|, Cv, X as function of temperature for different L
        with usage of parallalization. It uses 7 threads for calculating
        7 different temperatures at the same time.'''
    T              = np.round(np.arange(2.15,2.331,0.03),decimals=2) # (1,7)
    T = T[np.newaxis,:]
    fgasd         = Pool(7)         # Need to change this name sometimes..
    cycles         = 10**2
    L_vec          = [40,60,80,100]
    L_vec          = [2,4,6,8]
    J              = 1
    vec_vals       = 0 # 0 = Gives the correct vectors back from the Monte Carlo.
    Dim            = len(T)
    dim            = len(T[0])
    energy         = np.zeros((Dim,dim))
    magnetism      = np.zeros((Dim,dim))
    heatcapacity   = np.zeros((Dim,dim)) 
    susceptibility = np.zeros((Dim,dim))
    energy_vec         = []
    magnetism_vec      = []
    heatcapacity_vec   = []
    susceptibility_vec = []
    vals_holder = []
    print('Parameters:')
    print('Cycles: ',cycles,'Temperatures:',T)
    print('Lattize sizes:',T)
    for i, L in enumerate(L_vec):
        for k in range(Dim):
            lattice = np.ones((L,L))       # Ordered lattice
            # Following line gives a set of temperatures to monte carlo algo.
            vals = fgasd.starmap(monte_carlo,[(cycles,lattice,L,J,T[k,0],vec_vals),(cycles,lattice,L,J,T[k,1],vec_vals),(cycles,lattice,L,J,T[k,2],vec_vals),(cycles,lattice,L,J,T[k,3],vec_vals),(cycles,lattice,L,J,T[k,4],vec_vals),(cycles,lattice,L,J,T[k,5],vec_vals),(cycles,lattice,L,J,T[k,6],vec_vals)])
            vals = np.array(vals)
            np.save('vals'+str(L),vals)
            print('Time:',np.round(time.clock()-t0,3),'Size: ',L)
            # Order data
            # 0 = E, 1 = M, 2 = E2, 3 = M2, 4 = E_var, 5 = M_var, 6 = abs(M), 7 = Cv, 8 = X
            for j in range(dim):
                energy[k,j]         = vals[j,0]
                magnetism[k,j]      = vals[j,6]
                heatcapacity[k,j]   = vals[j,7]
                susceptibility[k,j] = vals[j,8]
        energy_vec.append(np.copy(energy))
        magnetism_vec.append(np.copy(magnetism))
        heatcapacity_vec.append(np.copy(heatcapacity))
        susceptibility_vec.append(np.copy(susceptibility))
        vals_holder.append(np.copy(vals))
        
    plots.plot_task_e(energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,T,Dim)
    fgasd.close()
    fgasd.join()
    return(energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,vals_holder,T)  
    
if __name__=='__main__':
    __spec__ = None
    # Create the pool
    cvpirate = Pool(7)
    time.sleep(2)
    cvpirate.close()
    cvpirate.join()
    energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,vals_holder,T = task_e()
