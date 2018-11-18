import sys
import numpy as np
from multiprocessing import Pool
from multiprocessing import Process, JoinableQueue
import os
import time
from methods import monte_carlo
import matplotlib.pyplot as plt
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

def plot_load():
    ''' Take in data from three vectors [energy,magnetism,acceptance]
        each cathegory holds [orderedT1,disorderedT1,orderedT2.4,disorderedT2.4]
        outputs plots of the data and saves to file '''
    temperatures   = np.round(np.arange(2.15,2.331,0.03),decimals=2)
    Dim = len(temperatures)
    a40 = np.load('vals40.npy')
    b60 = np.load('vals60.npy')
    c80 = np.load('vals80.npy')
    c100 = np.load('vals100.npy')
    energy_vec          = [a40[:,0],b60[:,0],c80[:,0],c100[:,0]]
    magnetism_vec       = [a40[:,6],b60[:,6],c80[:,6],c100[:,6]]
    heatcapacity_vec    = [a40[:,7],b60[:,7],c80[:,7],c100[:,7]]
    susceptibility_vec  = [a40[:,8],b60[:,8],c80[:,8],c100[:,8]]
        
    plt.figure(1)
    plt.xlabel('Temperature')
    plt.ylabel('E/N')
    plt.title('Mean Energy vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temperatures,energy_vec[0], label='L=40')
    plt.plot(temperatures,energy_vec[1], label='L=60')
    plt.plot(temperatures,energy_vec[2], label='L=80')
    plt.plot(temperatures,energy_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_energy.png', bbox_inches='tight')
    
    plt.figure(2)
    plt.xlabel('Temperature')
    plt.ylabel('|M|/N')
    plt.title('Mean Magnetism vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temperatures,magnetism_vec[0], label='L=40')
    plt.plot(temperatures,magnetism_vec[1], label='L=60')
    plt.plot(temperatures,magnetism_vec[2], label='L=80')
    plt.plot(temperatures,magnetism_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_magnetic.png', bbox_inches='tight')
    
    plt.figure(3)
    plt.xlabel('Temperature')
    plt.ylabel('Cv/N')
    plt.title('Heatcapacity vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temperatures,heatcapacity_vec[0], label='L=40')
    plt.plot(temperatures,heatcapacity_vec[1], label='L=60')
    plt.plot(temperatures,heatcapacity_vec[2], label='L=80')
    plt.plot(temperatures,heatcapacity_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_heatcapacity.png', bbox_inches='tight')
    
    plt.figure(4)
    plt.xlabel('Temperature')
    plt.ylabel('X/N')
    plt.title('Susceptibility vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temperatures,susceptibility_vec[0], label='L=40')
    plt.plot(temperatures,susceptibility_vec[1], label='L=60')
    plt.plot(temperatures,susceptibility_vec[2], label='L=80')
    plt.plot(temperatures,susceptibility_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_susceptibility.png', bbox_inches='tight')
    
    plt.show()
    return(energy_vec)

def plot_task_e(energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,temperatures,Dim):
    ''' Take in data from three vectors [energy,magnetism,acceptance]
        each cathegory holds [orderedT1,disorderedT1,orderedT2.4,disorderedT2.4]
        outputs plots of the data and saves to file '''
    for i in range(len(energy_vec)):
        energy_vec[i] = np.array(energy_vec[i]).flatten()
        magnetism_vec[i] = np.array(magnetism_vec[i]).flatten()
        heatcapacity_vec[i] = np.array(heatcapacity_vec[i]).flatten()
        susceptibility_vec[i] = np.array(susceptibility_vec[i]).flatten()
    temperatures = temperatures.flatten()
        
    plt.figure(1)
    plt.xlabel('Temperature')
    plt.ylabel('E/N')
    plt.title('Mean Energy vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temperatures,energy_vec[0], label='L=40')
    plt.plot(temperatures,energy_vec[1], label='L=60')
    plt.plot(temperatures,energy_vec[2], label='L=80')
    plt.plot(temperatures,energy_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_energy.png', bbox_inches='tight')
    
    plt.figure(2)
    plt.xlabel('Temperature')
    plt.ylabel('|M|/N')
    plt.title('Mean Magnetism vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temperatures,magnetism_vec[0], label='L=40')
    plt.plot(temperatures,magnetism_vec[1], label='L=60')
    plt.plot(temperatures,magnetism_vec[2], label='L=80')
    plt.plot(temperatures,magnetism_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_magnetic.png', bbox_inches='tight')
    
    plt.figure(3)
    plt.xlabel('Temperature')
    plt.ylabel('Cv/N')
    plt.title('Heatcapacity vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temperatures,heatcapacity_vec[0], label='L=40')
    plt.plot(temperatures,heatcapacity_vec[1], label='L=60')
    plt.plot(temperatures,heatcapacity_vec[2], label='L=80')
    plt.plot(temperatures,heatcapacity_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_heatcapacity.png', bbox_inches='tight')
    
    plt.figure(4)
    plt.xlabel('Temperature')
    plt.ylabel('X/N')
    plt.title('Susceptibility vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temperatures,susceptibility_vec[0], label='L=40')
    plt.plot(temperatures,susceptibility_vec[1], label='L=60')
    plt.plot(temperatures,susceptibility_vec[2], label='L=80')
    plt.plot(temperatures,susceptibility_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_susceptibility.png', bbox_inches='tight')
    
    plt.show()
    return(energy_vec)
    

def task_e():
    t0 = time.clock()
    ''' Task e '''
    ''' Plots E, |M|, Cv, X as function of temperature for different L'''
    T              = np.round(np.arange(2.15,2.331,0.03),decimals=2) # (1,7)
    T = T[np.newaxis,:]
    fgasd         = Pool(7)
    
    cycles         = 10**5
    L_vec          = [2,100]
    #L_vec          = [2,4,6,8]
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
    for i, L in enumerate(L_vec):
        for k in range(Dim):
            lattice = np.ones((L,L))       # Ordered lattice
            vals = fgasd.starmap(monte_carlo,[(cycles,lattice,L,J,T[k,0],vec_vals),(cycles,lattice,L,J,T[k,1],vec_vals),(cycles,lattice,L,J,T[k,2],vec_vals),(cycles,lattice,L,J,T[k,3],vec_vals),(cycles,lattice,L,J,T[k,4],vec_vals),(cycles,lattice,L,J,T[k,5],vec_vals),(cycles,lattice,L,J,T[k,6],vec_vals)])
            vals = np.array(vals)
            np.save('vals'+str(L),vals)
            print('Time:',np.round(time.clock()-t0,3),'Size: ',L)
            for j in range(dim):
                energy[k,j]         = vals[j,0]
                magnetism[k,j]      = vals[j,6]
                heatcapacity[k,j]   = vals[j,7]
                susceptibility[k,j] = vals[j,8]
                # 0 = E, 1 = M, 2 = E2, 3 = M2, 4 = E_var, 5 = M_var, 6 = abs(M), 7 = Cv, 8 = X
        energy_vec.append(np.copy(energy))
        magnetism_vec.append(np.copy(magnetism))
        heatcapacity_vec.append(np.copy(heatcapacity))
        susceptibility_vec.append(np.copy(susceptibility))
        vals_holder.append(np.copy(vals))
        
    test_array = plot_task_e(energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,T,Dim)
    fgasd.close()
    fgasd.join()
    return(energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,vals_holder,test_array,T)
    

# best practice to go through main for multiprocessing
if __name__=='__main__':
    __spec__ = None
    # create the pool
# =============================================================================
#     cvpirate = Pool(7)
#     time.sleep(2)
#     cvpirate.close()
#     cvpirate.join()
#     energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,vals,test,T = task_e()
# =============================================================================
    plot_load()
# =============================================================================
# if __name__ == '__main__':
#     __spec__ = None
#     #T              = np.round(np.arange(2.15,2.331,0.03),decimals=2) # (1,8)
#     energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,vals,test,T = task_e()
# =============================================================================
