import matplotlib.pyplot as plt
import numpy as np

def plot_task_e(energy_vec,magnetism_vec,heatcapacity_vec,susceptibility_vec,temps,Dim,load):
    ''' Take in data from three vectors [energy,magnetism,acceptance]
        each cathegory holds [orderedT1,disorderedT1,orderedT2.4,disorderedT2.4]
        outputs plots of the data and saves to file '''
    
    if load == 1:
        temps   = np.round(np.arange(2.15,2.331,0.03),decimals=2)
        a40     = np.load('vals40.npy')
        b60     = np.load('vals60.npy')
        c80     = np.load('vals80.npy')
        c100    = np.load('vals100.npy')
        energy_vec          = [a40[:,0],b60[:,0],c80[:,0],c100[:,0]]
        magnetism_vec       = [a40[:,6],b60[:,6],c80[:,6],c100[:,6]]
        heatcapacity_vec    = [a40[:,7],b60[:,7],c80[:,7],c100[:,7]]
        susceptibility_vec  = [a40[:,8],b60[:,8],c80[:,8],c100[:,8]]
        
    for i in range(len(energy_vec)):
        energy_vec[i] = np.array(energy_vec[i]).flatten()
        magnetism_vec[i] = np.array(magnetism_vec[i]).flatten()
        heatcapacity_vec[i] = np.array(heatcapacity_vec[i]).flatten()
        susceptibility_vec[i] = np.array(susceptibility_vec[i]).flatten()
    temps = temps.flatten()
        
    plt.figure(1)
    plt.xlabel('Temperature')
    plt.ylabel('E/N')
    plt.title('Mean Energy vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temps,energy_vec[0], label='L=40')
    plt.plot(temps,energy_vec[1], label='L=60')
    plt.plot(temps,energy_vec[2], label='L=80')
    plt.plot(temps,energy_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_energy.png', bbox_inches='tight')
    
    plt.figure(2)
    plt.xlabel('Temperature')
    plt.ylabel('|M|/N')
    plt.title('Mean Magnetism vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temps,magnetism_vec[0], label='L=40')
    plt.plot(temps,magnetism_vec[1], label='L=60')
    plt.plot(temps,magnetism_vec[2], label='L=80')
    plt.plot(temps,magnetism_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_magnetic.png', bbox_inches='tight')
    
    plt.figure(3)
    plt.xlabel('Temperature')
    plt.ylabel('Cv/N')
    plt.title('Heatcapacity vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temps,heatcapacity_vec[0], label='L=40')
    plt.plot(temps,heatcapacity_vec[1], label='L=60')
    plt.plot(temps,heatcapacity_vec[2], label='L=80')
    plt.plot(temps,heatcapacity_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_heatcapacity.png', bbox_inches='tight')
    
    plt.figure(4)
    plt.xlabel('Temperature')
    plt.ylabel('X/N')
    plt.title('Susceptibility vs Temperature')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(temps,susceptibility_vec[0], label='L=40')
    plt.plot(temps,susceptibility_vec[1], label='L=60')
    plt.plot(temps,susceptibility_vec[2], label='L=80')
    plt.plot(temps,susceptibility_vec[3], label='L=100')
    plt.legend()
    plt.savefig('T_vs_L_susceptibility.png', bbox_inches='tight')
    
    plt.show()
    return()
    
def plot_task_d(t1,t2):
    plt.figure(1)
    hist_t1 = np.histogram(t1, bins=20)
    x = hist_t1[1] 
    x = x[0:-1]
    y = hist_t1[0]/np.sum(hist_t1[0])
    plt.bar(x,y, width=0.002)
    plt.ylabel('Probability [%]')
    plt.xlabel('E/N')
    plt.title('P(E), T=1, Cycles = [10^4 to 10^6]')
    
    plt.figure(2)
    hist_t2 = np.histogram(t2, bins=10)
    x = hist_t2[1] 
    x = x[0:-1]
    y = hist_t2[0]/np.sum(hist_t2[0])
    plt.bar(x,y, width=0.001)
    plt.ylabel('Probability [%]')
    plt.xlabel('E/N')
    plt.title('P(E), T=2.4, Cycles = [10^4 to 10^6]')
 
    plt.show()
    return()

def plot_task_c(energy_vec,magnetic_vec,accepted_spins_vec):
    ''' Take in data from three vectors [energy,magnetism,acceptance]
        each cathegory holds [orderedT1,disorderedT1,orderedT2.4,disorderedT2.4]
        outputs plots of the data and saves to file '''
    plt.figure(1)
    plt.xlabel('Cycles')
    plt.ylabel('E/N')
    plt.title('Mean Energy vs Cycles [T=1]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(energy_vec[0], label='Ordered')
    plt.plot(energy_vec[1], label='Disordered')
    plt.legend()
    plt.ylim(-2.01, -1.91)
    plt.savefig('T1_energies_cycles.png', bbox_inches='tight')
    
    plt.figure(2)
    plt.xlabel('Cycles')
    plt.ylabel('E/N')
    plt.title('Mean Energy vs Cycles [T=2.4]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(energy_vec[2], label='Ordered')
    plt.plot(energy_vec[3], label='Disordered')
    plt.legend()
    plt.ylim(-1.4, -1)
    plt.savefig('T24_energies_cycles.png', bbox_inches='tight')
    
    plt.figure(3)
    plt.xlabel('Cycles')
    plt.ylabel('|M|/N')
    plt.title('Mean Magnetism vs Cycles [T=1]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(magnetic_vec[0], label='Ordered')
    plt.plot(magnetic_vec[1], label='Disordered')
    plt.legend()
    plt.ylim(0.85, 1.02)
    plt.savefig('T1_magnetic_cycles.png', bbox_inches='tight')
    
    plt.figure(4)
    plt.xlabel('Cycles')
    plt.ylabel('|M|/N')
    plt.title('Mean Magnetism vs Cycles [T=2.4]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(magnetic_vec[2], label='Ordered')
    plt.plot(magnetic_vec[3], label='Disordered')
    plt.legend()
    plt.ylim(0.25, 0.65)
    plt.savefig('T24_magnetic_cycles.png', bbox_inches='tight')
    
    plt.figure(5)
    plt.xlabel('Cycles')
    plt.ylabel('Nr of Accepted spins')
    plt.title('Accepted spins vs Cycles [T=1]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(accepted_spins_vec[0], label='Ordered')
    plt.plot(accepted_spins_vec[1], label='Disordered')
    plt.legend()
    plt.savefig('T1_acceptance_cycles.png', bbox_inches='tight')
    
    plt.figure(6)
    plt.xlabel('Cycles')
    plt.ylabel('Nr of Accepted spins')
    plt.title('Accepted spins vs Cycles [T=2.4]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(accepted_spins_vec[2], label='Ordered')
    plt.plot(accepted_spins_vec[3], label='Disordered')
    plt.legend()
    plt.savefig('T24_acceptance_cycles.png', bbox_inches='tight')
    return()