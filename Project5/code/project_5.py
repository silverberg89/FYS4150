import numpy as np
import methods_5
import matplotlib.pyplot as plt
from MCarlo import MC
np.random.seed(12)

print('1: Task for Model [A] - Random transactions')
print('2: Task for Model [B] - Saving interaction')
print('3: Task for Model [C] - Relationship interaction')
print('4: Task for Model [D] - Memory interaction')
task = int(input('Choose task:'))

# Values
N           = 500               # Nr of agents
m0          = 1                 # Initial money
trades      = int(1*10**4)    # Number of transactions (Patrica had 10**3)
cycles      = int(2)      # Number of MC cycles  (Patrica had 10**7)
lmbd        = 0.0               # Saving factor / taxation factor
gamma       = 5.0               # Importance of historical trades
alpha       = 0.0               # Relationship between agents
    
if task == 4:
    ''' Task for Model [D] - Memory interaction '''
    # Memory factors
    gamma_vec = [0.0,1.0,2.0,3.0,4.0]
    
    # Load data
    loaded = str(input('Load data?: [j/n]'))
    if loaded == 'j':
        # Values used in produced file ---------------------------------------
        cycles      = int(1*10**3)
        trades      = int(5*10**5)
        N           = 1000
        m0          = 1
        lmbd        = 0.5
        gamma       = 0
        alpha       = 2
        gamma_vec = [0.0,1.0,2.0,3.0,4.0]
        # ---------------------------------------------------------------------
        print('1: Look at data for alpha 1.0')
        print('2: Look at data for alpha 2.0')
        alpha = str(input('Choose alpha value:'))
        vals = np.load('vals_para_'+alpha+'.npy')
        bin_vecs = []
        xaxis_vec = []
        bin_size_vec = []
        for i in range(len(gamma_vec)):
            bin_vecs.append(vals[i][1])
            xaxis_vec.append(vals[i][2])
            bin_size_vec.append(vals[i][3])
            
    else:
        # Run Monte Carlo simulation
        bin_vecs     = []
        var_vecs     = []
        xaxis_vec    = []
        bin_size_vec = []
        for i,gamma in enumerate(gamma_vec):
            m_carlo = MC(cycles,trades,alpha,gamma,lmbd,N,m0)
            a,b,v,xa,bs   = m_carlo.mc()
            bin_vecs.append(b)
            var_vecs.append(v)
            xaxis_vec.append(xa)
            bin_size_vec.append(bs)
        
        # Save data
        np.save('bin_vecs_alpha.npy',bin_vecs)
        np.save('var_vecs_alpha.npy',var_vecs)
    
    # Plot loglog frame of data
    methods_5.plot_loglog(bin_vecs,cycles,N,m0,bin_size_vec,xaxis_vec,gamma_vec,'Gamma = ')
    
    # Plot power fit in semilog frame
    a_vec = []
    for h,gamma in enumerate(gamma_vec):
        bin_bar = bin_vecs[h]/cycles/N/bin_size_vec[h]
        low_lim = max(xaxis_vec[h])*0.9
        a,cut = methods_5.curve_fitting(xaxis_vec[h],bin_bar,low_lim)
        a_vec.append(a)
        pow_fit = methods_5.func(xaxis_vec[h][cut],a)
        methods_5.plot_power(a_vec[h],xaxis_vec[h][cut],bin_bar[cut],pow_fit,gamma,low_lim,x_lim=False)

if task == 3:
    ''' Task for Model [C] - Relationship interaction '''
    # Relationship factors
    alpha_vec = [0.5,1.0,1.5,2.0]
    
    # Load data
    loaded = str(input('Load data?: [j/n]'))
    if loaded == 'j':
        # Values used in produced file ---------------------------------------
        cycles      = int(1*10**3)
        trades      = int(1*10**5)
        N           = 1000
        m0          = 1
        lmbd        = 0
        gamma       = 0
        alpha       = 0
        alpha_vec   = [0.5,1.0,1.5,2.0]
        # ---------------------------------------------------------------------
        data = np.load('vals_paras_0.npy') # [agent_wallet,bin_vec,xaxis,bin_size]
        bin_vecs     = []
        var_vecs     = []
        xaxis_vec    = []
        bin_size_vec = []
        for i in range(len(data)):
            bin_vecs.append(data[i][1])
            xaxis_vec.append(data[i][2])
            bin_size_vec.append(data[i][3])
    else:
        # Run Monte Carlo simulation
        bin_vecs     = []
        var_vecs     = []
        xaxis_vec    = []
        bin_size_vec = []
        for i,alpha in enumerate(alpha_vec):
            m_carlo = MC(cycles,trades,alpha,gamma,lmbd,N,m0)
            a,b,v,xa,bs   = m_carlo.mc()
            bin_vecs.append(b)
            var_vecs.append(v)
            xaxis_vec.append(xa)
            bin_size_vec.append(bs)
        
        # Save data
        np.save('bin_vecs_alpha'+str(alpha)+'.npy',bin_vecs)
        np.save('var_vecs_alpha'+str(alpha)+'.npy',var_vecs)
    
    # Plot loglog frame of data
    methods_5.plot_loglog(bin_vecs,cycles,N,m0,bin_size_vec,xaxis_vec,alpha_vec,'Alpha = ')
        
    # Plot power fits
    a_vec = []
    for h,alpha in enumerate(alpha_vec):
        xaxis = xaxis_vec[h]
        bin_bar = bin_vecs[h]/cycles/N/bin_size_vec[h]
        low_lim = max(xaxis_vec[h])*0.8
        a,cut = methods_5.curve_fitting(xaxis,bin_bar,low_lim)
        a_vec.append(a)
        pow_fit = methods_5.func(xaxis[cut],a)
        methods_5.plot_power(a_vec[h],xaxis[cut],bin_bar[cut],pow_fit,alpha,low_lim,x_lim=False)

if task == 2:
    ''' Task for Model [B] - Saving interaction '''
    # Saving factors
    lmbd_vec = [0,0.25,0.5,0.9]
    
    # Load data
    loaded = str(input('Load data?: [j/n]'))
    if loaded == 'j':
        # Values used in produced file ---------------------------------------
        cycles      = int(1*10**3)
        trades      = int(5*10**5)
        N           = 500
        m0          = 1
        lmbd        = 0
        gamma       = 0
        alpha       = 0
        lmbd_vec    = [0,0.25,0.5,0.9]
        # ---------------------------------------------------------------------
        data = np.load('vals_paras_0.npy') # [agent_wallet,bin_vec,xaxis,bin_size]
        bin_vecs     = []
        var_vecs     = []
        xaxis_vec    = []
        bin_size_vec = []
        for i in range(len(data)):
            bin_vecs.append(data[i][1])
            xaxis_vec.append(data[i][2])
            bin_size_vec.append(data[i][3])    
    else:
        # Run Monte Carlo simulation
        bin_vecs     = []
        var_vecs     = []
        xaxis_vec    = []
        bin_size_vec = []
        for i,lmbd in enumerate(lmbd_vec):
            m_carlo = MC(cycles,trades,alpha,gamma,lmbd,N,m0)
            a,b,v,xa,bs   = m_carlo.mc()
            bin_vecs.append(b)
            var_vecs.append(v)
            xaxis_vec.append(xa)
            bin_size_vec.append(bs)
        
    # Save data
    np.save('bin_vecs_lambda'+str(alpha)+'.npy',bin_vecs)
    np.save('var_vecs_lambda'+str(alpha)+'.npy',var_vecs)
    
    # Plot rel. frequency for each saving factor 
    bin_vec = []
    fig = plt.figure()
    for j in range(i+1):
        bin_bar = bin_vecs[j]/cycles/N
        bin_vec.append(bin_bar)
        pow_fit = methods_5.power_fit(lmbd_vec[j],xaxis_vec[j]) # Power fit by Patriarca
        plt.scatter(xaxis_vec[j],pow_fit*bin_size_vec[j],s=8,color='k',marker='s')
        plt.plot(xaxis_vec[j],bin_bar,label='Lambda = '+str(lmbd_vec[j]))
        plt.legend()
        plt.xlim(0,4)
        plt.xlabel('Money [m]')
        plt.ylabel('f(m)')
        plt.savefig('saving.png',dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot power fits
    a_vec = []
    for h,lmbd in enumerate(lmbd_vec):
        xaxis = xaxis_vec[h]
        bin_bar = bin_vecs[h]/cycles/N/bin_size_vec[h]
        low_lim = max(xaxis_vec[h])*0.90
        a,cut = methods_5.curve_fitting(xaxis,bin_bar,low_lim)
        a_vec.append(a)
        pow_fit = methods_5.func(xaxis[cut],a)
        methods_5.plot_power(a_vec[h],xaxis[cut],bin_bar[cut],pow_fit,lmbd,low_lim,x_lim=False)

if task == 1:
    ''' Task for Model [A] - Random transactions '''
    # Run Monte Carlo simulation
    print('Estimated comp.time [hours]:', np.round(methods_5.func_time(trades),decimals=3))
    m_carlo = MC(cycles,trades,alpha,gamma,lmbd,N,m0)
    agent_wallet,bin_vec,variance,xaxis,bin_size = m_carlo.mc()
    
    # Save data
    np.save('bin_vecs_zero'+str(alpha)+'.npy',bin_vec)
    np.save('var_vecs_zero'+str(alpha)+'.npy',variance)
    np.save('agent_vecs_zero'+str(alpha)+'.npy',agent_wallet)
    
    # Calculate distribution functions
    m_avg   = np.sum(agent_wallet)/N
    beta    = 1/m_avg
    w_gibbs = (beta * np.exp(-beta * agent_wallet))  # Gibbs distribution
    bin_bar = bin_vec/cycles/N/bin_size              # MC simulated distribution
    
    # Plot bin differences as funcing of cycles.
    diff = m_carlo.wallet
    methods_5.plot_cycles(diff,cycles)
    
    # Plot rel. frequency and variance
    methods_5.plot_distribution(xaxis,bin_bar,w_gibbs,bin_size,agent_wallet) 
    methods_5.plot_var(trades,variance,m0,beta)