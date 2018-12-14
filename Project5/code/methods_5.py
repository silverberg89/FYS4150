import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
import scipy.optimize as so

def power_fit(lmbd,xaxis):
    ''' Receives saving factor and money axis
        Outputs the distribution by fitted function from Patriarca'''
    n       = 1 + 3.0 * lmbd / (1 - lmbd)
    a       = n**(n) / sc.gamma(n)
    pow_fit = a * xaxis**(n-1) * np.exp(-n*xaxis) # Patrica distribution func
    return(pow_fit)
    
def func(x,v):
    ''' Just a call function for scipys fitting feature. '''
    f = x**-(1+v)
    return (f)

def func_time(x):
    ''' Receives nr of transactions and outputs estimated computational time
        in hours.'''
    time = (0.000560301*x**0.971508)/60
    return (time)
    
def curve_fitting(xaxis,bin_bar,low_lim):
    ''' Receives the distribution and fit the end tail
        Outsputs a exponent (v) and the tail part of the money axis'''
    cut = np.where(xaxis>=low_lim)                # Only looking at the tail
    parmas = so.curve_fit(func, xaxis[cut], bin_bar[cut])[0]
    v = np.round(parmas[0],decimals=2)
    #b = parmas[1]
    return(v,cut)
    
def plot_loglog(bin_vecs,cycles,N,m0,bin_size_vec,xaxis_vec,alpha_vec,lab):
    ''' Receives distribution of MS simulation and plot it in a loglog frame
        as a function of money.'''
    bin_vec = []
    fig1 = plt.figure()
    for j in range(len(bin_vecs)):
        bin_bar = bin_vecs[j]/cycles/N/bin_size_vec[j]  # Simulated distribution
        bin_vec.append(bin_bar)
        plt.loglog(xaxis_vec[j],bin_bar,label=lab+str(alpha_vec[j]))
        plt.legend()
        plt.xlabel('log Money [m]')
        plt.ylabel('Log(f(m)')
        fig1.savefig('loglog.png',dpi=300, bbox_inches='tight')
    plt.show()
    return()
    
def plot_power(v,xaxis,bin_bar,pow_fit,para,low_lim,x_lim):
    ''' Receives MC distribution of end tail and a power function fitting the tail
        Outsputs a semilog plot illustrating the fit '''
    fig = plt.figure()
    plt.semilogy(xaxis,bin_bar,label='MC Simulation, value: '+str(para),color='c')
    plt.semilogy(xaxis,pow_fit,label='Power fit, '+'v = '+str(v)+' ,value: '+str(para),color='r')
    if x_lim == True:
        plt.xlim(low_lim,max(xaxis))
    plt.xlabel('Money [m]')
    plt.ylabel('Log(f(m))')
    plt.legend()
    fig.savefig('tail'+str(para)+'.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    return()

def plot_var(trades,variance,m0,beta):
    ''' Receives data for showing the variance as a function of transactions. '''
    
    transactions = np.arange(0,trades)
    analytic_var = 1/beta**2
    
    fig = plt.figure()
    plt.plot([0, max(transactions)], [analytic_var, analytic_var],label='Analytical variance')
    plt.plot(transactions,variance,label='MC variance')
    plt.title('Variance as function of transactions')
    plt.xlabel('Transactions')
    plt.ylabel('Variance')
    plt.legend()
    fig.savefig('Variance_transactions.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    return()
    
def plot_cycles(diff,cycles):
    ''' Receives bin vectors throuh a number of MC cycles
        Outputs a plot of the maximum bin difference decay as function of cycles'''
    delta = []
    for r in range(len(diff)):
        delta.append((abs(diff[r]-diff[r-1])/abs(diff[r])))
    delta = np.max(abs(np.array(delta)),axis=1)
    fig = plt.figure()
    plt.plot(np.arange(0,cycles),delta,label='MC bin difference')
    plt.title('Bin differences as function of cycles')
    plt.xlabel('Cycles')
    plt.ylabel('Max bin difference')
    plt.legend()
    fig.savefig('bin_diff_cycles.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    return()
    
def plot_distribution(xaxis,bin_bar,w_gibbs,bin_size,agent_wallet):
    '''Receives needed data for the distribution from gibbs and the MC simulation
       Outputs two figures, one standard frame and one semilog frame '''
    fig0 = plt.figure()
    plt.bar(xaxis,bin_bar*bin_size,width=bin_size,label='MC distribution',color='c')
    plt.plot(agent_wallet,w_gibbs*bin_size,label='Gibbs distribution',color='r')
    plt.xlabel('Money [m]')
    plt.ylabel('f(m)')
    plt.legend()
    plt.xlim(0,max(xaxis))
    fig0.savefig('Frequency_Wm.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    fig1 = plt.figure()
    plt.semilogy(xaxis,bin_bar,label='MC distribution',color='c',linewidth=6)
    plt.semilogy(agent_wallet, w_gibbs,label='Gibbs distribution',color='r',linewidth=2)
    plt.xlabel('Money [m]')
    plt.ylabel('Log(Wm)')
    plt.legend()
    plt.xlim(0,max(xaxis))
    fig1.savefig('Log_Wm.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    return()