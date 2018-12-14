import numpy as np
import time
np.random.seed(12)

class MC:
    '''Class that runs the Monte Carlo alogrithm'''
    def __init__(self,cycles,trades,N,m0):# Convention to use self 
        ''' Self draws information from input variables to class variables.'''
        self.cycles     = cycles                   # Keep similar name if possible
        self.trades     = trades 
        self.N          = N
        self.m0         = m0
        self.variance   = []
        self.bin_vec    = []
        self.xaxis      = []
        self.nr_bins    = 0
        self.bin_last   = 0
        self.bin_size   = 0
        
    def hist_vals(self,agent_wallet):
        ''' Take in agent wealth and create bins for each new cycle '''
        self.bin_size = 0.05 * self.m0
        self.bin_last = max(agent_wallet)
        self.nr_bins  = int(self.bin_last/self.bin_size)
        self.xaxis   = np.linspace(0,self.bin_last,self.nr_bins)
        self.bin_vec = np.zeros(self.nr_bins)
        return()

    def bin_sort(self,agent_wallet,u):
        ''' Take in agents wealth and sort into histogram bins '''
        for y in range(self.N): 
            for w in range(len(self.bin_vec)):
                if agent_wallet[y] > (w)*self.bin_size and agent_wallet[y] <= (w+1)*self.bin_size:
                  self.bin_vec[w] += 1
        return()
                      
    def mc(self,alpha,lmbd,gamma):
        ''' Core of monte carlo algorithm '''
        for u in range(self.cycles):
            print(u)
            t0 = time.clock()
            agent_wallet = np.ones(self.N)*self.m0     # Initilize agents capital
            c_matrix     = np.zeros((self.N,self.N))   # Set up transaction memory as matrix
            for k in range(self.trades):
                i = np.random.randint(self.N)          # Random pair of agents selected.
                j = np.random.randint(self.N)
                if (i != j):
                    epsilon = np.random.uniform(0.0,1) # Random nr drawn
                    dw      = np.abs((agent_wallet[i] - agent_wallet[j])) # Pre-calc for efficency
                    if (dw == 0):                      # Evaluate probability conditions
                        prob = 1
                    else:
                        prob = dw**(-(alpha))*((c_matrix[i,j]+c_matrix[j,i])+1)**(gamma) #Add both c(i,j)&c(j,i) since they are the same.
                    if (np.random.uniform(0.0,1) <= prob ): # Transaction accepted, make deal
                        delta_money      = (1-lmbd)*(epsilon*agent_wallet[j]-(1-epsilon)*agent_wallet[i])
                        agent_wallet[i] += delta_money       # Add deal outcome for each agent
                        agent_wallet[j] -= delta_money
                        c_matrix[i,j]   += 1                 # Add transaction to memory

            agent_wallet = np.sort(-agent_wallet)*(-1)  # Sort wealth of agents from largest to smallest.
            if u == 0:
                self.hist_vals(agent_wallet)
            print(time.clock()-t0)
            self.bin_sort(agent_wallet,u)                 # Sort wealth distribution into bins
        return(agent_wallet,np.copy(self.bin_vec),self.xaxis,self.bin_size)