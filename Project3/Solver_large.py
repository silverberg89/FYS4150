import numpy as np
import time

class Solver_large:
    '''Class that solves the equations by use of algorithms
    and test against angular momentum, potential energy and kinetic energy'''
    def __init__(self,N,h,BP):
        self.error  = 10**(-3)
        self.fpt    = 4*np.pi*np.pi
        self.N      = N
        self.h      = h
        self.BP     = BP
        self.num    = len(BP)
        self.test   = 0
        
    def acc_new(self,xe,ye,xj,yj,i):
        ''' Calculates acceleration for bodies ''' 
        d_x = xe-xj
        d_y = ye-yj
        rej = np.sqrt(d_x**2+d_y**2)
        res = np.sqrt(xe**2+ye**2)
        ax = -self.fpt*((xe/res**3) - (self.BP[i+1].mass/rej**3)*(d_x))
        ay = -self.fpt*((ye/res**3) - (self.BP[i+1].mass/rej**3)*(d_y))
        if i == 0:
            ax = -self.fpt*(0- (self.BP[i+1].mass/rej**3)*(d_x))
            ay = -self.fpt*(0- (self.BP[i+1].mass/rej**3)*(d_y))
        return(ax,ay)
        
    def acc_last(self,xj,yj):
        ''' Calculates acceleration for last body '''
        rj      = np.sqrt(xj**2 + yj**2) 
        multiplier = (self.fpt)/rj**3
        axj     = -xj*multiplier
        ayj     = -yj*multiplier
        return(axj,ayj)
        
    def verlet(self):
        '''Nummerical algorithm for solving differential equations'''
        TS  = time.clock()
        x = np.zeros((self.num,self.N))
        y = np.zeros((self.num,self.N))
        vx = np.zeros((self.num,self.N))
        vy = np.zeros((self.num,self.N))
        ax = np.zeros((self.num,self.N))
        ay = np.zeros((self.num,self.N))
        
        # Initial values for all planets [i]
        for i in range(0,self.num):
            x[i,0]    = self.BP[i].pos[0]
            y[i,0]    = self.BP[i].pos[1]
            vx[i,0]   = self.BP[i].vc[0]
            vy[i,0]   = self.BP[i].vc[1]
        for i in range(0,self.num-1):
            ax[i,0],ay[i,0] = self.acc_new(x[i,0],y[i,0],x[i+1,0],y[i+1,0],i)
            if i == self.num-1:
                ax[i,0],ay[i,0] = self.acc_last(x[i,0],y[i,0])
                
        # Update values for all planets [i]
        for i in range(0,self.num-1):
            for j in range(self.N-1):
                multiplier1 = self.h*0.5
                x[i,j+1]      = x[i,j]    + self.h*(vx[i,j]   + multiplier1 * ax[i,j])
                y[i,j+1]      = y[i,j]    + self.h*(vy[i,j]   + multiplier1 * ay[i,j])
                x[i+1,j+1]    = x[i+1,j]  + self.h*(vx[i+1,j] + multiplier1 * ax[i+1,j])
                y[i+1,j+1]    = y[i+1,j]  + self.h*(vy[i+1,j] + multiplier1 * ay[i+1,j])

                ax[i,j+1],ay[i,j+1]     = self.acc_new(x[i,j+1],y[i,j+1],x[i+1,j+1],y[i+1,j+1],i)
                if i == self.num-1:
                    ax[i,j+1],ay[i,j+1] = self.acc_last(x[i,j+1],y[i,j+1])
                
                vx[i,j+1]     = vx[i,j] + multiplier1 * (ax[i,j+1] + ax[i,j])
                vy[i,j+1]     = vy[i,j] + multiplier1 * (ay[i,j+1] + ay[i,j])
                
        T = (time.clock() - TS)
        print('Time: '+str(T))
        self.kinetic_energy(vx,vy)
        self.potential_energy(x,y)
        self.angular_momentum(x,y,vx,vy)
        if self.test == 0:
            print('Ang.momentum, Pot.energy and Kin.energy conserved')
        return(x,y,vx,vy,ax,ay)
    
    def angular_momentum(self,x,y,vx,vy):
        ''' Test such that angular momentum is conserved'''
        for i in range(self.num-1):
            self.BP[i].mass = 1
            px = [self.BP[i].mass*vx[i,0], self.BP[i].mass*vx[i,-1]]
            py = [self.BP[i].mass*vy[i,0], self.BP[i].mass*vy[i,-1]]
            cross = [x[i,0]*py[0] - y[i,0]*px[0], x[i,-1]*py[-1] - y[i,-1]*px[-1]]
            if abs(cross[0]-cross[-1]) > self.error:
                print('Angular Momentum Not Conserved for: '+self.BP[i].name)
                self.test = 1
                print(abs(cross[0]-cross[-1]))
        return(self)
        
    def kinetic_energy(self,vx,vy):
        ''' Test such that KE is conserved'''
        KE = np.zeros((self.num,2))
        for i in range(self.num-1):
            halfmass  = 0.5*self.BP[i].mass
            KE[i,0] = (halfmass*(vx[i,0]**2+vy[i,0]**2))
            KE[i,1] = (halfmass*(vx[i,-1]**2+vy[i,-1]**2))
            if abs(KE[i,0]-KE[i,-1]) > self.error:
                print('Kinetic Energy Not Conserved for: '+self.BP[i].name)
                self.test = 1
                print(abs(KE[i,0]-KE[i,-1]))
        return(self)
        
    def potential_energy(self,x,y):
        ''' Test such that PE is conserved'''
        PE = np.zeros((self.num,2))
        for i in range(self.num-1):
            numerator  = self.fpt*self.BP[i].mass*self.BP[i+1].mass
            PE[i,0] = (numerator/np.sqrt(x[i,0]**2+y[i,0]**2))
            PE[i,1] = (numerator/np.sqrt(x[i,-1]**2+y[i,-1]**2))
            if abs(PE[i,0]-PE[i,-1]) > self.error:
                print('Potential Energy Not Conserved for: '+self.BP[i].name)
                self.test = 1
                print(abs(PE[i,0]-PE[i,-1]))
        return(self)