import numpy as np
import time

class Solver_small:
    '''Class that solves the equations by use of algorithms'''
    def __init__(self,N,h,x0,y0,vx0,vy0,BP,beta,rel):
        self.error  = 10**(-6)
        self.fpt    = 4*np.pi*np.pi
        self.N      = N
        self.h      = h
        self.x0     = x0
        self.y0     = y0
        self.vx0    = vx0
        self.vy0    = vy0
        self.BP     = BP
        self.sun    = BP[0] # self.earth.(name,mass,radius)
        self.earth  = BP[1]
        self.rel    = rel
        self.beta   = beta
        self.num    = len(BP)
        
    def relative(self,xm,xs,ym,ys,vxm,vym):
        c = 63285                           # Speed of light [AU/y]
        r = np.sqrt((xm-xs)**2+(ym-ys)**2)  # Distance between sun and mercury
        L = (xm*vym-ym*vxm)                 # Ang.mom of mercury
        rel = (3*L**2)/(r**2*c**2)          # Relativistic term
        return(rel)
        
    def euler(self):
        '''Nummerical algorithm for solving differential equations'''
        TS  = time.clock()
        x   = np.zeros(self.N)
        y   = np.zeros(self.N)
        vx  = np.zeros(self.N)
        vy  = np.zeros(self.N)
        r   = np.zeros(self.N)
        x[0]  = self.x0
        y[0]  = self.y0
        vx[0] = self.vx0
        vy[0] = self.vy0
        r[0]  = np.sqrt(x[0]**2+y[0]**2)
        for i in range (self.N-1):
            x[i+1]      = x[i]  + self.h*vx[i]
            y[i+1]      = y[i]  + self.h*vy[i]
            vx[i+1]     = vx[i] - self.h*(self.fpt)*x[i+1]/((r[i]*r[i]**(self.beta)))
            vy[i+1]     = vy[i] - self.h*(self.fpt)*y[i+1]/((r[i]*r[i]**(self.beta)))
            r[i+1]      = np.sqrt(x[i+1]**2+y[i+1]**2)

        T = (time.clock() - TS)
        print(T,'Euler')
        return(x,y,vx,vy,T)
        
    def verlet(self):
        '''Nummerical algorithm for solving differential equations'''
        TS  = time.clock()
        x   = np.zeros(self.N)
        y   = np.zeros(self.N)
        vx  = np.zeros(self.N)
        vy  = np.zeros(self.N)
        ax  = np.zeros(self.N)
        ay  = np.zeros(self.N)
        r   = np.zeros(self.N)
        x[0]  = self.x0
        y[0]  = self.y0
        vx[0] = self.vx0
        vy[0] = self.vy0
        r[0]  = np.sqrt(x[0]**2+y[0]**2)
        if self.rel != 0:
            gR = self.relative(x[0],0,y[0],0,vx[0],vy[0])
            ax[0] = (-x[0]*(self.fpt)/(r[0]*r[0]**(self.beta)))*(1+gR)
            ay[0] = (-y[0]*(self.fpt)/(r[0]*r[0]**(self.beta)))*(1+gR)
        else:
            ax[0] = -x[0]*(self.fpt)/(r[0]*r[0]**(self.beta))
            ay[0] = -y[0]*(self.fpt)/(r[0]*r[0]**(self.beta))
        
        multiplier1 = self.h*0.5
        for i in range (self.N-1):
            x[i+1]      = x[i]  + self.h*(vx[i] + multiplier1 * ax[i])
            y[i+1]      = y[i]  + self.h*(vy[i] + multiplier1 * ay[i])
            r[i+1]      = np.sqrt(x[i+1]**2 + y[i+1]**2)
            multiplier2 = (self.fpt)/(r[i+1]*r[i+1]**(self.beta))
            if self.rel != 0:
                gR = self.relative(x[i+1],0,y[i+1],0,vx[i],vy[i])
                ax[i+1]     = (-x[i+1]*multiplier2)*(1+gR)
                ay[i+1]     = (-y[i+1]*multiplier2)*(1+gR)
            else:
                ax[i+1]     = -x[i+1]*multiplier2
                ay[i+1]     = -y[i+1]*multiplier2
            vx[i+1]     = vx[i] + multiplier1 * (ax[i+1] + ax[i])
            vy[i+1]     = vy[i] + multiplier1 * (ay[i+1] + ay[i])
        T = (time.clock() - TS)
        print(T,'Verlet')
        return(x,y,vx,vy,ax,ay,T)
        
# Jupiter added (No priority to fuse atm)-----------------------------
        
    def acc_earth(self,xe,ye,xj,yj):
        ''' Calculates acceleration for Earth '''
        d_x = xe-xj
        d_y = ye-yj
        rej = np.sqrt(d_x**2+d_y**2)
        res = np.sqrt(xe**2+ye**2)
        ax = -self.fpt*((xe/res**3) - (self.BP[2].mass/rej**3)*(d_x))
        ay = -self.fpt*((ye/res**3) - (self.BP[2].mass/rej**3)*(d_y))
        return(ax,ay)
        
    def acc_jup(self,xj,yj):
        ''' Calculates acceleration for Jupiter '''
        rj      = np.sqrt(xj**2 + yj**2)
        multiplier = (self.fpt)/rj**3
        axj     = -xj*multiplier
        ayj     = -yj*multiplier
        return(axj,ayj)
        
    def verlet_jup(self):
        '''Nummerical algorithm for solving differential equations'''
        TS  = time.clock()
        x   = np.zeros(self.N)
        y   = np.zeros(self.N)
        vx  = np.zeros(self.N)
        vy  = np.zeros(self.N)
        ax  = np.zeros(self.N)
        ay  = np.zeros(self.N)
        xj   = np.zeros(self.N)
        yj   = np.zeros(self.N)
        vxj  = np.zeros(self.N)
        vyj  = np.zeros(self.N)
        axj  = np.zeros(self.N)
        ayj  = np.zeros(self.N)
        x[0]  = self.BP[1].pos[0]
        xj[0]  = self.BP[2].pos[0]
        y[0]  = self.BP[1].pos[1]
        yj[0]  = self.BP[2].pos[1]
        vx[0] = self.BP[1].vc[0]
        vxj[0] = self.BP[2].vc[0]
        vy[0] = self.BP[1].vc[1]
        vyj[0] = self.BP[2].vc[1]
        ax[0],ay[0] = self.acc_earth(x[0],y[0],xj[0],yj[0])
        axj[0],ayj[0] = self.acc_jup(xj[0],yj[0])
        for i in range (self.N-1):
            multiplier1 = self.h*0.5
            x[i+1]      = x[i]  + self.h*(vx[i] + multiplier1 * ax[i])
            y[i+1]      = y[i]  + self.h*(vy[i] + multiplier1 * ay[i])
            xj[i+1]      = xj[i]  + self.h*(vxj[i] + multiplier1 * axj[i])
            yj[i+1]      = yj[i]  + self.h*(vyj[i] + multiplier1 * ayj[i])
            ax[i+1],ay[i+1]     = self.acc_earth(x[i+1],y[i+1],xj[i+1],yj[i+1])
            axj[i+1],ayj[i+1]     = self.acc_jup(xj[i+1],yj[i+1])
            vx[i+1]     = vx[i] + multiplier1 * (ax[i+1] + ax[i])
            vy[i+1]     = vy[i] + multiplier1 * (ay[i+1] + ay[i])
            vxj[i+1]     = vxj[i] + multiplier1 * (axj[i+1] + axj[i])
            vyj[i+1]     = vyj[i] + multiplier1 * (ayj[i+1] + ayj[i])
        T = (time.clock() - TS)      
        return(x,y,vx,vy,ax,ay,T,xj,yj,vxj,vyj,axj,ayj)