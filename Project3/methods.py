import numpy as np
import matplotlib.pyplot as plt
from Solver_small import Solver_small # Imports a class saved as .py

def PLOT_F(x,y,xlab,ylab,ptit,plab,st):
    ''' PLOT_F(X,Y,'xlabel','ylabel','title','legend') '''
    fig = plt.figure()
    plt.plot(x,y, label=plab)
    plt.scatter([0,0],[0,0], label='Sun', s=50, color='yellow')
    plt.scatter(x[-1],y[-1], s=50, color='black')
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.legend()
    plt.title(ptit)
    plt.axis('equal')
    if st == 1:
        plt.axis([0.9999,1.0001,-0.0001,0.0001])
        plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    plt.show()
    fig.savefig(ptit+str('.png'), bbox_inches = "tight")
    return()
    
def stability(x0,y0,vx0,vy0,tend,t0,body_package,error,fpt):
    ''' Test different time steps and shows the implication on the caluclations'''
    for n in range(4,6):
        N = 10**(n)
        h = (tend-t0)/N
        
        initial                            = Solver_small(N,h,x0,y0,vx0,vy0,body_package,2,0)
        x_E,y_E,vx_E,vy_E,Time_E           = Solver_small.euler(initial)
        x_V,y_V,vx_V,vy_V,ax_V,ay_V,Time_V = Solver_small.verlet(initial)

        ange = angular_momentum(body_package[1].mass,error,x_E,y_E,vx_E,vy_E,'Euler')
        PEe = potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x_E,y_E,'Euler')
        KEe = kinetic_energy(body_package[1].mass,error,vx_E,vy_E,'Euler')
        angv = angular_momentum(body_package[1].mass,error,x_V,y_V,vx_V,vy_V,'Verlet')
        PEv = potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x_V,y_V,'Verlet')
        KEv = kinetic_energy(body_package[1].mass,error,vx_V,vy_V,'Verlet')

        PLOT_F(x_E,y_E,'x [AU]','y [AU]','Euler'+', dt=%1.4f' %h+', N=%i' %N+', Years=%i' %tend,'Earth',1)
        PLOT_F(x_V,y_V,'x [AU]','y [AU]','Verlet'+', dt=%1.4f' %h+', N=%i' %N+', Years=%i' %tend,'Earth',1)
    return()
    
def escape(x0,y0,vx0,vy0,N,h,tend,body_package,error,fpt):
    '''Shows how the initial velocity can become an escape velocity'''
    initial                            = Solver_small(N,h,x0,y0,vx0,vy0,body_package,2,0)
    x_E,y_E,vx_E,vy_E,Time_E           = Solver_small.euler(initial)
    x_V,y_V,vx_V,vy_V,ax_V,ay_V,Time_V = Solver_small.verlet(initial)
    
    ange = angular_momentum(body_package[1].mass,error,x_E,y_E,vx_E,vy_E,'Euler')
    PEe = potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x_E,y_E,'Euler')
    KEe = kinetic_energy(body_package[1].mass,error,vx_E,vy_E,'Euler')
    angv = angular_momentum(body_package[1].mass,error,x_V,y_V,vx_V,vy_V,'Verlet')
    PEv = potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x_V,y_V,'Verlet')
    KEv = kinetic_energy(body_package[1].mass,error,vx_V,vy_V,'Verlet')
    
    PLOT_F(x_E,y_E,'x [AU]','y [AU]','Euler escape'+', dt=%1.4f' %h+', N=%i' %N+', Years=%i' %tend,'Earth',0)
    PLOT_F(x_V,y_V,'x [AU]','y [AU]','Verlet escape'+', dt=%1.4f' %h+', N=%i' %N+', Years=%i' %tend,'Earth',0)
    return()
    
def beta1(N,h,x0,y0,vx0,vy0,body_package,B,error,fpt):
    '''Test different values for the exponent of the radius in Newtons Law.'''
    for i in range(len(B)):
        initial                            = Solver_small(N,h,x0,y0,vx0,vy0,body_package,B[i],0)
        x_E,y_E,vx_E,vy_E,Time_E           = Solver_small.euler(initial)
        x_V,y_V,vx_V,vy_V,ax_V,ay_V,Time_V = Solver_small.verlet(initial)
        
        ange = angular_momentum(body_package[1].mass,error,x_E,y_E,vx_E,vy_E,'Euler')
        PEe = potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x_E,y_E,'Euler')
        KEe = kinetic_energy(body_package[1].mass,error,vx_E,vy_E,'Euler')
        angv = angular_momentum(body_package[1].mass,error,x_V,y_V,vx_V,vy_V,'Verlet')
        PEv = potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x_V,y_V,'Verlet')
        KEv = kinetic_energy(body_package[1].mass,error,vx_V,vy_V,'Verlet')
        
        PLOT_F(x_E,y_E,'x [AU]','y [AU]',('Sun & Earth System [Euler] beta: '+str(B[i])),'Earth',0)
        PLOT_F(x_V,y_V,'x [AU]','y [AU]',('Sun & Earth System [Verlet] beta: '+str(B[i])),'Earth',0)
    return()
    
def center_system(body_package):
    ''' Sets the system to a origin calculated by center of mass'''
    # Center of mass (System)
    c_of_m = [0,0]
    M = len(body_package)-1
    mass_v = np.zeros(M)
    pos_vx = np.zeros(M)
    pos_vy = np.zeros(M)
    vc_vx = np.zeros(M)
    vc_vy = np.zeros(M)
    for i in range(M):
        mass_v[i] = body_package[i].mass
        pos_vx[i] = body_package[i].pos[0]
        pos_vy[i] = body_package[i].pos[1]
        vc_vx[i] = body_package[i].vc[0]
        vc_vy[i] = body_package[i].vc[1]
    c_of_m[0] = np.dot(mass_v,pos_vx)/(sum(mass_v))
    c_of_m[1] = np.dot(mass_v,pos_vy)/(sum(mass_v))
    for j in range(M):
        body_package[j].pos[0] += c_of_m[0]
        body_package[j].pos[1] += c_of_m[1]
    
    # Velocity of sun from total.sys.momentum = 0
    Lx = np.zeros(M)
    Ly = np.zeros(M)
    for i in range(1,M):
        Lx[i] = mass_v[i]*(vc_vx[i])
        Ly[i] = mass_v[i]*(vc_vy[i])
    body_package[0].vc[0] = sum(Lx)/mass_v[0]
    body_package[0].vc[1] = sum(Ly)/mass_v[0]
    return(body_package)
    
def savedata(body_package,x,y,vx,vy,ax,ay,tend,method):
    if method == '_Three_body_cofs':
        body_package[0] = body_package[1]
        body_package[1] = body_package[2]
    for i in range(len(x)):
        data = np.c_[x[i,:], y[i,:], vx[i,:], vy[i,:], ax[i,:], ay[i,:]]
        np.savetxt(body_package[i].name+method+'.txt', data,fmt='%1.9f',header='x,y,vx,vy,ax,ay'+'  (From 0y to '+str(tend)+'y)')
    return()
    
def angular_momentum(mass,error,x,y,vx,vy,method):
    ''' Test such that angular momentum is conserved'''
    mass = 1
    py = [mass*vy[0], mass*vy[-1]]
    px = [mass*vx[0], mass*vx[-1]]
    ang = np.zeros(2)
    ang[0] = (x[0]*py[0] - y[0]*px[0])
    ang[1] = (x[-1]*py[-1] - y[-1]*px[-1])
    e = abs(ang[0]-ang[-1])
    if e < error:
        print('Angular Momentum Conserved for: '+method)
    else:
        print('Angular Momentum Not Conserved for: '+method)
    return (e)
    
def potential_energy(fpt,mass1,mass2,error,x,y,method):
    ''' Test such that PE is conserved'''
    numerator = -fpt*mass1*mass2
    PE = np.zeros(2)
    PE[0] = (numerator/np.sqrt(x[0]**2+y[0]**2))
    PE[1] = (numerator/np.sqrt(x[-1]**2+y[-1]**2))
    e = abs(PE[0]-PE[-1])
    if e < error:
        print('Potential Energy Conserved for: '+method)
    else:
        print('Potential Energy Not Conserved for: '+method)
    return(e)
    
def kinetic_energy(mass,error,vx,vy,method):
    ''' Test such that KE is conserved'''
    halfmass  = 0.5*mass
    KE = np.zeros(2)
    KE[0] = (halfmass*(vx[0]**2+vy[0]**2))
    KE[1] = (halfmass*(vx[-1]**2+vy[-1]**2))
    e = abs(KE[0]-KE[-1])
    if e < error:
        print('Kinetic Energy Conserved for: '+method)
    else:
        print('Kinetic Energy Not Conserved for: '+method)
    return(e)