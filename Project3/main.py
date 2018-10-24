import numpy as np
import matplotlib.pyplot as plt

from Solver_small import Solver_small # Imports a class saved as .py
from Solver_large import Solver_large
from Bodies import Bodies
from methods import PLOT_F,stability,escape,beta1,center_system,savedata,angular_momentum, potential_energy,kinetic_energy

# General data
AU     = 1.5*10**11 # Astronomical unit (One distance Sun-Earth) [Approx 1.496*10**11 [m]]
N      = 10**5                           # Gridpoints
t0     = 0                               # Time zero [Years]
tend   = 20                              # End time  [Years]
h      = (tend-t0)/N                     # Steplength
s_mass = 2*10**30                        # True solar mass
year   = 365
beta   = 2
error  = 10**(-6)
fpt    = 4*np.pi*np.pi
# Create Bodies real data
sun     = Bodies('Sun'      ,s_mass/s_mass,0*AU,[0,0],[0,0]) # Name, Solarmasses, Radius [AU], In.Position, In.Velocity
earth   = Bodies('Earth'    ,(6.0*10**24)/s_mass,1.0*AU,[9.151749056223659E-01,4.017451030775965E-01],[-7.102662898678862E-03*year,1.573492229087253E-02*year])
jup     = Bodies('Jupiter'  ,(1.9*10**27)/s_mass,5.2*AU,[-2.641080983846100E+00,-4.669163343930127E+00],[6.477738810434417E-03*year,-3.355309832993142E-03*year])
sat     = Bodies('Saturn'   ,(5.5*10**26)/s_mass,9.54*AU,[1.564770606246282E+00,-9.932658498359926E+00],[5.203015304970756E-03*year,8.501558670441044E-04*year])
mer     = Bodies('Mercur'   ,(3.3*10**23)/s_mass,0.39*AU,[-8.903090021719566E-02,-4.499236229857439E-01],[2.196444903159524E-02*year,-3.942491639469906E-03*year])
ven     = Bodies('Venus'    ,(4.9*10**24)/s_mass,0.72*AU,[6.905276989940136E-01,2.255177843947447E-01],[-6.172153245704117E-03*year,1.919886933404715E-02*year])
mar     = Bodies('Mars'     ,(6.6*10**23)/s_mass,1.52*AU,[1.383107718074162E+00,-1.022737099513783E-01],[1.631535483888925E-03*year,1.514825084837160E-02*year])
ura     = Bodies('Uranius'  ,(8.8*10**25)/s_mass,19.19*AU,[1.716988990887938E+01,1.000730997738031E+01],[-2.009307928131884E-03*year,3.214700937182314E-03*year])
nep     = Bodies('Neptun'   ,(1.03*10**26)/s_mass,30.06*AU,[2.892266581320278E+01,-7.713384630103900E+00],[7.879662190531933E-04*year,3.051370215497573E-03*year])
plu     = Bodies('Pluto'    ,(1.31*10**22)/s_mass,39.53*AU,[1.165292574078221E+01,-3.157386607794047E+01],[3.007448701026305E-03*year,4.261973528298765E-04*year])
none    = Bodies('None'     ,(0)/s_mass,0*AU,[0,0],[0*year,0*year])

print('1:Earth moves without Jupiter')
print('2:Earth moves with Jupiter')
print('3:Three-Body-System')
print('4:Solar-System')
print('5:Mercury-System')
task = input("Enter the task to execute, ex: 1:_")

if task == str(1):
    '''###_Sun as origin / Earth rotate_###__3b'''
    # Initial data
    x0     = 1                               # Initial x-position (1[AU] from sun along x-axis)
    y0     = 0                               # Initial y-position
    vx0    = 0                               # Initial velocity in x-direction
    vy0    = 2*np.pi                         # Initial velocity in y-direction
    beta   = 2
    
    # Create Bodies
    sun     = Bodies('Sun'      ,s_mass/s_mass,0*AU,[0,0],[0,0])       # Name, Solarmasses, Radius [AU], In.Position, In.Velocity
    earth   = Bodies('Earth'    ,(6.0*10**24)/s_mass,1.0*AU,[1,0],[0,2*np.pi])
    body_package = [sun,earth]
    
    # Solve ODE:s [Eart & Sun]
    initial                            = Solver_small(N,h,x0,y0,vx0,vy0,body_package,beta,0)
    x_E,y_E,vx_E,vy_E,Time_E           = Solver_small.euler(initial)
    x_V,y_V,vx_V,vy_V,ax_V,ay_V,Time_V = Solver_small.verlet(initial)
    
    ange = angular_momentum(body_package[1].mass,error,x_E,y_E,vx_E,vy_E,'Euler')
    PEe  = potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x_E,y_E,'Euler')
    KEe  = kinetic_energy(body_package[1].mass,error,vx_E,vy_E,'Euler')
    angv = angular_momentum(body_package[1].mass,error,x_V,y_V,vx_V,vy_V,'Verlet')
    PEv  = potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x_V,y_V,'Verlet')
    KEv  = kinetic_energy(body_package[1].mass,error,vx_V,vy_V,'Verlet')
    
    PLOT_F(x_E,y_E,'x [AU]','y [AU]','Sun & Earth System [Euler]','Earth',0)
    PLOT_F(x_V,y_V,'x [AU]','y [AU]','Sun & Earth System [Verlet]','Earth',0)
    
    # Save data
    euler_data_write    = np.c_[x_E, y_E, vx_E, vy_E]
    verlet_data_write   = np.c_[x_V, y_V, vx_V, vy_V, ax_V, ay_V]
    np.savetxt("Euler_data_earth.txt", euler_data_write,fmt='%1.9f',header='x,y,vx,vy,ax,ay'+'  (From 0y to '+str(tend)+'y)')
    np.savetxt("Verlet_data_earth.txt", verlet_data_write,fmt='%1.9f',header='x,y,vx,vy,ax,ay'+'  (From 0y to '+str(tend)+'y)')
    
    # Test stability of methods with different steplength.
    what = input('Test stability? [J/N]:_')
    if what == 'J':
        stability(x0,y0,vx0,vy0,tend,t0,body_package,error,fpt)
    
    # Test escape situation
    what = input('Test escape velocity? [J/N]:_')
    if what == 'J':
        print('Calculated escape velocity used: np.sqrt(8*np.pi**2/1)')
        vy0esc = np.sqrt(8*np.pi**2/1)
        escape(x0,y0,vx0,vy0esc,N,h,tend,body_package,error,fpt)
    # Test beta values
    what = input('Test beta values? [J/N]:_')
    if what == 'J':
        B = [2.95,3]
        beta1(N,h,x0,y0,vx0,vy0,body_package,B,error,fpt)
    
if task == str(2):
    '''###_Sun as origin / Earth & Jupiter rotate_###__3e'''
    body_package = [sun,earth,jup]
    
    # Solve ODE:s [Jupiter and Earth moving, Sun as center]
    initial_J                                 = Solver_small(N,h,0,0,0,0,body_package,beta,0)
    x,y,vx,vy,ax,ay,T,xj,yj,vxj,vyj,axj,ayj   = Solver_small.verlet_jup(initial_J)
    
    angular_momentum(body_package[1].mass,error,x,y,vx,vy,'Earth')
    potential_energy(fpt,body_package[0].mass,body_package[1].mass,error,x,y,'Earth')
    kinetic_energy(body_package[1].mass,error,vx,vy,'Earth')

    fig = plt.figure()
    plt.plot(x,y, linewidth=0.5, color='blue')
    plt.plot(xj,yj, linewidth=0.5, color='orange')
    plt.scatter([0,0],[0,0], label='Sun', s=40, color='yellow')
    plt.scatter(x[-1],y[-1], s=40, color='blue', label='Earth')
    plt.scatter(xj[-1],yj[-1], s=40, color='black', label='Jupiter')
    plt.xlabel('x [AU]', fontsize=12)
    plt.ylabel('y [AU]', fontsize=12)
    plt.legend(loc='upper left')
    plt.title('Three body system cofs for '+str(tend)+'years')
    plt.axis('equal')
# =============================================================================
#     plt.axis([0.99,1.01,-0.01,0.01]) # For zooming
#     plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# =============================================================================
    plt.show()
    fig.savefig('Jupiter_Earth_moving'+str('.png'))
    savedata(body_package,np.vstack((x,xj)),np.vstack((y,yj)),np.vstack((vx,vxj)),np.vstack((vy,vyj)),np.vstack((ax,axj)),np.vstack((ay,ayj)),tend,'_Three_body_cofs')

if task == str(3):
    '''###_Center of mass as origin / Three body system_###__3f1'''
    body_package = [sun,earth,jup,none]
    body_package = center_system(body_package)
    
    # Solve ODE:s [All planets, Center of mass as origin]
    initial_J        = Solver_large(N,h,body_package)
    x,y,vx,vy,ax,ay  = Solver_large.verlet(initial_J)
    
    # Plot
    fig = plt.figure()
    for i in range (len(x)-1):
        plt.plot(x[i,:],y[i,:], linewidth=0.5)
        plt.scatter(x[i,-1],y[i,-1], s=40, label=body_package[i].name)
    plt.xlabel('x [AU]', fontsize=12)
    plt.ylabel('y [AU]', fontsize=12)
    plt.legend(loc='upper left')
    plt.title('Three body system cofm for '+str(tend)+'years')
    plt.axis('equal')
    # =============================================================================
    # plt.axis([0.99,1.01,-0.01,0.01]) # For zooming
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # =============================================================================
    plt.show()
    fig.savefig('Three body system'+str('.png'))
    
    # Save
    savedata(body_package,x,y,vx,vy,ax,ay,tend,'_Three_body_cofm')

if task == str(4):
    '''###_Center of mass as origin / Solar system_###__3f2'''
    body_package = [sun,mer,ven,earth,mar,jup,sat,ura,nep,plu,none]
    body_package = center_system(body_package)
    
    # Solve ODE:s [All planets, Center of mass as origin]
    initial_J        = Solver_large(N,h,body_package)
    x,y,vx,vy,ax,ay  = Solver_large.verlet(initial_J)

    # Plot
    fig = plt.figure()
    for i in range (len(x)-1):
        plt.plot(x[i,:],y[i,:], linewidth=0.5)
        plt.scatter(x[i,-1],y[i,-1], s=40, label=body_package[i].name) 
    plt.xlabel('x [AU]', fontsize=12)
    plt.ylabel('y [AU]', fontsize=12)
    plt.legend(loc='upper left')
    plt.title('Solar system for '+str(tend)+'years')
    plt.axis('equal')
    plt.axis([-70,70,-70,70])
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    plt.show()
    fig.savefig('Solar system'+str('.png'))

if task == str(5):
    print('Warning extreme calculation time required (23 min)')
    what = input('Continue? [J/N]')
    if what == 'J':
        '''###_Sun as origin / Mercury and Sun system_###__3g'''
        N      = 10**8                           # Gridpoints
        t0     = 0                               # Time zero [Years]
        tend   = 100                             # End time  [Years]
        h      = (tend-t0)/N                     # Steplength
        x0     = 0.3075                          # Initial x-position (1[AU] from sun along x-axis)
        y0     = 0                               # Initial y-position
        vx0    = 0                               # Initial velocity in x-direction
        vy0    = 12.44                           # Initial velocity in y-direction
        
        # Run ODE:s with relativistic term
        body_package         = [sun,mer,none]
        initial              = Solver_small(N,h,x0,y0,vx0,vy0,body_package,beta,1)
        x,y,vx,vy,ax,ay,Time = Solver_small.verlet(initial)
        
        # For saving 23 minutes of life
        # =============================================================================
        # x = np.load('XVALS_10_8.npy')
        # y = np.load('YVALS_10_8.npy')
        # perihelion = np.load('per2.npy')
        # perihelion = np.trim_zeros(perihelion)
        # =============================================================================
        
        # Find all perihelion points
        r0 = 0      # -1 point distance to Sun
        r00 = 0     # -2 point distance to Sun
        K = 0
        rev = 415   # Nr of orbits
        perihelion = np.zeros(rev)
        for i in range(0,len(x)):
            distance = np.sqrt(x[i]**2+y[i]**2) 
            if distance > r0 and r0 < r00:
                perihelion[K] = (i-1)               # Save as perihelion point
                K += 1
            r00 = r0
            r0 = distance
        
        # Calculates angles of perihelion points 
        theta = np.zeros(len(perihelion))
        revs = np.zeros(len(perihelion))
        for i in range(len(perihelion)):
            revs[i] = i
            G = int(perihelion[i])
            theta[i]   = np.arctan(y[G]/x[G])       # Calculate angle
        theta = theta.T*(206265)                    # Convert to Arcsec
        perihelion_precession = round(((theta[-1]+theta[-2]+theta[-3]))/3,2)
        
        # Plot
        fig = plt.figure()
        plt.plot(revs,theta) 
        plt.title('Perhelion Precession Angle')
        plt.ylabel('theta(p) [Arcsec]', fontsize=12)
        plt.xlabel('Orbits around Sun', fontsize=12)
        plt.show()
        fig.savefig('Mercury_perhelion3'+str('.png'))
        
        G = int(perihelion[-1])
        fig = plt.figure()
        plt.plot(x[-100000:-1],y[-100000:-1], linewidth=1, label='100 year orbit')
        plt.plot([0,x[G]],[0,y[G]], linewidth=1, color='black')
        plt.plot([0,x[G]],[0,y[0]], linewidth=1, color='black')
        plt.scatter(x[G],y[G], s=40, label='Last Perhelion Point') 
        plt.scatter(x[0],y[0], s=40, label='Initial Perhelion Point') 
        plt.scatter(0,0, s=60, label='Sun', color='yellow') 
        plt.xlabel('x [AU]', fontsize=12)
        plt.ylabel('y [AU]', fontsize=12)
        plt.legend(loc='upper left')
        plt.title('Mercury Perhelion Precession '+str(tend)+' years')
        plt.text(0.10, 0.000007, ')', fontsize=25)
        plt.text(0.15, 0.000015, '~'+str(perihelion_precession)+' Arcsec', fontsize=15)
        plt.axis([-0.05,0.35,-0.00006,0.0001])
        plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
        plt.show()
        fig.savefig('Mercury_perhelion2'+str('.png'), bbox_inches = "tight")