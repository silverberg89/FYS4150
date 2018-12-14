import sys
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import time
import methods_5
from MC_para_5 import MC
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

def run():
    hade       = Pool(5)
    # Relationship factors
    cycles      = int(1*10**3)
    trades      = int(5*10**5)
    N           = 1000
    m0          = 1
    lmbd        = 0.5
    gamma       = 0
    alpha       = 2
    gamma_vec   = [0.0,1.0,2.0,3.0,4.0]
    #alpha_vec = [0.5,1.0,1.5,2.0]
    #lmbd_vec    = [0,0.25,0.5,0.9]
    val_nr = 5
    
    print('MC starttime: ',datetime.now())
    print('Estimated comp.time [hours]:', np.round(methods_5.func_time(trades)*val_nr,decimals=3))
    t0 = time.clock()
    # Run Monte Carlo simulation
    monte = MC(cycles,trades,N,m0)
    vals = hade.starmap(monte.mc,[(alpha,lmbd,gamma_vec[0]),
                                   (alpha,lmbd,gamma_vec[1]),
                                   (alpha,lmbd,gamma_vec[2]),
                                   (alpha,lmbd,gamma_vec[3]),
                                   (alpha,lmbd,gamma_vec[4])])
    #np.save('vals_para_'+str(alpha)+'.npy',vals)
    np.save('vals_para_'+str(alpha)+'.npy',vals)
    print('Computation time: ',time.clock()-t0)
    hade.close()
    hade.join()
    
if __name__=='__main__':
    __spec__ = None
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # Create the pool
    gogo = Pool(5)
    time.sleep(2)
    gogo.close()
    gogo.join()
    run()
