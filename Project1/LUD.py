def LUD(M,f):
    start = time.clock()
    LU = linalg.lu_factor(M)
    x = linalg.lu_solve(LU,f)
    end = time.clock()
    print ('CPU time (LUD case):',end-start)
    return(x)