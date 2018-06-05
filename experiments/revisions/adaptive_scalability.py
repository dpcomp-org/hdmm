import experiments.utility as utility
import workload
import time
from dpcomp_core.algorithm.dawa import greedyH
import numpy as np
import pandas as pd
import pickle
import templates

if False:
    print 'GreedyH'
    engine = greedyH.greedyH_engine()

    for n in [2**k for k in range(5, 15)]:
        P = workload.Prefix(n)
        x = np.zeros(n)
        t0 = time.time()
        engine.Run(P.WtW, x, 1.0, 0)
        #A = utility.greedyH(P.WtW)
        t1 = time.time()
        print n, t1 - t0 

if True:
    print 'HDMM1D'
    df = pd.read_csv('../results/togus_scalability.csv')
    df = df.set_index('domain').time
    dom = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    restarts = 25
    
    for n in dom:
        pid = templates.PIdentity(n//16, n)
        if n >= 128:
            t_opt = restarts*df[n]
            p = n//16
            A=np.load('/home/ryan/Desktop/strategies/all-range_%d_%d.npy'%(n,p)).astype(np.float32)
            B = A[n:] / np.diag(A[:n])       
            pid.set_params(B.flatten())
        else:
            W = workload.Prefix(n)
            t0 = time.time() 
            pid.restart_optimize(W, restarts)
            t_opt = time.time() - t0
        
        t0 = time.time()
        pid.run_mechanism(np.zeros(n), 1.0)
        t1 = time.time()

        t_run = t1 - t0 
        print n, t_opt + t_run #, t_opt, t_run

if False:
    print 'HDMM3D'
    df = pd.read_csv('../results/togus_scalability.csv')
    df = df.set_index('domain').time
    time3d = pickle.load(open('../results/scalability_running.pkl', 'rb'))[1]
    dom = [32, 64, 128, 256, 512, 1024]
    t_marg = 0.020103
    restarts = 25
    for n in [2,4,8,16]:
        W = workload.Prefix(n)
        W = workload.Kron([W,W,W])
        T = templates.KronPIdentity([n,n,n], [1,1,1])
        T2 = templates.Marginals([n,n,n])
        t0 = time.time()
        T.restart_optimize(W, restarts)
        #T2.restart_optimize(workload.Concat([W]), restarts)
        T.run_mechanism(np.zeros(n**3), 1.0)
        total_time = time.time() - t0 + t_marg * restarts
        print n**3, total_time
    
    for i in range(6):
        n = dom[i]
        if n >= 128:
            t_opt = 3*df[n]
        else:
            T = templates.KronPIdentity([n]*3, [n//16]*3)
            W = workload.Kron([workload.Prefix(n)]*3)
            t_opt = T.optimize(W)['time']
        t_run = time3d[i]
        print n**3, restarts*(t_marg + t_opt) + t_run

if False:
    print 'HDMM 8D'
    restarts = 25
    for n in [11]: #range(2,14):
        dom = tuple([n]*8)
        X = np.zeros(n**8, dtype=np.float32)
        W = workload.DimKMarginals(dom, [0,1,2,3])
        T = templates.Marginals(dom)
        T2 = templates.KronPIdentity(dom, [1]*8)
        #t_opt = T.optimize(W)['time']
        #t_opt2 = T2.optimize(W)['time']
        t0 = time.time()
        T.restart_optimize(W, restarts)
        T2.restart_optimize(W, restarts)
        t_opt = time.time() - t0
        T.set_params(T.get_params().astype(np.float32))
        t0 = time.time()
        T.run_mechanism(X, 1.0)
        t1 = time.time()
        t_run = t1 - t0
        print n**8, t_opt + t_run, t_opt


    n = range(2, 11)
