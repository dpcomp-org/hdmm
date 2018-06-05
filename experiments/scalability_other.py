import numpy as np
from dpcomp_core.algorithm import HB2D, privelet2D, QuadTree, identity
import time
import pandas as pd
from IPython import embed
import workload
import utility

if False:
    methods = {}
    methods['HB'] = lambda X: HB2D.HB2D_engine().Run(None, X, 1.0, 0)
    methods['Privelet'] = lambda X: privelet2D.privelet2D_engine().Run(None, X, 1.0, 0)
    methods['QuadTree'] = lambda X: QuadTree.QuadTree_engine().Run(None, X, 1.0, 0)
    methods['Identity'] = lambda X: identity.identity_engine().Run(None, X, 1.0, 0)

    active = { name : True for name in ['HB', 'Privelet', 'QuadTree', 'Identity'] }

    df = pd.DataFrame(columns=['Domain', 'Mechanism', 'Time'])
    idx = 0
    time_limit = 40

    for dom in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 2**14]:
        X = np.zeros((dom, dom))
        for name, mech in methods.items():
            if active[name]:
                t0 = time.time()
                mech(X)
                t1 = time.time()
                df.loc[idx] = [dom**2, name, t1-t0]
                idx += 1
                print dom, name, t1-t0
            if t1 - t0 > time_limit: 
                active[name] = False

    summary = df.set_index(['Domain', 'Mechanism']).unstack('Mechanism')
    summary.columns = summary.columns.droplevel(0)
    summary.to_csv('results/scalability_other.csv')

if True:
    opt = pd.DataFrame(columns=['Domain', 'Mechanism', 'Time'])
    idx = 0
    for n in [2**k for k in range(5, 15)]:
        WtW = workload.AllRange(n).WtW
        t0 = time.time()
        utility.greedyH(WtW)
        t1 = time.time()
        opt.loc[idx] = [n, 'GreedyH', t1-t0]
        print n, t1-t0
        idx += 1 

    opt.to_csv('results/scalability_other_opt.csv')
