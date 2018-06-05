import numpy as np
import workload
from dpcomp_core.algorithm import HB2D, privelet2D, QuadTree, identity, privelet, HB, dawa
from dpcomp_core.workload import Prefix1D
from IPython import embed
#import optimize
#import templates
import pandas as pd
import time

df = pd.DataFrame(columns=['Workload', 'Domain', 'Mechanism', 'Time'])
idx = 0

if False:
    for dom in [64, 1024]:
        X = np.zeros((dom, dom))
        t0 = time.time()
        HB2D.HB2D_engine().Run(None, X, 1.0, 0)
        t1 = time.time()
        privelet2D.privelet2D_engine().Run(None, X, 1.0, 0)
        t2 = time.time()
        QuadTree.QuadTree_engine().Run(None, X, 1.0, 0)
        t3 = time.time()
        identity.identity_engine().Run(None, X, 1.0, 0)
        t4 = time.time()
        df.loc[idx] = ['2D Range', dom, 'HB', t1-t0]
        df.loc[idx+1] = ['2D Range', dom, 'Privelet', t2-t1]
        df.loc[idx+2] = ['2D Range', dom, 'QuadTree', t3-t2]
        df.loc[idx+3] = ['2D Range', dom, 'Identity', t4-t3]
        idx += 4


    for dom in [128, 8192]:
        X = np.load('fnlwgt.npy')
        if dom == 128:
            A = np.kron(np.eye(128), np.ones(8192//128))
            X = A.dot(X).astype(int)

        Q = Prefix1D(dom)
        t0 = time.time()
        identity.identity_engine().Run(None, X, 1.0, 0)
        t1 = time.time()
        HB.HB_engine().Run(None, X, 1.0, 0)
        t2 = time.time()
        privelet.privelet_engine().Run(None, X, 1.0, 0)
        t3 = time.time()
        dawa.dawa_engine().Run(Q, X, 1.0, 0)
        t4 = time.time()
        dawa.greedyH_only_engine().Run(Q, X, 1.0, 0)
        t5 = time.time()
        df.loc[idx] = ['1D Prefix', dom, 'Identity', t1-t0]
        df.loc[idx+1] = ['1D Prefix', dom, 'HB', t2-t1]
        df.loc[idx+2] = ['1D Prefix', dom, 'Privelet', t3-t2]
        df.loc[idx+3] = ['1D Prefix', dom, 'DAWA', t4-t3]
        df.loc[idx+4] = ['1D Prefix', dom, 'GreedyH', t5-t4]
        idx += 5

X = np.zeros((2,2,64,17,115), dtype=int)
t0 = time.time()
identity.identity_engine().Run(None, X, 1.0, 0)
t1 = time.time()
df.loc[idx] = ['Census SF1', None, 'Identity', t1-t0]

X = np.zeros((2,2,64,17,115,51), dtype=int)
t0 = time.time()
identity.identity_engine().Run(None, X, 1.0, 0)
t1 = time.time()
df.loc[idx+1] = ['Census SF1+', None, 'Identity', t1-t0]

X = np.zeros((100,50,7,4,2), dtype=int)
t0 = time.time()
identity.identity_engine().Run(None, X, 1.0, 0)
t1 = time.time()
df.loc[idx+2] = ['CPS (Any)', None, 'Identity', t1-t0]

X = np.zeros((75,16,5,2,20), dtype=int)
t0 = time.time()
identity.identity_engine().Run(None, X, 1.0, 0)
t1 = time.time()
df.loc[idx+3] = ['Adult (Any)', None, 'Identity', t1-t0]




print df




