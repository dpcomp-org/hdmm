import benchmarks
from hdmm import workload, error, templates
import argparse
import numpy as np
from IPython import embed

def svdb(W):
    ans = []
    for QtQ in W.gram().matrices:
        svdb = 1
        for sub in QtQ.matrices:
            eigs = np.maximum(0, np.real(np.linalg.eigvalsh(sub.dense_matrix())))
            tmp = np.sqrt(eigs).sum()**2 / sub.shape[1]
            svdb *= tmp
        ans.append(svdb)
    svdb = np.sqrt(ans).sum()**2
    return np.sqrt(svdb / W.shape[0])

if __name__ == '__main__':

    for dataset in ['census','cps','adult','loans']:
        for work in [1,2]:
            for approx in [False, True]:
                W = benchmarks.get_workload(dataset, work)

                if approx:
                    baseline = 0
                    for Q in W.matrices:
                        w = 1.0
                        if isinstance(Q, workload.Weighted):
                            w = Q.weight
                            Q = Q.base
                        baseline += w*np.prod([K.diag().max() for K in Q.gram().matrices])
                    baseline = np.sqrt(baseline)
                else:
                    baseline = 0
                    for Q in W.matrices:
                        baseline += Q.sensitivity()
             
                err2 = sum(Wi.gram().trace() for Wi in W.matrices) #error.rootmse(W, I)
                err2 = np.sqrt(err2 / W.shape[0])

                print(dataset, work, approx, err2, baseline)#, svdb(W))
