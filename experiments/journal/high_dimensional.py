import benchmarks
from ektelo import workload
from ektelo.hdmm import error, templates
import argparse
import numpy as np
from IPython import embed

def get_domain(W):
    if isinstance(W, workload.VStack):
        W = W.matrices[0]
    if isinstance(W, workload.Weighted):
        W = W.base
    return tuple(Wi.shape[1] for Wi in W.matrices)

def svdb(W):
    ans = []
    for QtQ in W.gram().matrices:
        svdb = 1
        for sub in QtQ.matrices:
            eigs = np.maximum(0, np.real(np.linalg.eigvalsh(sub.dense_matrix())))
            tmp = np.sqrt(eigs).sum()**2 / sub.shape[1]
            svdb *= tmp
        ans.append(svdb)
    print(ans)
    svdb = np.sqrt(ans).sum()**2
    return np.sqrt(svdb / W.shape[0])

if __name__ == '__main__':

    approx = True
    
    #W = benchmarks.cps()[0]
    #W = benchmarks.adult()[1]
    #W = benchmarks.adult_big()
    #W = benchmarks.census()
    W = benchmarks.census2()
    print(len(W.matrices))
    if isinstance(W, workload.Kronecker):
        W = workload.VStack([W])
  
    ns = get_domain(W)
    print('Domain', ns)
    #ps = [max(n//16, 1) for n in ns]
    #ps = [6,6,6,6,6,1,1,1,1,1,1,1]
    
    """
    unions = []

    delta = len(W.matrices)**2
    prof = 0
    for Wi in W.matrices:
        union = []
        Ai = []
        for j in range(len(ns)):
            n = ns[j]
            Wij = Wi.matrices[j]
            if isinstance(Wij, workload.Ones):
                Ai.append(workload.Total(n))
                union.append(templates.Total(n))
            else:
                Ai.append(workload.Identity(n))
                union.append(templates.PIdentity(ps[j], ns[j]))
        unions.append(templates.Kronecker(union))
        Ai = workload.Kronecker(Ai)
        X = Wi.gram() @ Ai.gram().pinv()
        prof += X.trace()
                
    baseline = np.sqrt(delta * prof / W.shape[0])
    """

    temp1 = templates.DefaultKron(ns, approx)
    temp2 = templates.DefaultUnionKron(ns, len(W.matrices), approx)
    temp3 = templates.Marginals(ns, approx)

    A1, loss1 = temp1.restart_optimize(W, 5)
    A2, loss2 = temp2.restart_optimize(W, 5)
    A3, loss3 = temp3.restart_optimize(W, 5)

    print(error.rootmse(W, A1))

    #temp = templates.BestHD(ns, len(W.matrices))
    #A, loss = temp.restart_optimize(W, 5)

    if approx:
        baseline = 0
        for Q in W.matrices:
            baseline += np.prod([K.diag().max() for K in Q.gram().matrices])
        baseline = np.sqrt(baseline)
    else:
        baseline = 0
        for Q in W.matrices:
            baseline += Q.sensitivity()
    #print(A)
 
    I = workload.Kronecker([workload.Identity(n) for n in ns])
   
    #err0 = np.sqrt(loss / W.shape[0]) 
    #err1 = error.rootmse(W, A)
    err2 = error.rootmse(W, I)
    #err3 = W.sensitivity()
    #err4 = np.sqrt(W.gram().diag().max())

    print(err2, baseline, np.sqrt(loss1/W.shape[0]), np.sqrt(loss2/W.shape[0]), np.sqrt(loss3/W.shape[0]))
