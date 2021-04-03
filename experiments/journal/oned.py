import numpy as np
from hdmm.workload import Identity, AllRange, Prefix, EkteloMatrix, WidthKRange, Permuted, AllNormK
from hdmm.templates import PIdentity, YuanConvex
from hdmm import error
from ektelo.client.selection import H2, Wavelet, HB, Wavelet, GreedyH
import pandas as pd
from IPython import embed

def rootmse(W, A):
    WtW = W.gram().dense_matrix()
    AtA = A.T.dot(A)
    delta2 = np.diag(AtA).max()
    #tse = (WtW @ AtA.pinv()).trace() * delta2
    tse = np.trace(WtW @ np.linalg.inv(AtA)) * delta2
    return np.sqrt(tse / W.shape[0])

approx = True
HDMM = 'OPT0' #'YuanConvex' if approx else 'PIdentity'
if not approx:
    rootmse = error.rootmse

base = 'strategies'

results = pd.DataFrame(columns=['domain', 'workload', 'Identity', 'H2', 'Wavelet', 'HB', 'GreedyH', HDMM, 'SVDB'])

workloads = {}
workloads['all-range'] = AllRange
workloads['prefix'] = Prefix
workloads['width32'] = lambda n: WidthKRange(n, 32)
workloads['permuted'] = lambda n: Permuted(AllRange(n))
workloads['norm32'] = lambda n: AllNormK(n, 32)

idx = 0

for n in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    for workload, matrix in workloads.items():
        W = matrix(n)
        A1 = Identity(n)
        A2 = H2((n,)).select()
        A3 = Wavelet((n,)).select()
        A4 = HB((n,)).select()
        A5 = GreedyH((n,), W).select()
        if approx:
            A6 = EkteloMatrix(np.load('%s/%s-%d-approx.npy' % (base, workload, n)))
        else:
            A6 = EkteloMatrix(np.load('%s/%s-%d.npy' % (base, workload, n)))

        eigs = np.linalg.eig(W.gram().dense_matrix())[0]
        svdb = np.sqrt(np.maximum(0, np.real(eigs))).sum()**2 / W.shape[1]
        svdb = np.sqrt(svdb / W.shape[0])

        row = [n, workload]
        for A in [A1, A2, A3, A4, A5, A6]:
            A = A.dense_matrix()
            #row.append(error.rootmse(W, A))
            row.append(rootmse(W, A))
        row.append(svdb)
        results.loc[idx] = row
        print(n, workload)
        idx += 1

results.to_csv('oned.csv', index=False)

print(results)

print(results.set_index(['workload', 'domain']).sort_index())
