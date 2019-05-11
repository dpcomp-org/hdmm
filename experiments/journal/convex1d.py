from hdmm.workload import Prefix, AllRange, WidthKRange, Permuted, AllNormK, ExplicitGram
from hdmm.templates import YuanConvex, McKennaConvex
import numpy as np

base = '/Users/rmckenna/Desktop/strategies'

def norm32(n):
    WtW = AllNormK(n, 32).gram().dense_matrix()
    return ExplicitGram(WtW / WtW.min())

workloads = {}
#workloads['all-range'] = AllRange
#workloads['prefix'] = Prefix
#workloads['width32'] = lambda n: WidthKRange(n, 32)
workloads['permuted'] = lambda n: Permuted(AllRange(n))
workloads['norm32'] = norm32

for n in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    for workload, matrix in workloads.items():
        print(workload, n)
        W = matrix(n)
        temp = McKennaConvex(n) #YuanConvex()
        temp.optimize(W) 
        A = temp.strategy().dense_matrix()
        np.save('%s/%s-%d-approx.npy' % (base, workload, n), A)
