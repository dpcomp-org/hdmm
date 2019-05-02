from ektelo.workload import Prefix, AllRange, WidthKRange
from ektelo.hdmm.templates import YuanConvex
import numpy as np

base = '/home/ryan/Desktop/strategies'

workloads = {}
workloads['all-range'] = AllRange
workloads['prefix'] = Prefix
workloads['width32'] = lambda n: WidthKRange(n, 32)

for n in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    for workload, matrix in workloads.items():
        print(workload, n)
        W = matrix(n)
        temp = YuanConvex()
        temp.optimize(W) 
        A = temp.strategy().dense_matrix()
        np.save('%s/%s-%d-yuan.npy' % (base, workload, n), A)
