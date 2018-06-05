import dpcomp_core.algorithm.dawa as dawa
from dpcomp_core.workload import Prefix1D, Workload
from dpcomp_core.query_nd_union import ndRangeUnion
from dpcomp_core.dataset import DatasetFromFile
import numpy as np

if False:

    n = 1024

    x = DatasetFromFile('PATENT', n).payload

    #x = np.load('fnlwgt.npy')

    if n == 128:
        A = np.kron(np.eye(128), np.ones(8192//128))
        x = A.dot(x).astype(int)

    ranges = [ndRangeUnion().add1DRange(i, i+31) for i in range(n-32+1)]

    Q = Prefix1D(n)
    Q = Workload(ranges, (n,))
    P = Q.get_matrix('dense')

    error = 0.00

    trials = 25
    for i in range(trials):
        print i
        xest = dawa.dawa_engine().Run(Q, x, 1.0, i)
        diff = P.dot(x-xest)
        error += np.dot(diff, diff) / trials

    baseline = 2 * np.trace(np.dot(P.T, P))

    print np.sqrt(error / baseline)


### 2D experiment
if True:
    n = 256
    x = DatasetFromFile('BEIJING-CABS-E').payload
    ranges = [ndRangeUnion().addRange((0,0), (i,j)) for i in range(n) for j in range(n)]
    Q = Workload(ranges, (n,n))

    error = 0.00

    trials = 25
    for i in range(trials):
        print i
        xest = dawa.dawa2D_engine().Run(Q, x, 1.0, i)
        diff = P.dot(x-xest)
        error += np.dot(diff, diff) / trials
