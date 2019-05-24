import numpy as np
from hdmm import workload, error
from hdmm.workload import AllRange, Prefix, WidthKRange, Permuted
import matlab
import matlab.engine
from IPython import embed

workloads = {}
workloads['width32'] = lambda n: WidthKRange(n, 32)
workloads['all-range'] = AllRange
#workloads['prefix'] = Prefix
#workloads['permuted'] = lambda n: Permuted(AllRange(n))

eng = matlab.engine.start_matlab()

embed()

for n in [64,128,256,512,1024,4096,2048,8192]:
    for workload, matrix in workloads.items():
        W = matrix(n)
        if W.shape[0] * W.shape[1] >= 10**8:
            print('skipping', workload, n)
            continue
        WW = matlab.double(list([[float(c) for c in r] for r in W.dense_matrix()]))    

        AA = eng.LowRankDP(WW, nargout=2)[1]
        A = np.array(AA)
        np.save('/home/ryan/Desktop/strategies/%s-%d-lrm.npy' % (workload, n), A)
        print('completed', workload, n)

eng.quit()
