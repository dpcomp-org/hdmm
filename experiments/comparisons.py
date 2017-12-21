import numpy as np
import utility 
from workload import AllRange, Prefix, Permuted
import pandas as pd

base = '/home/ryan/Desktop/strategies/'

results = pd.DataFrame(columns=['domain', 'workload', 'Identity', 'H2', 'Wavelet', 'HB', 'GreedyH', 'PMM'])

workloads = {}
workloads['all-range'] = AllRange
workloads['prefix'] = Prefix
workloads['permuted-range'] = lambda n: Permuted(AllRange(n))

idx = 0

for domain in [128, 256, 512, 1024, 2048, 4096, 8192]:
    for workload, matrix in workloads.items():
        W = matrix(domain)
        Identity = utility.identity(domain)
        H2 = utility.hier(domain, 2)
        Wavelet = utility.wavelet(domain)
        GreedyH = utility.greedyH(W.WtW)
        HB = utility.hb(domain)
        PMM = np.load('%s/%s-%d.npy' % (base, workload, domain))
        row = [domain, workload]
        for A in [Identity, H2, Wavelet, HB, GreedyH, PMM]:
            row.append(utility.rootmse(W.WtW, A, W.queries))
        results.loc[idx] = row
        print domain, workload
        idx += 1

results.to_csv('data-independent.csv', index=False)

print results

print results.set_index(['workload', 'domain']).sort_index()
