import numpy as np
from hdmm.workload import *
from hdmm import templates, error
from census_workloads import build_workload
#import seaborn as sns
import matplotlib.pyplot as plt
from IPython import embed

def opt_strategy(workload=None):
    ns = [2,2,64,17,115]
    ps = [1, 1, 8, 1, 10]   # hard-coded parameters
    template = templates.KronPIdentity(ps, ns)
    template.optimize(workload)
    return template.strategy()



if __name__ == '__main__':

    full = ['P1', 'P3', 'P4', 'P5', 'P8', 'P9', 'P10', 'P11', 'P12', 'P12A_I', 'PCT12','PCT12A_O']

    w_sf1_full = build_workload(full)
    print('Query count, full', w_sf1_full.shape[0])

    stats = {}
    trials = 10 #150
    strategies = [opt_strategy(w_sf1_full) for i in range(trials)]
    errors = [np.sqrt(error.per_query_error(w_sf1_full, a)) for a in strategies]
    stats['rootmse      '] = [error.rootmse(w_sf1_full, a) for a in strategies]
    stats['max_query_err'] = [np.max(e) for e in errors]
    stats['min_query_err'] = [np.min(e) for e in errors]
    stats['mean_query_err'] = [np.mean(e) for e in errors]

    for k in stats.keys():
        print(f'{k} \t {np.mean(stats[k]):.5f} \t {np.std(stats[k]):5f} \t ({np.min(stats[k]):.5f},{np.max(stats[k]):.5f})')

