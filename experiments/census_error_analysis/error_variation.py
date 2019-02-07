import numpy as np
from workload import *
import templates
from examples.census_workloads import build_workload
import seaborn as sns
import matplotlib.pyplot as plt


def opt_strategy(workload=None):
    ps = [1, 1, 8, 1, 10]   # hard-coded parameters
    template = templates.KronPIdentity(workload.domain, ps)
    template.optimize(workload)
    return [sub.A for sub in template.strategies]



if __name__ == '__main__':

    full = ['P1', 'P3', 'P4', 'P5', 'P8', 'P9', 'P10', 'P11', 'P12', 'P12A_I', 'PCT12','PCT12A_O']

    w_sf1_full = build_workload(full)
    print('Query count, full', w_sf1_full.queries)

    stats = {}
    trials = 150
    strategies = [opt_strategy(w_sf1_full) for i in range(trials)]
    stats['rootmse      '] = [w_sf1_full.rootmse(strategy=a) for a in strategies]
    stats['max_query_err'] = [np.max(w_sf1_full.per_query_rmse(strategy=a)) for a in strategies]
    stats['min_query_err'] = [np.min(w_sf1_full.per_query_rmse(strategy=a)) for a in strategies]
    stats['mean_query_err'] = [np.mean(w_sf1_full.per_query_rmse(strategy=a)) for a in strategies]

    for k in stats.keys():
        print(f'{k} \t {np.mean(stats[k]):.5f} \t {np.std(stats[k]):5f} \t ({np.min(stats[k]):.5f},{np.max(stats[k]):.5f})')

