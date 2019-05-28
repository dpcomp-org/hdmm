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


def error_variance_stats(w, trials=5):
    stats = {}
    strategies = [opt_strategy(w) for i in range(trials)]
    stats['rootmse      '] = [w.rootmse(strategy=a) for a in strategies]
    stats['max_query_err'] = [np.max(w.per_query_rmse(strategy=a)) for a in strategies]
    stats['min_query_err'] = [np.min(w.per_query_rmse(strategy=a)) for a in strategies]
    stats['mean_query_err'] = [np.mean(w.per_query_rmse(strategy=a)) for a in strategies]

    for k in stats.keys():
        print(f'{k} \t {np.mean(stats[k]):.5f} \t {np.std(stats[k]):5f} \t ({np.min(stats[k]):.5f},{np.max(stats[k]):.5f})')


def mean_vs_max_error(w, trials=5):
    strategies = [opt_strategy(w) for i in range(trials)]
    query_errors_trials = [w.per_query_rmse(strategy=a) for a in strategies]
    qet_array = np.array(query_errors_trials)

    mean = np.mean(qet_array, axis=0)
    max = np.max(qet_array, axis=0)

    ax = sns.scatterplot(x=mean, y=max)
    ax.set(xlabel='Average Error',
           ylabel='Max Error',
           title=f'Avg vs Max Error (each dot is a query) over re-optimizations, SF1')
    plt.savefig('error_variation_max.png')
    plt.show()


def mean_vs_std_error(w, trials=5):
    strategies = [opt_strategy(w) for i in range(trials)]
    query_errors_trials = [w.per_query_rmse(strategy=a) for a in strategies]
    qet_array = np.array(query_errors_trials)

    mean = np.mean(qet_array, axis=0)
    std = np.std(qet_array, axis=0)

    ax = sns.scatterplot(x=mean, y=std)
    ax.set(xlabel='Average Error',
           ylabel='Stdev Error',
           title=f'Avg vs Stdev of Error over re-optimizations, SF1')
    plt.savefig('error_variation_std.png')
    plt.show()


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

    # mean_vs_max_error(w_sf1_full)
    mean_vs_max_error(w_sf1_full, trials=100)
