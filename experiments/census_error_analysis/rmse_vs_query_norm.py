import numpy as np
from hdmm.workload import *
from hdmm import templates, error
from census_workloads import build_workload
import seaborn as sns
import matplotlib.pyplot as plt

def opt_strategy(workload=None):
    ns = [2,2,64,17,115]
    ps = [1, 1, 8, 1, 10]   # hard-coded parameters
    template = templates.KronPIdentity(ps, ns)
    template.optimize(workload)
    return template.strategy()

def query_norms(workload):
    ones = np.ones(workload.shape[1]).flatten()
    return workload.dot(ones)


if __name__ == '__main__':

    w_pl94 = build_workload(['P1', 'P8', 'P9', 'P10', 'P11'])

    print('Number of queries', w_pl94.shape[0])

    a = opt_strategy(w_pl94)

    # array of per-query errors
    eps=1.0
    err = np.sqrt(error.per_query_error(w_pl94, a, eps=eps))

    # array of normalized query norms (Total query has norm 1)
    norms = query_norms(w_pl94) / w_pl94.shape[1]

    ax = sns.scatterplot(norms, err, alpha=0.25)
    ax.set(xlabel='Normalized query norm (|Total Query|=1)',
           ylabel='Per Query RMSE',
           title=f'Query RMSE vs Query Norm, Workload=PL94, eps={eps}')
    plt.savefig('rmse_vs_query_norm.pdf')
    plt.show()



