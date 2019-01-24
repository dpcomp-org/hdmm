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

def query_norms(workload):
    ones = np.ones(workload.domain).flatten()
    return workload.evaluate(ones)


if __name__ == '__main__':

    w_pl94 = build_workload(['P1', 'P8', 'P9', 'P10', 'P11'])

    print('Number of queries', w_pl94.queries)

    a = opt_strategy(w_pl94)

    # array of per-query errors
    eps=1.0
    err = w_pl94.per_query_rmse(a, eps=eps)

    # array of normalized query norms (Total query has norm 1)
    norms = query_norms(w_pl94) / np.prod(w_pl94.domain)

    ax = sns.scatterplot(norms, err, alpha=0.25)
    ax.set(xlabel='Normalized query norm (|Total Query|=1)',
           ylabel='Per Query RMSE',
           title=f'Query RMSE vs Query Norm, Workload=PL94, eps={eps}')
    plt.savefig('rmse_vs_query_norm.pdf')
    plt.show()



