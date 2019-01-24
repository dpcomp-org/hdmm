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
    assumed_total_population = 1000
    scaled_query_norms = norms * assumed_total_population

    # plot of relative error distribution, assuming uniform data (query answer is query norm)
    ax=sns.distplot(err / scaled_query_norms, bins=20, kde=False, norm_hist=False)

    ax.set(xlabel='Relative Error',
           ylabel='Number of Queries',
           title=f'Distribution of Relative Errors, Workload=PL94, eps={eps}, pop={assumed_total_population}')
    plt.savefig('relative_error_uniform_data.pdf')

    plt.show()


