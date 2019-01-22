import numpy as np
from workload import *
import templates
from census_workloads import SF1_Persons


# helper function
def opt_strategy(workload=None):
    ps = [1, 1, 8, 1, 10]   # hard-coded parameters
    template = templates.KronPIdentity(workload.domain, ps)
    template.optimize(workload)
    return [sub.A for sub in template.strategies]


if __name__ == '__main__':

    # define workload
    w_sf1_persons = SF1_Persons()

    print('Number of queries in workload:', w_sf1_persons.queries)

    # compute strategy
    a_sf1_persons = opt_strategy(w_sf1_persons)

    # compute errors of workload queries using strategy, for given epsilon
    errors = w_sf1_persons.per_query_rmse(strategy=a_sf1_persons, eps=1.0)

    # print error histogram
    print('RMSE histogram')
    hist = np.histogram(errors)
    for b, c in zip(hist[1], hist[0]):
        print('{} \t {}'.format(b, c))
