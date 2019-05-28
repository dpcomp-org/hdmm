import numpy as np
from hdmm.workload import *
from hdmm import templates, error
from census_workloads import SF1_Persons


# helper function
def opt_strategy(workload=None):
    ns = [2,2,64,17,115]
    ps = [1, 1, 8, 1, 10]   # hard-coded parameters
    template = templates.KronPIdentity(ps, ns)
    template.optimize(workload)
    return template.strategy()


if __name__ == '__main__':

    # define workload
    w_sf1_persons = SF1_Persons()

    print('Number of queries in workload:', w_sf1_persons.shape[0])

    # compute strategy
    a_sf1_persons = opt_strategy(w_sf1_persons)

    # compute errors of workload queries using strategy, for given epsilon
    errors = np.sqrt(error.per_query_error(w_sf1_persons, a_sf1_persons, eps=1.0))

    # print error histogram
    print('RMSE histogram')
    hist = np.histogram(errors)
    for b, c in zip(hist[1], hist[0]):
        print('{} \t {}'.format(b, c))
