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

    include = ['P1', 'P3', 'P4', 'P5', 'P8', 'P9', 'P10', 'P11', 'P12', 'P12A_I', 'PCT12']
    exclude = ['PCT12A_O']   # consider omitting these tables
    full = include + exclude

    w_sf1_full = build_workload(full)
    print('Query count, full', w_sf1_full.queries)
    a_sf1_full = opt_strategy(w_sf1_full)

    w_sf1_include = build_workload(include)
    print('Query count, include', w_sf1_include.queries)
    a_sf1_include = opt_strategy(w_sf1_include)

    w_sf1_exclude = build_workload(exclude)

    # RMSE per query, under full strategy
    err_w_full_a_full = w_sf1_full.per_query_rmse(strategy=a_sf1_full)
    err_w_inc_a_full = w_sf1_include.per_query_rmse(strategy=a_sf1_full)
    err_w_exc_a_full = w_sf1_exclude.per_query_rmse(strategy=a_sf1_full)

    # RMSE per query, under include strategy
    err_w_full_a_inc = w_sf1_full.per_query_rmse(strategy=a_sf1_include)
    err_w_inc_a_inc = w_sf1_include.per_query_rmse(strategy=a_sf1_include)
    err_w_exc_a_inc = w_sf1_exclude.per_query_rmse(strategy=a_sf1_include)

    print('strategy \t w_full \t w_include \t w_exclude')
    print(f'w_include \t {np.mean(err_w_full_a_inc):.4f} \t {np.mean(err_w_inc_a_inc):.4f} \t {np.mean(err_w_exc_a_inc):.4f}')
    print(f'w_full    \t {np.mean(err_w_full_a_full):.4f} \t {np.mean(err_w_inc_a_full):.4f} \t {np.mean(err_w_exc_a_full):.4f}')


    error_ratio = err_w_inc_a_inc / err_w_inc_a_full

    hist = np.histogram(error_ratio, bins=[.1, .25, .5,.75, 1, 1.5, 2,5,10,100,1000,10000])

    print('Error ratio histogram')
    for b, c in zip(hist[1], hist[0]):
        print('{}  {}'.format(b, c))






# class Wkld_comparator(object):
#
#     def __init__(self, w1, w2):
#         self.w1 = w1
#         self.w2 = w2
#         rand_vec = np.random.random_sample(w1.domain).flatten()
#         self.w1_hash = w1.evaluate(rand_vec)
#         self.w2_hash = w2.evaluate(rand_vec)
#
#     def compare(self, i, j):
#         return self.w1_hash[i] == self.w2_hash[j]
#
#     def all_matches(self):
#         pass

