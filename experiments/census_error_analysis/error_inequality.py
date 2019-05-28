import numpy as np
from workload import *
import templates
from examples.census_workloads import build_workload, __race1
import seaborn as sns
import matplotlib.pyplot as plt


def opt_strategy(workload=None):
    ps = [1, 1, 8, 1, 10]   # hard-coded parameters
    template = templates.KronPIdentity(workload.domain, ps)
    template.optimize(workload)
    return [sub.A for sub in template.strategies]


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
    plt.savefig('error_variation_max.pdf')
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
           title=f'Avg vs Stdev of Error (each dot is a query) over re-optimizations, SF1')
    plt.savefig('error_variation_std.pdf')
    plt.show()


if __name__ == '__main__':

    full = ['P1', 'P3', 'P4', 'P5', 'P8', 'P9', 'P10', 'P11', 'P12', 'P12A_I', 'PCT12','PCT12A_O']
    age_id = ['age_identity']
    w_sf1_full = build_workload(full)

    #augmented_w = list(full.extend(age_id))
    print(full + age_id)
    w_sf1_withage = build_workload(full+age_id)
    w = w_sf1_withage
    print('Query count, full', w.queries)
    A = opt_strategy(w)

    male = np.zeros((1, 2))
    male[0,0]=1

    female = np.zeros((1, 2))
    female[0, 1] = 1

    w_age_total = Kron([Total(2), Total(2), Total(64), Total(17), Identity(115)])
    w_age_male = Kron([Matrix(male), Total(2), Total(64), Total(17), Identity(115)])
    w_age_female = Kron([Matrix(female), Total(2), Total(64), Total(17), Identity(115)])

    err_age_total = w_age_total.per_query_rmse(strategy=A, eps=1)
    # err_age_male = w_age_male.per_query_rmse(strategy=A, eps=1)
    # err_age_female = w_age_female.per_query_rmse(strategy=A, eps=1)

    # for i in range(len(err_age_total)):
    #     print(i, err_age_total[i], err_age_male[i], err_age_female[i])

    # w_races = Kron([Total(2), Total(2), __race1(), Total(17), Total(115)])
    # err_races = w_races.per_query_rmse(strategy=A, eps=1)
    #
    # for i,e in enumerate(err_races):
    #     print(i, e)

    #ax = sns.barplot(x=list(range(len(err_age_total))),y=err_age_total, palette=None,hue=None)

    sns.set()
    ax = sns.scatterplot(x=range(115), y=err_age_total, marker='1')
    plt.show()

    exit()

    plt.bar(x=list(range(len(err_age_total))), height=err_age_total, width=0.2, bottom=None, align='center', data=None)
    plt.xlabel('Age')
    plt.ylabel('Per Query RMSE')
    plt.suptitle(f'Error over Age dimension, SF1-Person Queries')

    plt.savefig('error_balance_age.png')
    plt.show()