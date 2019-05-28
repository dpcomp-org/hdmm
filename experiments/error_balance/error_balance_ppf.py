import numpy as np
from workload import *
import templates, more_templates
import itertools
from examples.census_workloads import build_workload
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def opt_row_weighted(workload, rows):
    A = more_templates.RowWeighted(rows)
    A.optimize(workload)
    return A.A

def opt_kron(workload=None, p=4):
    ps = [p, p]   # hard-coded parameters
    template = templates.KronPIdentity(workload.domain, ps)
    #template = templates.Marginals(workload.domain)
    template.optimize(workload)
    return [sub.A for sub in template.strategies]

def opt_p_identity(workload=None, p=4):
    pid = templates.PIdentity(p, workload.domain)
    pid.optimize(workload)
    return pid.A

def parse_geo():
    cols = ['STATE', 'STATEFP', 'COUNTYFP', 'COUNTYNAME', 'CLASSFP']
    data = pd.read_csv('national_county.txt', names=cols)
    counties = data.groupby('STATE').COUNTYNAME.count().sort_values()
    return counties.values




class WidthKRangeNormWeighted(Workload):
    def __init__(self, domain, widths):
        """
        Width K Range Queries for Sliding Average
        :param domain: The domain size
        :param widths: the width of the queries (int or list of ints)
        """
        self.domain = domain
        if type(widths) is int:
            widths = [widths]
        self.widths = widths
        self.queries = sum(domain-k+1 for k in widths)

    @property
    def W(self):
        m, n = self.queries, self.domain
        W = np.zeros((m, n))
        row = 0
        for k in self.widths:
            for i in range(n-k+1):
                W[row+i, i:i+k] = k
            row += n - k + 1
        return W




def dev(err_vector):
    mean = np.mean(err_vector)
    return sum(abs(err_vector - mean)) / len(err_vector)

def max_minus_min(err_vector):
    return (np.max(err_vector) - np.min(err_vector)) / np.mean(err_vector)

def total_query_rmse(err_vector):
    return -err_vector.mean()


# def dictify(W_target, W_bal, strategy_dict, metric_err, metric_bal):
#     d = []
#     for k in strategy_dict.keys():
#         errs_on_target = [W_target.per_query_rmse(strategy=a, eps=1) for a in strategy_dict[k]]
#         errs_on_bal = [W_bal.per_query_rmse(strategy=a, eps=1) for a in strategy_dict[k]]
#         [print(np.min(e), np.max(e)) for e in errs_on_bal]
#         d += [dict(label=k, err=metric_err(e_tar), bal=metric_bal(e_bal)) for (e_tar, e_bal) in zip(errs_on_target, errs_on_bal)]
#     return pd.DataFrame(d)


def dictify(W_target, W_bal, strategy_dict, metric_err, metric_bal):
    d = []
    for k in strategy_dict.keys():
        print('.')
        for a in strategy_dict[k]:
            err_on_target = -W_target.rootmse(strategy=a, eps=1)
            print(err_on_target)
            errs_on_bal = W_bal.per_query_rmse(strategy=a, eps=1)
            print(np.min(errs_on_bal), np.max(errs_on_bal))
            d.append(dict(label=k, err=err_on_target, bal=metric_bal(errs_on_bal)))
    return pd.DataFrame(d)


def balance_plot_1d_ranges():

    d = 256 # dimension size

    # workloads
    widths = [32, 40, 48, 56, 64, 128, 256]
    W_target = WidthKRange(d, widths)
    W_target = Concat([Prefix(d), Identity(d), W_target])
    W_bal = Identity(d)

    # strategies
    A = {}
    trials = 1
    A['HDMM (varying p)'] = [opt_p_identity(W_target, p) for p in [1, 2, 4, 8, 16, 32, 64]*trials]
    #A['Workload+Inf'] = [W_target.W]

    Wnorm = WidthKRangeNormWeighted(d, widths)
    A['HDMM norm wgt (varying p)'] = [opt_p_identity(Wnorm, p) for p in [1, 2, 4, 8, 16, 32, 64] * trials]

    A['Identity'] = [np.eye(W_target.domain)]

    df = dictify(W_target, W_bal, A, total_query_rmse, max_minus_min)
    print(df)
    sns.set()
    ax = sns.scatterplot(x='bal', y='err', hue='label', data=df)
    ax.set(xlabel='Error Imbalance on W_bal',
           ylabel='Error on W_target',
           title=f'Error vs. Imbalance, W_target=1D Ranges, W_bal=Identity')
    plt.savefig('error_balance_1d_ranges.png')
    plt.show()

    # return per query errors of least total error for plotting
    min_error_A = opt_p_identity(W_target, 32)
    return W_bal.per_query_rmse(strategy=min_error_A, eps=1)



def balance_plot_2d_ranges():

    d = 64  # dimension size

    # workloads
    ar = AllRange(d)
    W_target = Kron([AllRange(d), AllRange(d)])
    W_bal = Kron([Identity(d), Identity(d)])

    # strategies
    A = {}
    trials = 5
    A['HDMM (varying p)'] = [opt_kron(W_target, p) for p in [1, 2, 4, 8, 16] * trials]

    #Wnorm = WidthKRangeNormWeighted(d, widths)
    #A['HDMM norm wgt (varying p)'] = [opt_p_identity(Wnorm, p) for p in [1, 2, 4, 8, 16, 32, 64] * trials]

    #A['Identity'] = [templates.Kronecker([templates.Identity(d), templates.Identity(d)])]
    A['Identity'] = [[np.eye(d), np.eye(d)]]

    df = dictify(W_target, W_bal, A, total_query_rmse, max_minus_min)
    print(df)
    sns.set()
    ax = sns.scatterplot(x='bal', y='err', hue='label', data=df)
    ax.set(xlabel='Error Imbalance on W_bal',
           ylabel='Error on W_target',
           title=f'Error vs. Imbalance, W_target=({d},{d}) AllRange, W_bal=Identity')
    plt.savefig('error_balance_2d_allrange.png')
    plt.show()


def balance_geo_counties():

    counties = parse_geo()

    N = np.sum(counties)
    level0 = np.ones((1, N))    # total
    level1 = sparse.block_diag([np.ones(n) for n in counties]).toarray()    # state level
    level2 = np.eye(N)  # counties

    def union_levels(wgt=(1,1,1)):
        return Concat([Matrix(W) for W in [level0*wgt[0], level1*wgt[1], level2*wgt[2]]])

    W_target = union_levels()

    W_counties = Matrix(level2)
    W_states = Matrix(level1)

    # comment one of these:
    #W_bal, bal = W_counties, 'W_counties'
    W_bal, bal = W_states, 'W_states'

    # strategies
    A = {}
    trials = 1

    use_cache = False
    if use_cache:
        with open('geo_strategies.pickle', 'rb') as handle:
            A['HDMM'] = pickle.load(handle)
    else:
        A['HDMM'] = [opt_row_weighted(W_target, level1), opt_row_weighted(union_levels((1,10,1)), level1)]
        with open('geo_strategies.pickle', 'wb') as handle:
            pickle.dump(A['HDMM'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    A['Workload+Inf'] = [W_target.W]
    A['I'] = [np.eye(N)]
    A['state-level-weighted'] = [union_levels((1,w,1)).W for w in [2, 5]]



    df = dictify(W_target, W_bal, A, total_query_rmse, max_minus_min)
    print(df)
    sns.set_style("darkgrid")
    ax = sns.scatterplot(x='bal', y='err', style='label', data=df)
    ax.set(xlabel='Error Imbalance on W_bal',
           ylabel='Error on W_target',
           title=f'Error vs. Imbalance, W_target=County-State-Natl, W_bal={bal}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
    plt.tight_layout()
    plt.savefig(f'error_balance_geo_{bal}.png')
    plt.show()

    # return per query errors of least total error for plotting
    min_error_A = A['HDMM'][0]
    return W_bal.per_query_rmse(strategy=min_error_A, eps=1)


def balance_redistricting():

    w = np.load('MA_Precincts_12_16.npy')
    print(w.shape)
    exit()

    laplace, workload, identity, hdmm = np.load('redistricting_errors.npz').values()
    hdmm = np.array(sorted(hdmm))
    print(hdmm.shape)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 2))
    pal = sns.color_palette("Greens_d", 1)
    ax = sns.lineplot(x=list(range(len(hdmm))), y=sorted(hdmm), palette=pal, markers=False)
    ax.get_xaxis().set_ticklabels([])
    plt.savefig(f'redistricting_hdmm_errors.png')
    plt.show()



if __name__ == '__main__':

    balance_redistricting()

    exit()


    err = balance_geo_counties()

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 2))
    pal = sns.color_palette("Greens_d", 1)
    ax = sns.barplot(x=list(range(len(err))), y=sorted(err), palette=pal)
    ax.get_xaxis().set_ticklabels([])
    plt.savefig(f'error_balance_geo_errors.png')
    plt.show()

    print(min(err), np.mean(err), max(err))

    exit()

    err = balance_plot_1d_ranges()

    sns.set()
    ax = sns.scatterplot(x=range(len(err)), y=err)

    plt.show()

    exit()


