from hdmm import workload, templates, error
import numpy as np

from workload_analysis import util


'''
This is a firm_workload workload based on the Census "Business Dynamics Statistics" product

This workload corresponds to the Firms tables; see p52 of technical manual for reference

'''

bds_schema = {
    # 'column_name' : domain_size
    'AGE4' : 11,
    'FAGE4': 11,    # Firm Age
    'SIZE' : 9,
    'ISIZE' : 9,
    'FSIZE' : 9,
    'IFSIZE': 9,
    'METRO': 2,
    'MSA': 367,
    'SIC1': 9,      # Industry
    'STATE': 51
}

domain = list(bds_schema.values())
variables = list(bds_schema.keys())

firm_workload = {
    () : 1,
    ('SIC1',): 1,
    ('IFSIZE',) : 1,
    ('FSIZE',) : 1,
    ('FAGE4',) : 1,
    ('STATE',) : 1,
    ('METRO',) : 1,
    ('MSA',): 1,
    ('IFSIZE', 'SIC1') : 1,
    ('FSIZE', 'SIC1'): 1,
    ('FAGE4', 'SIC1'): 1,
    ('FAGE4', 'IFSIZE'): 1,
    ('FAGE4', 'FSIZE'): 1,
    ('IFSIZE', 'STATE'): 1,
    ('FSIZE', 'STATE'): 1,
    ('FAGE4', 'STATE'): 1,
    ('IFSIZE', 'METRO'): 1,
    ('FSIZE', 'METRO'): 1,
    ('FAGE4', 'METRO'): 1,
    ('FSIZE', 'MSA'): 1,
    ('FAGE4', 'MSA'): 1,
    ('FAGE4','FSIZE','STATE', 'METRO'): 1,
    ('FAGE4', 'IFSIZE', 'STATE', 'METRO'): 1,
    ('FAGE4', 'IFSIZE', 'SIC1'): 1,
    ('FAGE4', 'FSIZE', 'SIC1'): 1,
    ('FAGE4', 'IFSIZE', 'STATE'): 1,
    ('FAGE4', 'FSIZE', 'STATE'): 1,
    ('FAGE4', 'IFSIZE', 'METRO'): 1,
    ('FAGE4', 'FSIZE', 'METRO'): 1,
    ('FAGE4', 'FSIZE', 'MSA'): 1,
}


def wkld_census_bds_firms():
    W = workload.Marginals.fromtuples(domain, firm_workload, columns=variables)
    return W, domain


def wkld_marginals_weighted():
    m = firm_workload.copy()
    for k in m.keys():
        wgt = np.prod([bds_schema[x] for x in k])
        wgt = np.sqrt(wgt) / 2.0
        m[k] = 1.0/wgt
    return m

def manual_strategy():
    w_dict = {
        ('FAGE4', 'FSIZE', 'STATE', 'METRO'): 1,    #
        ('FAGE4', 'IFSIZE', 'STATE', 'METRO'): 1,   #
        ('FAGE4', 'IFSIZE', 'SIC1'): 1,             #
        ('FAGE4', 'FSIZE', 'SIC1'): 1,              #
        ('FAGE4', 'FSIZE', 'MSA'): 1,               #
    }
    return workload.Marginals.fromtuples(domain, w_dict, columns=variables)

def summarize_strategy(W, A, domain):
    # helper function
    # print table of firm_workload in binary, workload weights, and strategy weights
    for i in range(2**len(domain)):
        if W.weights[i] > 0 or A._params[i] > 0:
            print(
                i,
                util.marginal_index_repr(i,len(domain), join_string=' '),
                W.weights[i] / sum(W.weights),  # normalized weights of workload
                A._params[i] / sum(A._params)   # normalized weights of strategy
            )
    return None


if __name__ == '__main__':


    # get the workload
    W, domain = wkld_census_bds_firms()

    #
    # Define alternative strategies
    #

    # HDMM workload_analysis param
    A_marg = templates.Marginals(domain, seed=1004) # 1004 is a good seed!
    A_marg.optimize(W)

    # the workload, to be used as a strategy
    A_wkld = workload.Marginals.approximate(W)

    # full identity on all attributes in schema
    A_identity_full = templates.Marginals(domain)
    A_identity_full._params = np.zeros(2 ** len(domain))
    A_identity_full._params[-1] = 1.0

    # identity on attributes used in the workload
    used_attr = tuple(set().union(*firm_workload.keys()))    # get all attributes used in BDS marginal set
    A_identity = workload.Marginals.fromtuples(domain, {used_attr:1}, columns=variables)

    # Manual strategy proposed by David
    A_manual = manual_strategy()

    # Manual strategy, with weights optimized by HDMM
    temp = workload.Marginals.approximate(A_manual)
    A_manual_opt = templates.Marginals(domain, seed=1003)
    A_manual_opt._params = temp.weights
    A_manual_opt.optimize(W)
    A_manual_opt._params = np.clip(A_manual_opt._params, 0, float('inf'))

    print('Num queries:', W.shape[0])
    print('Sensitivity:', W.sensitivity())
    print('Marg', '\t\t', f'{error.rootmse(W, A_marg.strategy()):10.3f}')
    print('Ident_full', '\t', f'{error.rootmse(W, A_identity_full.strategy()):10.3f}')
    print('Identity', '\t', f'{error.rootmse(W, A_identity):10.3f}')
    print('Workload', '\t', f'{error.rootmse(W, A_wkld):10.3f}')
    print('Manual', '\t', f'{error.rootmse(W, A_manual):10.3f}')
    print('Manual Opt', '\t', f'{error.rootmse(W, A_manual_opt.strategy()):10.3f}')

    summarize_strategy(W, A_marg, domain)

    print('')

    print('')

    for m in W.matrices:
         print(m.base.key, m.base.shape[0], '\t', error.rootmse(m, A_marg.strategy()))

    for m in W.matrices:
         print(m.base.key, m.base.shape[0], '\t', error.rootmse(m, A_manual))

    for m in W.matrices:
         print(m.base.key, m.base.shape[0], '\t', error.rootmse(m, A_manual_opt.strategy()))

