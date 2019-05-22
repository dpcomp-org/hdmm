from hdmm import workload, templates, error
import numpy as np
from experiments.marginals import util


'''
This is a marginals workload based on the Census "Business Dynamics Statistics" product

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

marginals = {
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



def census_bds():
    W = workload.Marginals.fromtuples(domain, marginals, columns=variables)
    return W, domain


def summarize_strategy(W, A, domain):
    # print table of marginals in binary, workload weights, and strategy weights
    for i in range(2**len(domain)):
        if W.weights[i] > 0 or A._params[i] > 0:
            print(
                i,
                util.marginal_index_repr(i,len(domain), join_string=' '),
                W.weights[i] / sum(W.weights),  # normalized weights of workload
                A._params[i] / sum(A._params)   # normalized weights of strategy
            )


if __name__ == '__main__':

    W, domain = census_bds()

    # Define alternative strategies
    A_marg = templates.Marginals(domain)
    A_marg.optimize(W)

    A_wkld = workload.Marginals.approximate(W)

    # full identity on all attributes
    A_identity_full = templates.Marginals(domain)
    A_identity_full._params = np.zeros(2 ** len(domain))
    A_identity_full._params[-1] = 1.0

    # identity on used attributes
    used_attr = tuple(set().union(*marginals.keys()))    # get all attributes used in BDS marginal set
    A_identity = workload.Marginals.fromtuples(domain, {used_attr:1}, columns=variables)

    print('Num queries:', W.shape[0])
    print('Sensitivity:', W.sensitivity())
    print('Marg', '\t\t', f'{error.rootmse(W, A_marg.strategy()):10.3f}')
    print('Ident_full', '\t', f'{error.rootmse(W, A_identity_full.strategy()):10.3f}')
    print('Identity', '\t', f'{error.rootmse(W, A_identity):10.3f}')
    print('Workload', '\t', f'{error.rootmse(W, A_wkld):10.3f}')

    print(*variables)
    print(*domain)

    summarize_strategy(W, A_marg, domain)

    print('')

    for m in W.matrices:
        print(m.base.key, m.base.shape[0], '\t', error.rootmse(m, A_marg.strategy()), error.rootmse(m, A_wkld))


