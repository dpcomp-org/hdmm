
from hdmm import workload, templates, error
from experiments.marginals import util

'''
Below are marginals workloads motivated by private visualization using a dashboard
'''

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

def scenario1(domain):
    # Define marginals workload -- scenario 1
    # workload consisting of all one- and two-way marginals
    return workload.DimKMarginals(domain, [0,1,2])


def scenario2(domain, wkld_depth):
    # Define marginals workload -- scenario 2
    '''
        - step 1: H(A1)
        - step 2: H(A2), H(A1,A2)
        - step 3: H(A3), H(A1,A3), H(A2,A3)
        - step 4: H(A4), H(A1,A4), H(A2,A4), H(A3,A4)
    '''

    dim = len(domain)
    assert wkld_depth <= dim

    wkld_triangular_tuples = {}
    for i in range(wkld_depth):
        for j in range(i+1):
            new_tup = [0] * dim
            new_tup[i] = 1
            new_tup[j] = 1
            wkld_triangular_tuples[tuple(new_tup)] = 1

    return workload.Marginals.frombinary(domain, wkld_triangular_tuples)



if __name__ == '__main__':

    dim = 10
    domain = tuple([50]*dim)

    W_scen1 = scenario1(domain)
    # compute HDMM strategy using Marginals param
    A_scen1 = templates.Marginals(domain)
    A_scen1.optimize(W_scen1)


    W_scen2 = scenario2(domain, 5)
    # compute HDMM strategy using Marginals param
    A_scen2 = templates.Marginals(domain)
    A_scen2.optimize(W_scen2)

    # Here we analyze the error of scenario 2, which is what the user really wants
    # But we consider the strategy from scenario1 and compare with strategy derived from scenario2

    print('Num queries:', W_scen2.shape[0])
    print('Sensitivity:', W_scen2.sensitivity())
    print('Per query RMSE, A_scen1', '\t\t', f'{error.rootmse(W_scen1, A_scen1.strategy()):10.3f}')
    print('Per query RMSE, A_scen2', '\t\t', f'{error.rootmse(W_scen1, A_scen2.strategy()):10.3f}')
    print('')

    summarize_strategy(W_scen1, A_scen1, domain)

    print('')

    summarize_strategy(W_scen1, A_scen2, domain)

    print('')
    for m in W_scen1.matrices:
        print(
            m.base.key,
            util.marginal_index_repr(m.base.key, dim, " "),
            m.base.shape[0], '\t',
            error.rootmse(m, A_scen1.strategy()),
            error.rootmse(m, A_scen2.strategy()),
        )


