from src.hdmm import workload, templates, error
from experiments.marginals import wkld_census_bds
from experiments.marginals import wkld_dhc_household


def marginal_index_repr(i, dimensions, join_string=""):
    formatter = '{:0%db}' % dimensions
    return join_string.join(formatter.format(i))


def error_evaluation_marginal(W, A_list, columns=None):
    I = 2**len(W.domain)
    for i in range(I):
        if W.weights[i] > 0:
            print(
                f"{i :4d}", \
                "".join('{:09b}'.format(i)),  # bit representation of marginal, space-delim
                W.weights[i],
                *[A.strategy().weights[i] for A in A_list],
            )



if __name__ == '__main__':

    W, domain = wkld_dhc_household.dhc_household()

    M = templates.Marginals(domain)
    M.optimize(W)

    print(error.rootmse(W, M.strategy(), eps=1.0))

    error_evaluation_marginal(W, [M])