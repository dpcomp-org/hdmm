import numpy as np
from workload import *
import templates
from experiments.census_workloads import __race1, __race2, __notHispanic, __isHispanic, __adult, \
    __age1, __age2, __age3, __white

P1 = Kron([Total(2), Total(2), Total(64), Total(17), Total(115)])
P3a = Kron([Total(2), Total(2), __race1(), Total(17), Total(115)])
P3b = P1
P4a = Kron([Total(2), Identity(2), Total(64), Total(17), Total(115)])
P4b = P1
P5a = Kron([Total(2), Identity(2), __race1(), Total(17), Total(115)])
P5b = Kron([Total(2), IdentityTotal(2), Total(64), Total(17), Total(115)])
P8a = Kron([Total(2), Total(2), __race2(), Total(17), Total(115)])
P8b = P1
P9a = Kron([Total(2), Identity(2), Total(64), Total(17), Total(115)])
P9b = Kron([Total(2), __notHispanic(), __race2(), Total(17), Total(115)])
P9c = P1
P10a = Kron([Total(2), Total(2), __race2(), Total(17), __adult()])
P10b = Kron([Total(2), Total(2), Total(64), Total(17), __adult()])
P11a = Kron([Total(2), Identity(2), Total(64), Total(17), __adult()])
P11b = Kron([Total(2), __notHispanic(), __race2(), Total(17), __adult()])
P11c = P10b
P12a = Kron([Identity(2), Total(2), Total(64), Total(17), __age1()])
P12b = Kron([IdentityTotal(2), Total(2), Total(64), Total(17), Total(115)])
P12_a = Kron([Identity(2), Total(2), __race1(), Total(17), __age1()])
P12_b = Kron([IdentityTotal(2), Total(2), __race1(), Total(17), Total(115)])
P12_c = Kron([Identity(2), __isHispanic(), Total(64), Total(17), __age1()])
P12_d = Kron([IdentityTotal(2), __isHispanic(), Total(64), Total(17), Total(115)])
P12_e = Kron([Identity(2), __notHispanic(), __white(), Total(17), __age1()])
P12_f = Kron([IdentityTotal(2), __notHispanic(), __white(), Total(17), Total(115)])
PCT12a = Kron([Identity(2), Total(2), Total(64), Total(17), __age3()])
PCT12b = P12b
PCT12_a = Kron([Identity(2), Total(2), __race1(), Total(17), __age3()])
PCT12_b = Kron([IdentityTotal(2), Total(2), __race1(), Total(17), Total(115)])
PCT12_c = Kron([Identity(2), __isHispanic(), Total(64), Total(17), __age3()])
PCT12_d = Kron([IdentityTotal(2), __isHispanic(), Total(64), Total(17), Total(115)])
PCT12_e = Kron([Identity(2), __notHispanic(), __race1(), Total(17), __age3()])
PCT12_f = Kron([IdentityTotal(2), __notHispanic(), __race1(), Total(17), Total(115)])

# reduced and full version of SF1
# the full version has the additional queries at the end
#
def CensusSF1(reduced=False):
    workloads = [P1,P3a,P3b,P4a,P4b,P5a,P5b,P12a,P12b,P12_a,P12_b,P12_c,P12_d,P12_e,P12_f,PCT12a,PCT12b,PCT12_a,PCT12_b,PCT12_c,PCT12_d,PCT12_e,PCT12_f]
    if not reduced:
        workloads.extend([P8a,P8b,P9a,P9b,P9c,P10a,P10b,P11a,P11b,P11c])
    return Concat(workloads)




if __name__ == '__main__':

    sf1 = CensusSF1(reduced=False)
    print(sf1.domain)
    print('Full, number of queries:', sf1.queries)

    ps = [1 ,1 ,8 ,1 ,10]
    template = templates.KronPIdentity(sf1.domain, ps)
    template.optimize(sf1)

    strategy = [sub.A for sub in template.strategies]

    print('Total RMSE Full', sf1.rootmse(strategy))

    # RMSE per query
    errors = np.sqrt(sf1.per_query_error(strategy))


    sf1_reduced = CensusSF1(reduced=True)
    print('Reduced, number of queries:', sf1_reduced.queries)

    template.optimize(sf1_reduced)

    strategy = [sub.A for sub in template.strategies]

    print('Total RMSE Reduced', sf1_reduced.rootmse(strategy))

    # RMSE per query
    errors_reduced = np.sqrt(sf1_reduced.per_query_error(strategy))

    # only compare initial prefix of the full workload, to match reduced workload
    errors_full_truncated = errors[:sf1_reduced.queries]

    # for i, e in enumerate(zip(errors_reduced, errors_full_truncated)):
    #     print(i, e[0], e[1], e[0]/e[1])


    ratio_reduced_by_full = errors_reduced / errors_full_truncated

    hist = np.histogram(ratio_reduced_by_full, bins=10)

    print('Error ratio histogram')
    for b, c in zip(hist[1], hist[0]):
        print('{}  {}'.format(b,c))





#    print(sf1.expected_error(strategy)) # total variance of HDMM