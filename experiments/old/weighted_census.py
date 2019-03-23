from census_workloads import *
import optimize
from IPython import embed
import workload

restarts = 25
eps = 1.0
dims = [[0],[1],[2],[4]]
sf1 = CensusSF1().project_and_merge(dims)
sf1_pl94 = workload.Concat(sf1.workloads[:7] + sf1.workloads[-16:])
pl94 = CensusPL94().project_and_merge(dims)

I = [np.eye(n) for n in sf1.domain]

q1 = sf1.queries
q2 = pl94.queries
q3 = sf1_pl94.queries

print sf1.queries, pl94.queries, sf1.expected_error(I, eps), pl94.expected_error(I, eps)

p = [1,1,8,10]

A = optimize.restart_union_kron(pl94, restarts, [1,1,8,2])
#A = optimize.union_kron([[S.WtW for S in K.workloads] for K in pl94.workloads], init)['As']
print 'pl94', np.sqrt(pl94.expected_error(A, eps) / q2)
A = optimize.restart_union_kron(sf1_pl94, restarts, p)
print 'sf1-pl94', np.sqrt(sf1_pl94.expected_error(A, eps) / q3)


#A = optimize.restart_union_kron(sf1, restarts, p)
#print 'sf1', np.sqrt(sf1.expected_error(A, eps) / q1), np.sqrt(pl94.expected_error(A, eps) / q2)

def check():
    import matplotlib.pyplot as plt
    errs = []
    for i in range(100):
        init = [np.random.rand(pi, ni) for pi, ni in zip(p, sf1.domain)]
        ans = optimize.union_kron([[S.WtW for S in K.workloads] for K in pl94.workloads], init)
        errs.append(np.sqrt(ans['error'] / q2))
    print min(errs)
    plt.hist(errs)
    plt.show()

for c in [0,1,2,3,4,5,6,7,8,9,10,25,50,100,1000]:
    W = sf1 if c == 0 else sf1 + c*pl94
    A = optimize.restart_union_kron(W, restarts, p)
    err1 = sf1.expected_error(A, eps)
    err2 = pl94.expected_error(A, eps)
    err3 = err1 - err2
    print c, np.sqrt(err1 / q1), np.sqrt(err2 / q2), np.sqrt(err3 / (q1-q2))


