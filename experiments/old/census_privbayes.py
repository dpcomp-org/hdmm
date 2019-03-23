import workload
from census_workloads import *
import optimize
import approximation

geography = True
base = '/home/ryan/Desktop/privbayes_data/new/numpy'
X = np.load('%s/census.npy' % base)
axes = 3 if geography else (3,5)
x = X.sum(axis=axes).flatten()

dims = [[0],[1],[2],[4]]
if geography:
    dims.append([5])
sf1 = CensusSF1(geography=geography).project_and_merge(dims)
sf1a, sf1b = CensusSF1_split(geography)
sf1a = sf1a.project_and_merge(dims)
sf1b = sf1b.project_and_merge(dims)

marg = approximation.marginals_approx(sf1)
q = sf1.queries
sensitivity=sum(np.prod([np.abs(W.W).sum(axis=0).max() for W in K.workloads]) for K in sf1.workloads)

eye = [np.eye(n) for n in sf1.domain]
A = optimize.restart_union_kron(sf1, 50, [1,1,8,10,1])
Aa = optimize.restart_union_kron(sf1a, 50, [1,1,8,1,1])
Ab = optimize.restart_union_kron(sf1b, 50, [1,1,8,10,1])
theta, phi = optimize.restart_marginals(marg, 50)

for eps in [0.1, 1.0]:
    err1 = np.sqrt(sf1.expected_error(eye, eps) / q)
    err2 = np.sqrt(sf1.expected_error(A, eps) / q)

    err3 = np.zeros(25)
    for i in range(25):
        Y = np.load('%s/census%.1f-%d.npy' % (base, eps, i))
        y = Y.sum(axis=axes).flatten()
        err3[i] = sf1.squared_error(x - y)

    err3 = np.sqrt(np.mean(err3) / q)
    err4 = np.sqrt(2) * sensitivity / eps
    err5 = np.sqrt(marg.expected_error(theta, eps) / q)
  
    erra = sf1a.expected_error(Aa, eps/2.0)
    errb = sf1b.expected_error(Ab, eps/2.0)
    err6 = np.sqrt((erra+errb) / q)

    print '%.1f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (eps, err1, err2, err3, err4, err5, err6)
