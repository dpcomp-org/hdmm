from census_workloads import *
import optimize
from IPython import embed
import time
import pickle
import approximation
import implicit

eps = 0.1
X, Xs = pickle.load(open('/home/ryan/Desktop/privbayes_data/census-%.1f.pkl' % eps, 'rb'))
diffs = [(X - Y).sum(axis=3).flatten() for Y in Xs]

dims = [[0],[1],[2],[4]]

sf1 = CensusSF1().project_and_merge(dims)
approx = CensusSF1Approx().project_and_merge(dims)
proj = CensusSF1Projected().project_and_merge(dims)
proj2 = CensusSF1().unique_project().project_and_merge(dims)
kifer = CensusKifer().project_and_merge(dims)
marginals = approximation.marginals_approx(sf1)

eye = [np.eye(n) for n in sf1.domain]
t0 = time.time()
A_sf1 = optimize.restart_union_kron(sf1, 50, [1,1,8,10])
t1 = time.time()
A_approx = optimize.restart_union_kron(approx, 50, [1,1,8,10])
t2 = time.time()
A_proj = optimize.restart_kron(proj, 50, [1,1,8,10])
t3 = time.time()
A_kifer = optimize.restart_kron(kifer, 50, [1,1,1,10])
t4 = time.time()
A_proj2 = optimize.restart_kron(proj2, 50, [1,1,8,10])
A_marg, A1_marg = optimize.restart_marginals(marginals, 50)

err1 = sf1.expected_error(eye, eps)
pmm = sf1.expected_error(A_sf1, eps)
err3 = sf1.expected_error(A_approx, eps)
err4 = sf1.expected_error(A_proj, eps)
err5 = sf1.expected_error(A_kifer, eps)
err6 = sf1.squared_error(X.sum(axis=3).flatten() - X.sum() / np.prod(X.shape))
low7,high7 = sf1.average_error_ci(diffs)
err8 = sf1.expected_error(A_proj2, eps)
err9 = marginals.expected_error(A_marg, eps) 
# Note: it's okay to calculate error on marginals, it's the same as error on sf1
# See documentation from marginals_approx

pmm = pmm
serr7 = (np.sqrt(low7/pmm), np.sqrt(high7/pmm))

print (t1-t0), (t2-t1), (t3-t2), (t4-t3)
print np.sqrt(err1/pmm), np.sqrt(pmm/pmm), np.sqrt(err3/pmm), np.sqrt(err4/pmm), np.sqrt(err5/pmm), np.sqrt(err6/pmm), serr7, np.sqrt(err8/pmm), np.sqrt(err9/pmm)

q = sf1.queries
print 'Per Query Error', np.sqrt(err1 / q), np.sqrt(pmm / q), np.sqrt(err6 / q), np.sqrt(low7 / q), np.sqrt(high7 / q)

#err1 = kifer.expected_error(eye)
#pmm = kifer.expected_error(A_kifer)
#print np.sqrt(err1/pmm), np.sqrt(pmm/pmm)

###############

sf2 = CensusSF1Big().project_and_merge(dims)

B0 = [np.random.rand(p, n) for p, n in zip([1,1,8,10], sf1.domain)]

ans1 = optimize.union_kron([[W.WtW for W in K.workloads] for K in sf1.workloads], B0)
ans2 = optimize.union_kron([[W.WtW for W in K.workloads] for K in sf2.workloads], B0)

print ans1['time'], ans2['time']

embed()

