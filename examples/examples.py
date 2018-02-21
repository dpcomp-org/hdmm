import workload
import templates
import mechanism
import numpy as np
from experiments.census_workloads import CensusSF1

def example1():
    """ Optimize AllRange workload using PIdentity template and report the expected error """
    print 'Example 1'
    W = workload.AllRange(256)
    pid = templates.PIdentity(16, 256)
    res = pid.optimize(W)

    err = W.rootmse(pid.A)
    err2 = W.rootmse(np.eye(256))
    print err, err2

def example2():
    """ End-to-End algorithm for AllRange workload """
    print 'Example 2'
    W = workload.AllRange(256)
   
    M = mechanism.ParametricMM(W, np.zeros(256), 1.0)
    M.optimize(restarts=5)
    xest = M.run()

    print np.sum((xest - 0)**2)

def example3():
    """ Optimize Union-of-Kronecker product workload using kronecker parameterization
    and marginals parameterization """
    print 'Example 3'
    sub_workloads1 = [workload.Prefix(64) for _ in range(4)]
    sub_workloads2 = [workload.AllRange(64) for _ in range(4)]
    W1 = workload.Kron(sub_workloads1)
    W2 = workload.Kron(sub_workloads2)
    W = workload.Concat([W1, W2])

    K = templates.KronPIdentity([64]*4, [4]*4)
    K.optimize(W)

    sub_strategies = [S.A for S in K.strategies]

    # expects a list of sub-strategies to be passed in
    print W.expected_error(sub_strategies)
    
    M = templates.Marginals([64]*4)
    M.optimize(W)

    # error is calculated directly on the compact parameterization for marginals workload/strategy
    # M.workload is the marginals workload that is error-equivalent to W for marginals strategies
    # M.get_params() is the 2^4 optimized parameters that characterize the strategy
    print M.workload.expected_error(M.get_params()) 

    print W.expected_error([np.eye(64) for _ in range(4)])

def example4():
    """ End-to-End algorithm on census workload """

    print 'Example 4'
    sf1 = CensusSF1(geography=False)

    kron = templates.KronPIdentity(sf1.domain, [1,1,6,1,10])

    res = kron.optimize(sf1)
    print sf1.domain, sf1.queries, len(sf1.workloads)

    x = np.zeros(sf1.domain).flatten()
    mech = mechanism.ParametricMM(sf1, x, 1.0)

    mech.optimize()
    #xest = mech.run()

    print 'Done' 

if __name__ == '__main__':
    example1()
    example2()
    example3()
    example4()

