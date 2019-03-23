from hdmm import workload, templates, error, mechanism
from census_workloads import SF1_Persons
import numpy as np

def example1():
    """ Optimize AllRange workload using PIdentity template and report the expected error """
    print('Example 1')
    W = workload.AllRange(256)
    pid = templates.PIdentity(16, 256)
    res = pid.optimize(W)

    err = error.rootmse(W, pid.strategy())
    err2 = error.rootmse(W, workload.Identity(256))
    print(err, err2)

def example2():
    """ End-to-End algorithm for AllRange workload """
    print('Example 2')
    W = workload.AllRange(256)
   
    M = mechanism.HDMM(W, np.zeros(256), 1.0)
    M.optimize(restarts=5)
    xest = M.run()

    print(np.sum((xest - 0)**2))

def example3():
    """ Optimize Union-of-Kronecker product workload using kronecker parameterization
    and marginals parameterization """
    print('Example 3')
    sub_workloads1 = [workload.Prefix(64) for _ in range(4)]
    sub_workloads2 = [workload.AllRange(64) for _ in range(4)]
    W1 = workload.Kronecker(sub_workloads1)
    W2 = workload.Kronecker(sub_workloads2)
    W = workload.VStack([W1, W2])

    K = templates.KronPIdentity([4]*4, [64]*4)
    K.optimize(W)

    print(error.expected_error(W, K.strategy()))
    
    M = templates.Marginals([64]*4)
    M.optimize(W)

    print(error.expected_error(W, M.strategy()))

    identity = workload.Kronecker([workload.Identity(64) for _ in range(4)])
    print(error.expected_error(W, identity))

def example4():
    """ End-to-End algorithm on census workload """

    print('Example 4')
    sf1 = SF1_Persons()

    domain = [2,2,64,17,115]

    kron = templates.KronPIdentity([1,1,6,1,10], domain)

    res = kron.optimize(sf1)
    print(sf1.shape, len(sf1.matrices))

    x = np.zeros(sf1.shape[1])
    mech = mechanism.HDMM(sf1, x, 1.0)

    mech.optimize()
    #xest = mech.run()

    print('Done')

if __name__ == '__main__':
    example1()
    example2()
    example3()
    example4()

