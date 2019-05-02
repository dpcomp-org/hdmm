from hdmm import matrix, workload
import itertools
from census_workloads import SF1_Persons

def get_workload(dataset, workload):
    if dataset == 'adult':
        return adult_big()[workload-1]
    if dataset == 'loans':
        return loans_big()[workload-1]
    if dataset == 'census':
        return census()[workload-1]

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def DimKKrons(workloads, k=1):
    blocks = workloads
    base = [workload.Total(W.shape[1]) for W in blocks]
    d = len(blocks)
    
    concat = []
    for attr in itertools.combinations(range(d), k):
        subs = [blocks[i] if i in attr else base[i] for i in range(d)]
        W = workload.Kronecker(subs)
        concat.append(W)

    return workload.VStack(concat) 

def SmallKrons(blocks, size=5000):
    base = [workload.Total(W.shape[1]) for W in blocks]
    concat = []
    for attr in powerset(range(len(blocks))):
        subs = [blocks[i] if i in attr else base[i] for i in range(d)]
        W = workload.Kronecker(subs)
        if W.shape[1] <= size:
            concat.append(W)
    return workload.VStack(concat)

def adult_big():
    R = workload.AllRange
    P = workload.Prefix
    M = workload.IdentityTotal
    I = workload.Identity
    T = workload.Total

    ns = (100,100,100,99,85,42,16,15,9,7,6,5,2,2)
    W1 = workload.DimKMarginals(ns, [0,1,2,3])
    W2 = DimKKrons([P(100), P(100), P(100), P(99), P(85), I(42), I(16), I(15), I(9), I(7), I(6), I(5), I(2), I(2)], 2)
    return W1, W2

def loans_big():
    P = workload.Prefix
    I = workload.Identity
    # loan_amt, int_rate, annual_inc, installment, term, grade, sub_grade, home_ownership, state, settlement status, loan_status, purpose
    W1 = SmallKrons([I(101),I(101),I(101),I(101),I(3),I(8),I(36),I(6),I(51),I(4),I(5),I(15)],5000)
    W2 = SmallKrons([P(101),P(101),P(101),P(101),I(3),I(8),I(36),I(6),I(51),I(4),I(5),I(15)],5000)
    return W1, W2

def cps():
    R = workload.AllRange
    P = workload.Prefix
    M = workload.IdentityTotal
    I = workload.Identity
    T = workload.Total

    W1 = workload.Kronecker([R(50), R(100), M(7), M(4), M(2)])
    W2 = DimKKrons([R(50), R(100), I(7), I(4), I(2)], 2)

    return W1, W2

def adult():
    R = workload.AllRange
    P = workload.Prefix
    M = workload.IdentityTotal
    I = workload.Identity
    T = workload.Total

    W1 = workload.Kronecker([M(75), M(16), M(5), M(2), M(20)])
    W2 = DimKKrons([I(75), I(16), I(5), I(2), I(20)], 2)
    #W2 = workload.DimKMarginals((75, 16, 5, 2, 20), 2)

    return W1, W2

def census():
    W = SF1_Persons()
    workloads = []
    for K in W.matrices:
        w = K.matrices
        Wi = workload.Kronecker([w[0], w[1], w[2], w[4]])
        workloads.append(Wi)
    W1 = workload.VStack(workloads)

    M = workload.IdentityTotal
    workloads = []
    for K in W.matrices:
        w = K.matrices
        Wi = workload.Kronecker([w[0], w[1], w[2], w[4], M(51)])
        workloads.append(Wi)
    W2 = workload.VStack(workloads)
    
    return W1, W2 
