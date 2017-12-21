import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
import math
import itertools

def supports(A, W):
    """
    tests if the strategy supports the workload

    :param A: the strtegy
    :param W: the workload (note, W^T W may also be passed here)
    :return: True if A supports W
    """
    proj = np.eye(A.shape[1]) - np.linalg.pinv(A).dot(A)
    return np.abs(W.dot(proj)).max() < 1e-13
#    return np.linalg.norm(W.dot(proj)) < 1e-8

def squared_error(WtW, A, eps = np.sqrt(2)):
    """
    :param WtW: a workload in normal form
    :param A: a strategy
    """
    # if not supports(A, WtW): import IPython; IPython.embed()
    if not supports(A, WtW):
        print 'warning: A doesnt support W'
    X = np.linalg.lstsq(A.T.dot(A), WtW)[0]
    delta = np.abs(A).sum(axis=0).max()
    trace = np.trace(X)
    var = 2.0 / eps**2
    return var * delta**2 * trace

def squared_error_kron(WtWs, As, eps = np.sqrt(2)):
    """
    :param WtWs: a length d list of workloads in normal form
    :param As: a length d list of strategies
    """
    var = 2 / eps**2
    return var * np.prod([squared_error(WtW, A) for WtW, A in zip(WtWs, As)])
       
def squared_error_union_kron(WtWs, As, eps = np.sqrt(2)):
    """
    :param WtWs: a k x d size list of WtWs where WtWs
    :param As: a length d list of strategies
    """
    var = 2 / eps**2
    return var * np.sum([squared_error_kron(WtW, As) for WtW in WtWs]) 

# Note: A must support W (this is not checked)
def rootmse(WtW, A, queries, eps = np.sqrt(2)):
    if sparse.issparse(A):
        A = A.toarray()
    #X = np.linalg.solve(A.T.dot(A), WtW)
    X = np.linalg.lstsq(A.T.dot(A), WtW)[0]
    delta = np.abs(A).sum(axis=0).max()
    trace = np.trace(X)
    var = 2.0 / eps**2
    return np.sqrt(var * delta**2 * trace / queries)

def allrange(n):
    Q = np.zeros((n*(n+1)//2, n))
    r = 0
    for i in range(n):
        for j in range(i+1, n+1):
            Q[r, i:j] = 1.0
            r += 1
    return Q

def allrange_qtq(n, weighted = False):
    """
    Code to compute W^T W directly where W is all range queries
    Note: weighted weights each row by its query width
    Note: this works even when W is too large to fit into memory
    """
    QtQ = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            QtQ[i,j] = QtQ[j,i] = (i+1) * (n-j)
    if weighted:
        T = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                T[i,j] = (n+1) / 2.0 + 0.5*abs(i-j)
        QtQ *= T
    return QtQ

def prefix_qtq(n):
    Q = np.zeros((n,n))
    for i in range(n):
        Q[i,:i+1] = 1
    return Q.T.dot(Q)

def randomrange_qtq(n, m, replace=True):
    allrange = np.array([(i,j) for i in range(n) for j in range(i+1, n+1)])
    idx = np.random.choice(n*(n+1)//2, m, replace=replace)
    Q = np.zeros((m,n))
    for i, (low, high) in enumerate(allrange[idx]):
        Q[i, low:high] = 1.0
    return Q.T.dot(Q)

def wavelet(n):
    '''
    Returns a sparse (csr_matrix) wavelet matrix of size n = 2^k
    '''
    if n == 1:
        return sparse.identity(1, format='csr')
    m, r = divmod(n, 2)
    assert r == 0, 'n must be power of 2'
    H2 = wavelet(m)
    I2 = sparse.identity(m, format='csr')
    A = sparse.kron(H2, [1,1])
    B = sparse.kron(I2, [1,-1])
    return sparse.vstack([A,B])

def hb(n):
    b = find_best_branching(n)
    return hier(n, b)[1:]

def identity(n):
    return sparse.identity(n)

def greedyH(QtQ, branch = 2):
    n = QtQ.shape[1]
    err, inv, weights, queries = _GreedyHierByLv(QtQ, n, 0, branch, withRoot=False)

    # form matrix from queries and weights
    row_list = []
    for q, w in zip(queries, weights):
        if w > 0:
            row = np.zeros(n)
            row[q[0]:q[1]+1] = w
            row_list.append(row)
    return sparse.csr_matrix(np.vstack(row_list))

def hier(n, b):
    '''
    Builds a sparsely represented (csr_matrix) hierarchical matrix
    with n columns and a branching factor of b.  Works even when n 
    is not a power of b
    '''
    if n == 1:
        return sparse.csr_matrix([1.0])
    if n <= b:
        a = np.ones(n)
        b = sparse.identity(n)
        return sparse.vstack([a, b], format='csr')

    # n = mb + r where r < b
    # n = (m+1) r + m (b-r)
    # we need r hierarchical matrices with (m+1) cols 
    # and (b-r) hierarchical matrices with m cols
    m, r = divmod(n, b)
    hier0 = hier(m, b) # hierarchical matrix with m cols
    if r > 0:
        hier1 = hier(m+1, b) # hierarchical matrix with (m+1) cols

    # sparse.hstack doesn't work when matrices have 0 cols
    def hstack(left, hier, right):
        if left.shape[1] > 0 and right.shape[1] > 0:
            return sparse.hstack([left, hier, right])
        elif left.shape[1] > 0:
            return sparse.hstack([left, hier])
        else:
            return sparse.hstack([hier, right])

    res = [np.ones(n)]
    for i in range(r):
        rows = hier1.shape[0]
        start = (m+1)*i
        end = start + m+1
        left = sparse.csr_matrix((rows, start))
        right = sparse.csr_matrix((rows, n-end))
        res.append(hstack(left, hier1, right))
    for i in range(r, b):
        # (m+1) r + m (b-r) = (m+1) r + m (b-i) + m (i-r)
        rows = hier0.shape[0]
        start = (m+1)*r + m*(i-r)
        end = start + m
        left = sparse.csr_matrix((rows, start))
        right = sparse.csr_matrix((rows, n-end))
        res.append(hstack(left, hier0, right))
    return sparse.vstack(res, format='csr')


'''
Technique from Qardaji et al. PVLDB 2013.
'''
# N in this context is domain size
def find_best_branching(N):
    '''
    Try all branchings from 2 to N and pick one
    with minimum variance.
    '''
    min_v = float('inf')
    min_b = None
    for b in range(2,N+1):
        v = variance(N, b)
        if v < min_v:
            min_v = v
            min_b = b
    return min_b

def variance(N, b):
    '''Computes variance given domain of size N
    and branchng factor b.  Equation 3 from paper.'''
    h = math.ceil(math.log(N, b))
    return ( ((b - 1) * h**3) - ((2 * (b+1) * h**2) / 3))

def _GreedyHierByLv(fullQtQ, n, offset, branch, depth = 0, withRoot = False):
    """Compute the weight distribution of one node of the tree by minimzing
    error locally.

    fullQtQ - the same matrix as QtQ in the Run method
    n - the size of the submatrix that is corresponding
        to current node
    offset - the location of the submatrix in fullQtQ that
                is corresponding to current node
    depth - the depth of current node in the tree
    withRoot - whether the accurate root count is given

    Returns: error, inv, weights, queries
    error - the variance of query on current node with epsilon=1
    inv - for the query strategy (the actual weighted queries to be asked)
            matrix A, inv is the inverse matrix of A^TA
    weights - the weights of queries to be asked
    queries - the list of queries to be asked (all with weight 1)
    """
    if n == 1:
        return np.linalg.norm(fullQtQ[:, offset], 2)**2, \
                np.array([[1.0]]), \
                np.array([1.0]), [[offset, offset]]

    QtQ = fullQtQ[:, offset:offset+n]
    if (np.min(QtQ, axis=1) == np.max(QtQ, axis=1)).all():
        mat = np.zeros([n, n])
        mat.fill(1.0 / n**2)
        return np.linalg.norm(QtQ[:,0], 2)**2, \
                mat, np.array([1.0]), [[offset, offset+n-1]]

    if n <= branch:
        bound = zip(range(n), range(1,n+1))
    else:
        rem = n % branch
        step = (n-rem) / branch
        swi = (branch-rem) * step
        sep = range(0, swi, step) + range(swi, n, step+1) + [n]
        bound = zip(sep[:-1], sep[1:])

    serr, sinv, sdist, sq = zip(*[_GreedyHierByLv
                                (fullQtQ, c[1]-c[0], offset+c[0], branch,
                                depth = depth+1) for c in bound])
    invAuList = map(lambda c: c.sum(axis=0), sinv)
    invAu = np.hstack(invAuList)
    k = invAu.sum()
    m1 = sum(map(lambda rng, v:
                    np.linalg.norm(np.dot(QtQ[:, rng[0]:rng[1]], v), 2)**2,
                    bound, invAuList))
    m = np.linalg.norm(np.dot(QtQ, invAu), 2)**2
    sumerr = sum(serr)

    if withRoot:
        return sumerr, block_diag(*sinv), \
                np.hstack([[0], np.hstack(sdist)]), \
                [[offset, offset+n-1]] + list(itertools.chain(*sq))

    granu = 100
    decay = 1.0 / ( branch**(depth / 2.0))
    err1 = np.array(range(granu, 0, -1))**2
    err2 = np.array(range(granu))**2 * decay
    toterr = 1.0/err1 * (sumerr - ((m-m1)*decay+m1) * err2 / (err1+err2*k))

    err = toterr.min() * granu**2
    perc = 1 - np.argmin(toterr) / float(granu)
    inv = (1.0/perc)**2 * (block_diag(*sinv)
            - (1-perc)**2 / ( perc**2 + k * (1-perc)**2 )
            * np.dot(invAu.reshape([n, 1]), invAu.reshape([1, n])))
    dist = np.hstack([[1-perc], perc*np.hstack(sdist)])
    return err, inv, dist, \
            [[offset, offset+n-1]] + list(itertools.chain(*sq))
