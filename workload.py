import numpy as np
import collections
import itertools
import implicit
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

class Workload:
    def __init__(self):
        pass
   
    @property 
    def W(self):
        """
        return the workload matrix as a m x n numpy array
        """
        pass

    @property 
    def WtW(self):
        """
        return W^T W as a n x n numpy array
        Note: use this for large workloads (m >> n)
        """
        return self.W.T.dot(self.W)

    @property
    def domain(self):
        """ The domain size """
        return self.WtW.shape[0]
    
    @property
    def queries(self):
        """ The number of queries in the workload """
        return self.W.shape[0]

    def squared_error(self, noise):
        """ 
        Given a noise vector (x - xhat), compute the squared error on the workload
        """
        return noise.dot(self.WtW.dot(noise))
       
    def expected_error(self, strategy, eps=np.sqrt(2)):
        """
        Given a strategy and a privacy budget, compute the expected squared error
        """
        A = strategy 
        X = np.linalg.lstsq(A.T.dot(A), self.WtW)[0]
        delta = np.abs(A).sum(axis=0).max()
        trace = np.trace(X)
        var = 2.0 / eps**2
        return var * delta**2 * trace

    def average_error_ci(self, noises):
        """
        Given a list of noise vectors (x - xhat), compute a 95% confidence interval for the mean squared error.
        """
        samples = [self.squared_error(noise) for noise in noises]
        avg = np.mean(samples)
        pm = 1.96 * np.std(samples) / np.sqrt(len(samples))
        return (avg-pm, avg+pm)

    def rootmse(self, strategy, eps=np.sqrt(2)):
        """ compute a normalized version of expected squared error """ 
        return np.sqrt(self.expected_error(strategy, eps) / self.queries)

    def __rmul__(self, const):
        return Matrix(const * self.W)
#        return MatrixNormal(const**2 * self.WtW, self.queries)

    def __add__(self, other):
        return Concat([self, other])

class Permuted(Workload):
    def __init__(self, base, seed=0):
        """ the base workload to permute """
        self.idx = np.random.RandomState(seed).permutation(base.domain)
        self.base = base

    @property
    def W(self):
        return self.base.W[:,self.idx]

    @property
    def WtW(self):
        return self.base.WtW[self.idx,:][:,self.idx]
    
    @property
    def domain(self):
        return self.base.domain

    @property
    def queries(self):
        return self.base.queries

class Matrix(Workload):
    def __init__(self, matrix):
        self.matrix = matrix
   
    @property 
    def W(self):
        return self.matrix

class MatrixNormal(Workload):
    def __init__(self, normal, queries):
        self.queries = queries
        self.W = None
        self.WtW = normal 

class Prefix(Workload):
    def __init__(self, domain):
        """ 
        Prefix Queries for Empirical CDF
        :param domain: the domain size
        """
        self.domain = domain
        self.queries = domain

    @property 
    def W(self):
        n = self.domain
        Q = np.zeros((n,n))
        for i in range(n):
            Q[i,:i+1] = 1
        return Q

class AllRange(Workload):
    def __init__(self, domain):
        """
        All Range Queries
        :param domain: the domain size
        """
        self.domain = domain
        self.queries = domain * (domain + 1) // 2

    @property
    def W(self):
        m, n = self.queries, self.domain
        Q = np.zeros((m, n))
        r = 0
        for i in range(n):
            for j in range(i+1, n+1):
                Q[r, i:j] = 1.0
                r += 1
        return Q

    @property
    def WtW(self):
        n = self.domain
        QtQ = np.zeros((n,n))
        for i in range(n):
            for j in range(i, n):
                QtQ[i,j] = QtQ[j,i] = (i+1) * (n-j)
        return QtQ

class WidthKRange(Workload):
    def __init__(self, domain, widths):
        """ 
        Width K Range Queries for Sliding Average 
        :param domain: The domain size
        :param widths: the width of the queries (int or list of ints)
        """
        self.domain = domain
        if type(widths) is int:
            widths = [widths]
        self.widths = widths
        self.queries = sum(domain-k+1 for k in widths)

    @property
    def W(self):
        m, n = self.queries, self.domain
        W = np.zeros((m, n))
        row = 0
        for k in self.widths:
            for i in range(n-k+1):
                W[row+i, i:i+k] = 1.0
            row += n - k + 1
        return W

class Identity(Workload):
    def __init__(self, domain):
        self.domain = domain
        self.queries = domain

    @property
    def W(self):
        return np.eye(self.domain)

class Total(Workload):
    def __init__(self, domain):
        self.domain = domain
        self.queries = 1
    
    @property
    def W(self):
        return np.ones((1, self.domain))

class Concat(Workload):
    def __init__(self, workloads):
        doms = [W.domain for W in workloads]
        assert max(doms) == min(doms), 'domain sizes not compatible'
        self.workloads = []
        for w in workloads:
            if isinstance(w, Concat):
                self.workloads.extend(w.workloads)
            else:
                self.workloads.append(w)
        self.domain = workloads[0].domain
        self.queries = sum(w.queries for w in workloads)

    @property
    def W(self):
        return np.vstack([w.W for w in self.workloads])
       
    @property 
    def WtW(self):
        return sum(w.WtW for w in self.workloads)

    def squared_error(self, noise):
        return sum(W.squared_error(noise) for W in self.workloads)

    def expected_error(self, strategy, eps=np.sqrt(2)):
        return sum(W.expected_error(strategy, eps) for W in self.workloads) 

    def __rmul__(self, const):
        workloads = []  
        for W in self.workloads:
            workloads.append(const * W)
        return Concat(workloads)

    def project_and_merge(self, dims):
        assert isinstance(self.workloads[0], Kron)
        return Concat([W.project_and_merge(dims) for W in self.workloads])

    def unique_project(self):
        # Note: only works if subworkload have W defined
        assert isinstance(self.workloads[0], Kron)
        Ws = [[Wi.W for Wi in kron.workloads] for kron in self.workloads]
        proj = [Matrix(np.unique(np.vstack(S), axis=0)) for S in zip(*Ws)]
        return Kron(proj)

class Kron(Workload):
    def __init__(self, workloads):
        self.workloads = workloads
        self.domain = tuple([w.domain for w in workloads])
        self.queries = np.prod([w.queries for w in workloads])

    @property
    def W(self):
        """ Do not call this if domain is large """
        return reduce(np.kron, [w.W for w in self.workloads])

    @property
    def WtW(self):
        return reduce(np.kron, [w.WtW for w in self.workloads])

    def squared_error(self, noise):
        WtW = implicit.krons(*[w.WtW for w in self.workloads])
        return noise.dot(WtW.dot(noise))
    
    def expected_error(self, strategy, eps=np.sqrt(2)):
        var = 2.0 / eps**2
        errors = [W.expected_error(A) for W, A in zip(self.workloads, strategy)]
        return var * np.prod(errors)

    def __rmul__(self, const):
        workloads = list(self.workloads)
        workloads[0] = const * workloads[0]
        return Kron(workloads)

    def project_and_merge(self, dims):
        """ dims = [[0,1,2],[3],[4]] merges dimensions 0 1 and 2 and keeps dimensions 3 and 4"""
        workloads = []
        for d in dims:
            sub = [self.workloads[i] for i in d]
            W = reduce(np.kron, [W.W for W in sub])
            workloads.append(Matrix(W))
            #WtW = reduce(np.kron, [W.WtW for W in sub])
            #queries = np.prod([W.queries for W in sub])
            #workloads.append(MatrixNormal(WtW, queries))
        return Kron(workloads)

class Disjuncts(Concat):
    def __init__(self, workloads):
        WtWs = []
        for W in workloads:
            Q = 1 - W.W
            O = np.ones_like(Q)
            WtW1 = O.T.dot(O)
            WtW2 = O.T.dot(Q)
            WtW3 = Q.T.dot(O)
            WtW4 = Q.T.dot(Q)
            WtWs.append([WtW1, WtW2, WtW3, WtW4])
        WtWs = map(list, zip(*WtWs))
        WtWs[1][0] *= -1
        WtWs[2][0] *= -1
        
        workloads = [Kron([MatrixNormal(WtW, 1) for WtW in K]) for K in WtWs]
        Concat.__init__(self, workloads)

class Marginal(Workload):
    def __init__(self, domain):
        self.domain = domain
        self.queries = domain+1
   
    @property 
    def W(self):
        return np.vstack([np.ones(self.domain), np.eye(self.domain)])

class MarginalTable(Kron):
    def __init__(self, domain, binary):
        """ binary[i] = 1 if dimension i is included and binary[i] = 0 if it is marginalized out """
        self.binary = binary
        d = len(domain)
        subs = [None]*d
        for i in range(d):
            if binary[i] == 0: subs[i] = Total(domain[i])
            else: subs[i] = Identity(domain[i])
        Kron.__init__(self, subs) 

class Marginals(Concat):
    def __init__(self, domain, weights):
        """ 
        Marginals-like queries
        :param domain: the domain size
        :param weights: dict from marginal key to weight where the key is a d-length tuple corresponding to a single marginal

        Note: there are 2^d valid keys of the form (K1, ..., Kd) where 
                Ki = 0 if attribute i is marginalized out and
                Ki = 1 if attribute i is not marginalized out 
 
        Example: All One-Way Marginals on 3d domain of size 5 x 10 x 15
        weights = { (1, 0, 0) : 1.0, 
                    (0, 1, 0) : 1.0,
                    (0, 0, 1) : 1.0 }
        marginals = Marginals((5,10,15), weights)
        """
        self.domain = domain
        self.weights = collections.defaultdict(lambda: 0.0)
        self.weights.update(weights)
        subs = []
        for key, wgt in weights.items():
            if wgt > 0: subs.append(wgt * MarginalTable(domain, key))
        Concat.__init__(self, subs)

    def weight_vector(self):
        d = len(self.domain)
        vect = np.zeros(2**d)
        for i in range(2**d):
            key = tuple([int(bool(2**k & i)) for k in range(d)])
            vect[i] = self.weights[key]
        return vect 

    def expected_error(self, theta, eps=np.sqrt(2)):
        dom = self.domain
        weights = self.weight_vector()
        d = len(dom)
        mult = np.ones(2**d)
        for i in range(2**d):
            for k in range(d):
                if not (i & (2**k)):
                    mult[i] *= dom[k]
        A = np.arange(2**d)

        def Xmatrix(vect):
            values = np.zeros(3**d)
            rows = np.zeros(3**d, dtype=int)
            cols = np.zeros(3**d, dtype=int)
            start = 0
            for b in range(2**d):
                #uniq, rev = np.unique(a&B, return_inverse=True) # most of time being spent here
                mask = np.zeros(2**d, dtype=int)
                mask[A&b] = 1
                uniq = np.nonzero(mask)[0]
                step = uniq.size
                mask[uniq] = np.arange(step)
                rev = mask[A&b]
                values[start:start+step] = np.bincount(rev, vect*mult[A|b], step)
                if values[start+step-1] == 0:
                    values[start+step-1] = 1.0 # hack to make solve triangular work
                cols[start:start+step] = b
                rows[start:start+step] = uniq
                start += step
            X = sparse.csr_matrix((values, (rows, cols)), (2**d, 2**d))
            return X

        dphi = np.array([np.dot(weights**2, mult[A|b]) for b in range(2**d)])
        delta = np.sum(theta)**2
	theta2 = theta**2
	Y = Xmatrix(theta2)
	params = Y.dot(theta2)
	X = Xmatrix(params)
	phi = spsolve_triangular(X, theta2, lower=False)

        M = Xmatrix(Xmatrix(phi).dot(theta2))
        if not np.allclose(weights, M.dot(weights)):
            print 'Workload not supported by strategy'

	ans = np.prod(dom) * np.dot(phi, dphi)
        var = 2.0 / eps**2
        return var * delta * ans

class LowDimMarginals(Marginals):
    def __init__(self, domain, max_dim):
        weights = {}
        for key in itertools.product(*[[1,0]]*len(domain)):
            if sum(key) <= max_dim:
                weights[key] = 1.0
        Marginals.__init__(self, domain, weights) 

def DimKMarginals(domain, dims):
    if type(dims) is int:
        dims = [dims]
    weights = {}
    for key in itertools.product(*[[1,0]]*len(domain)):
        if sum(key) in dims:
            weights[key] = 1.0
    return Marginals(domain, weights)

def Width25(n):
    return WidthKRange(n, 25)

def Range2D(n):
    return Kron([AllRange(n), AllRange(n)])

def Prefix2D(n):
    return Kron([Prefix(n), Prefix(n)])

def RangeTotal2D(n):
    R = AllRange(n)
    T = Total(n)
    return Concat([Kron([R,T]), Kron([T,R])])

def RangeIdentity2D(n):
    R = AllRange(n)
    I = Identity(n)
    return Concat([Kron([R,I]), Kron([I,R])])

def PrefixIdentity2D(n):
    P = Prefix(n)
    I = Identity(n)
    return Concat([Kron([P,I]), Kron([I,P])])

if __name__ == '__main__':
    workload = Prefix(10)
    allrange = AllRange(10)
    krange = WidthKRange(10, [3,5])
    eye = Identity(10)
    tot = Total(10)
    concat = Concat([eye, tot, krange, workload])
    kron = Kron([eye, krange])
    print kron.W
    print concat.W  
    print concat.WtW
    print workload.WtW
    print allrange.WtW
    print krange.W
   
    weights = { (1, 0, 0) : 1.0, 
                (0, 1, 0) : 1.0,
                (0, 0, 1) : 1.0 }
    marginals = Marginals((2, 3, 4), weights)

    print marginals.W 

