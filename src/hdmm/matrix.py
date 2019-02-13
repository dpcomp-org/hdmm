from ektelo import util
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsqr
from functools import reduce
import math

class EkteloMatrix(LinearOperator):
    """
    An EkteloMatrix is a linear transformation that can compute matrix-vector products 
    """
    # must implement: _matmat, _transpose, matrix
    # can  implement: gram, sensitivity, sum, dense_matrix, sparse_matrix, __abs__

    def __init__(self, matrix):
        """ Instantiate an EkteloMatrix from an explicitly represented backing matrix
        
        :param matrix: a 2d numpy array or a scipy sparse matrix
        """
        self.matrix = matrix
        self.dtype = matrix.dtype
        self.shape = matrix.shape
    
    def asDict(self):
        d = util.class_to_dict(self, ignore_list=[])
        return d

    def _transpose(self):
        return EkteloMatrix(self.matrix.T)
    
    def _matmat(self, V):
        """
        Matrix multiplication of a m x n matrix Q
        
        :param V: a n x p numpy array
        :return Q*V: a m x p numpy aray
        """
        return self.matrix @ V

    def gram(self):
        """ 
        Compute the Gram matrix of the given matrix.
        For a matrix Q, the gram matrix is defined as Q^T Q
        """
        return self.T @ self # works for subclasses too
   
    def sensitivity(self):
        # note: this works because np.abs calls self.__abs__
        return np.max(np.abs(self).sum(axis=1))
 
    def sum(self, axis=None):
        # this implementation works for all subclasses too 
        # (as long as they define _matmat and _transpose)
        if axis == 0:
            return self.T.dot(np.ones(self.shape[0]))
        ans = self.dot(np.ones(self.shape[1]))  
        return ans if axis == 1 else np.sum(ans)

    def inv(self):
        return EkteloMatrix(np.linalg.inv(self.dense_matrix()))

    def pinv(self):
        return EkteloMatrix(np.linalg.pinv(self.dense_matrix()))

    def trace(self):
        return np.trace(self.dense_matrix())

    def _adjoint(self):
        return self._transpose()

    def __mul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)
        if type(other) == np.ndarray:
            return self.dot(other)
        if isinstance(other, EkteloMatrix):
            return Product(self, other)
            # note: this expects both matrix types to be compatible (e.g., sparse and sparse)
            # todo: make it work for different backing representations
            #return EkteloMatrix(self.matrix @ other.matrix)
        else:
            raise TypeError('incompatible type %s for multiplication with EkteloMatrix'%type(other))

    def __add__(self, other):
        if np.isscalar(other):
            other = Weighted(Ones(self.shape), other)
        return Sum([self, other])

    def __sub__(self, other):
        return self + -1*other
            
    def __rmul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)
        return NotImplemented

    def __getitem__(self, key):
        """ 
        return a given row from the matrix
    
        :param key: the index of the row to return
        :return: a 1xN EkteloMatrix
        """
        # row indexing, subclasses may provide more efficient implementation
        m = self.shape[0]
        v = np.zeros(m)
        v[key] = 1.0
        return EkteloMatrix(self.T.dot(v).reshape(1, m))
    
    def dense_matrix(self):
        """
        return the dense representation of this matrix, as a 2D numpy array
        """
        if sparse.issparse(self.matrix):
            return self.matrix.toarray()
        return self.matrix
    
    def sparse_matrix(self):
        """
        return the sparse representation of this matrix, as a scipy matrix
        """
        if sparse.issparse(self.matrix):
            return self.matrix
        return sparse.csr_matrix(self.matrix)
    
    @property
    def ndim(self):
        # todo: deprecate if possible
        return 2
    
    def __abs__(self):
        return EkteloMatrix(self.matrix.__abs__())

class Identity(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.shape = (n,n)
        self.dtype = dtype
   
    def _matmat(self, V):
        return V
 
    def _transpose(self):
        return self

    @property
    def matrix(self):
        return sparse.eye(self.n, dtype=self.dtype)
    
    def __mul__(self, other):
        assert other.shape[0] == self.n, 'dimension mismatch'
        return other

    def inv(self):
        return self

    def pinv(self):
        return self

    def trace(self):
        return self.n

    def __abs__(self):  
        return self

class Ones(EkteloMatrix):
    """ A m x n matrix of all ones """
    def __init__(self, m, n, dtype=np.float64):
        self.m = m
        self.n = n
        self.shape = (m,n)
        self.dtype = dtype
        
    def _matmat(self, V):
        ans = V.sum(axis=0, keepdims=True)
        return np.repeat(ans, self.m, axis=0)
    
    def _transpose(self):
        return Ones(self.n, self.m, self.dtype)
    
    def gram(self):
        return self.m * Ones(self.n, self.n, self.dtype)

    def pinv(self): 
        c = 1.0 / (self.m * self.n)
        return c * Ones(self.n, self.m, self.dtype)

    def trace(self):
        assert self.n == self.m, 'matrix is not square'
        return self.n
        
    @property
    def matrix(self):
        return np.ones(self.shape, dtype=self.dtype)
    
    def __abs__(self):
        return self
    
class Weighted(EkteloMatrix):
    """ Class for multiplication by a constant """
    def __init__(self, base, weight):
        if isinstance(base, Weighted):
            weight *= base.weight
            base = base.base
        self.base = base
        self.weight = weight
        self.shape = base.shape
        self.dtype = base.dtype
    
    def _matmat(self, V):
        return self.weight * self.base.dot(V)
    
    def _transpose(self):
        return Weighted(self.base.T, self.weight)
    
    def gram(self):
        return Weighted(self.base.gram(), self.weight**2)
        
    def pinv(self):
        return Weighted(self.base.pinv(), 1.0/self.weight)

    def inv(self):
        return Weighted(self.base.inv(), 1.0/self.weight)

    def trace(self):
        return self.weight * self.base.trace()
    
    def __abs__(self):
        return Weighted(self.base.__abs__(), np.abs(self.weight))
    
    @property
    def matrix(self):
        return self.weight * self.base.matrix

class Sum(EkteloMatrix):
    """ Class for the Sum of matrices """
    def __init__(self, matrices):
        # all must have same shape
        self.matrices = matrices
        self.shape = matrices[0].shape
        self.dtype = np.result_type(*[Q.dtype for Q in matrices])

    def _matmat(self, V):
        return sum(Q.dot(V) for Q in self.matrices)

    def _transpose(self):
        return Sum([Q.T for Q in self.matrices])
    
    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return sum(Q.sparse_matrix() for Q in self.matrices)
        return sum(Q.dense_matrix() for Q in self.matrices)

class VStack(EkteloMatrix):
    def __init__(self, matrices):
        m = sum(Q.shape[0] for Q in matrices)
        n = matrices[0].shape[1]
        assert all(Q.shape[1] == n for Q in matrices), 'dimension mismatch'
        self.shape = (m,n)
        self.matrices = matrices
        self.dtype = np.result_type(*[Q.dtype for Q in matrices])
    
    def _matmat(self, V):
        return np.vstack([Q.dot(V) for Q in self.matrices])

    def _transpose(self):
        return HStack([Q.T for Q in self.matrices])
    
    def __mul__(self, other):
        if isinstance(other,EkteloMatrix):
            return VStack([Q @ other for Q in self.matrices]) # should use others rmul though
        return EkteloMatrix.__mul__(self, other)

    def gram(self):
        return Sum([Q.gram() for Q in self.matrices])
    
    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return self.sparse_matrix()
        return self.dense_matrix()

    def dense_matrix(self):
        return np.vstack([Q.dense_matrix() for Q in self.matrices])

    def sparse_matrix(self):
        return sparse.vstack([Q.sparse_matrix() for Q in self.matrices])

    def __abs__(self):
        return VStack([Q.__abs__() for Q in self.matrices])

class HStack(EkteloMatrix):
    def __init__(self, matrices):
        # all matrices must have same number of rows
        cols = [Q.shape[1] for Q in matrices]
        m = matrices[0].shape[0]
        n = sum(cols)
        assert all(Q.shape[0] == m for Q in matrices), 'dimension mismatch'
        self.shape = (m,n)
        self.matrices = matrices
        self.dtype = np.result_type(*[Q.dtype for Q in matrices])
        self.split = np.cumsum(cols)[:-1]

    def _matmat(self, V):
        vs = np.split(V, self.split)
        return sum([Q.dot(z) for Q, z in zip(self.matrices, vs)])
    
    def _transpose(self):
        return VStack([Q.T for Q in self.matrices])
    
    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return self.sparse_matrix()
        return self.dense_matrix()
    
    def dense_matrix(self):
        return np.hstack([Q.dense_matrix() for Q in self.matrices])

    def sparse_matrix(self):
        return sparse.hstack([Q.sparse_matrix() for Q in self.matrices])
    
    def __mul__(self, other):
        if isinstance(other, VStack):
            # and shapes match...
            return Sum([A @ B for A,B in zip(self.matrices, other.matrices)])
        return EkteloMatrix.__mul__(self, other)

    def __abs__(self):
        return HStack([Q.__abs__() for Q in self.matrices])

class Kronecker(EkteloMatrix):
    def __init__(self, matrices):
        self.matrices = matrices
        self.shape = tuple(np.prod([Q.shape for Q in matrices], axis=0))
        self.dtype = np.result_type(*[Q.dtype for Q in matrices])

    def _matmat(self, V):
        X = V.T
        for Q in self.matrices[::-1]:
            m,n = Q.shape
            X = Q.dot(X.reshape(-1, n).T)
        return X.reshape(self.shape[0], -1)

    def _transpose(self):
        return Kronecker([Q.T for Q in self.matrices]) 
   
    def gram(self):
        return Kronecker([Q.gram() for Q in self.matrices])
    
    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return self.sparse_matrix()
        return self.dense_matrix()
 
    def dense_matrix(self):
        return reduce(np.kron, [Q.dense_matrix() for Q in self.matrices])

    def sparse_matrix(self):
        return reduce(sparse.kron, [Q.sparse_matrix() for Q in self.matrices])
  
    def sensitivity(self):
        return np.prod([Q.sensitivity() for Q in self.matrices])

    def inv(self):
        return Kronecker([Q.inv() for Q in self.matrices])

    def pinv(self):
        return Kronecker([Q.pinv() for Q in self.matrices])

    def trace(self):
        return np.prod([Q.trace() for Q in self.matrices])
    
    def __mul__(self, other):
        # perform the multiplication in the implicit representation if possible
        if isinstance(other, Kronecker):
            return Kronecker([A @ B for A,B in zip(self.matrices, other.matrices)])
        return EkteloMatrix.__mul__(self, other)
 
    def __abs__(self):
        return Kronecker([Q.__abs__() for Q in self.matrices]) 

class Product(EkteloMatrix):
    def __init__(self, A, B):
        assert A.shape[1] == B.shape[0]
        self._A = A
        self._B = B
        self.shape = (A.shape[0], B.shape[1])
        self.dtype = np.result_type(A.dtype, B.dtype)

    def _matmat(self, X):
        return self._A.dot(self._B.dot(X))

    def _transpose(self):
        return Product(self._B.T, self._A.T)

    @property
    def matrix(self):
        return self._A.matrix @ self._B.matrix

    def gram(self):
        return Product(self.T, self)

    def inv(self):
        return Product(self._B.inv(), self._A.inv())

def _any_sparse(matrices):
    return any(sparse.issparse(Q.matrix) for Q in matrices)