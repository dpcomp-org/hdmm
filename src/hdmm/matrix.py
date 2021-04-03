# This file is part of the Ektelo framework 
# For licensing terms please see: https://github.com/ektelo/ektelo
#
# Copyright 2019-2021, Tumult Labs Inc.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
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
        return np.max(np.abs(self).sum(axis=0))
 
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
        return self.diag().sum()

    def diag(self):
        return np.diag(self.dense_matrix())

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
        m, n = self.shape
        v = np.zeros(m)
        v[key] = 1.0
        return EkteloMatrix(self.T.dot(v).reshape(1, n))
    
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
    
    def __sqr__(self):
        if sparse.issparse(self.matrix):
            return EkteloMatrix(self.matrix.power(2))
        return EkteloMatrix(self.matrix**2)
   
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

    def __sqr__(self):
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

    def __sqr__(self):
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
    
    def __mul__(self, other):
        if isinstance(other,EkteloMatrix):
            return Weighted(self.base @ other, self.weight)
        return EkteloMatrix.__mul__(self, other)

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
        
    def __sqr__(self):
        return Weighted(self.base.__sqr__(), self.weight**2)

    def sensitivity(self):
        return self.weight * self.base.sensitivity()
    
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

    def __mul__(self, other):
        if isinstance(other,EkteloMatrix):
            return Sum([Q @ other for Q in self.matrices]) # should use others rmul though
        return EkteloMatrix.__mul__(self, other)

    def diag(self):
        return sum(Q.diag() for Q in self.matrices)

    def trace(self):
        return sum(Q.trace() for Q in self.matrices)

    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return sum(Q.sparse_matrix() for Q in self.matrices)
        return sum(Q.dense_matrix() for Q in self.matrices)

class BlockDiag(EkteloMatrix):
    def __init__(self, matrices):
        self.matrices = matrices
        rows = sum(Q.shape[0] for Q in matrices)
        cols = sum(Q.shape[1] for Q in matrices)
        self.shape = (rows, cols)
        self.dtype = np.result_type(*[Q.dtype for Q in matrices])

    # TODO: implement _matmat

    def diag(self):
        return np.concatenate([Q.diag() for Q in self.matrices])

    @property
    def matrix(self):
        return sparse.block_diag([Q.matrix for Q in self.matrices], format='csr')

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

    def __sqr__(self):
        return VStack([Q.__sqr__() for Q in self.matrices])


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
        ans = np.zeros((self.shape[0], V.shape[1]), dtype=self.dtype)
        for Q,z in zip(self.matrices, vs):
            ans += Q.dot(z)
        return ans
    
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

    def __rmul__(self, other):
        if isinstance(other, EkteloMatrix):
            return HStack([other @ Q for Q in self.matrices])
        return EkteloMatrix.__mul__(self, other)

    def __abs__(self):
        return HStack([Q.__abs__() for Q in self.matrices])

    def __sqr__(self):
        return HStack([Q.__sqr__() for Q in self.matrices])


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

    def diag(self):
        return reduce(np.kron, [Q.diag() for Q in self.matrices])

    def trace(self):
        return np.prod([Q.trace() for Q in self.matrices])
    
    def __mul__(self, other):
        # perform the multiplication in the implicit representation if possible
        if isinstance(other, Kronecker):
            return Kronecker([A @ B for A,B in zip(self.matrices, other.matrices)])
        elif isinstance(other, HStack):
            return other.__rmul__(self)
        return EkteloMatrix.__mul__(self, other)
 
    def __abs__(self):
        return Kronecker([Q.__abs__() for Q in self.matrices]) 

    def __sqr__(self):
        return Kronecker([Q.__sqr__() for Q in self.matrices]) 

class Haar(EkteloMatrix):
    """
    The Haar wavelet is a square matrix of size n x n where n is a power of 2
    """
    def __init__(self, n, dtype = np.float64):
        self.n = n
        self.k = int(math.log(n, 2))
        assert 2**self.k == n, 'n must be a power of 2'
        self.shape = (n,n)
        self.dtype = dtype

    def _matmat(self, X):
        y = X.copy()
        n = self.n
        for _ in range(self.k):
            y[:n] = np.vstack([y[:n][0::2] + y[:n][1::2], y[:n][0::2] - y[:n][1::2]])
            n = n // 2
        return y

    def _rmatvec(self, y):
        # can implement this instead of _transpose
        x = y.copy()
        m = 1
        for _ in range(self.k):
            n = 2*m
            # be careful here, don't separate into two calls
            x[0:n:2], x[1:n:2] = x[:m] + x[m:n], x[:m] - x[m:n]
            m *= 2
        return x

    def _transpose(self):
        return LinearOperator._adjoint(self)

    def sensitivity(self):
        return self.k + 1.0

    @property
    def matrix(self):
        H = sparse.eye(1, format='csr')
        for m in [2**c for c in range(self.k)]:
            I = sparse.eye(m, format='csr')
            A = sparse.kron(H, [1,1], format='csr')
            B = sparse.kron(I, [1,-1], format='csr')
            H = sparse.vstack([A,B], format='csr')
        return H

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
