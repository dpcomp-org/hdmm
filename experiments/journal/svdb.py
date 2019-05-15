from functools import reduce
from scipy import sparse
import numpy as np

def svdb_marg(W):
    G = W.gram()
    d = len(G.domain)
    # create Y matrix
    Y = sparse.dok_matrix((2**d, 2**d))
    for a in range(2**d):
        for b in range(2**d):
            if b&a == a:
                Y[a,b] = G._mult[b]
    Y = Y.tocsr()
    
    # compute unique eigenvalues
    e = Y.dot(G.weights)
    # now compute multiplicities 
    mult = reduce(np.kron, [[1,n-1] for n in G.domain])
    
    ans = np.dot(mult, np.sqrt(e))**2 / mult.sum()
    
    return ans
