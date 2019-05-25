import numpy as np
import math

def discretize(strategy, digits=2):
    """
    Discretize a strategy matrix for use in geometric mechanism
    
    :param strategy: the strategy to discretize.  May be a 2D numpy array for an explicit strategy 
    or a list of 2D numpy arrays for a kronecker product strategies 
    :param digits: the number of digits to truncate to
    """
    if type(strategy) is np.ndarray:
        return np.round(strategy*10**digits).astype(int)
    elif type(strategy) is list:
        return [discretize(S, digits) for S in strategy]

def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def supported(W, A):
    '''
    :param W: workload
    :param A: strategy
    :return: True is W is supported by A
    '''
    AtA = A.gram()
    AtA1 = AtA.pinv()
    WtW = W.gram()
    X = WtW @ AtA1 @ AtA
    y = np.random.rand(WtW.shape[1])
    return np.allclose(WtW @ y,X @ y)