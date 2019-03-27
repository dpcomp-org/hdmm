import numpy as np
from scipy import optimize
from scipy.sparse.linalg import lsmr

def nnls(A, y, l1_reg=0, l2_reg=0, maxiter = 15000):
    """
    Solves the NNLS problem min || Ax - y || s.t. x >= 0 using gradient-based optimization
    :param A: numpy matrix, scipy sparse matrix, or scipy Linear Operator
    :param y: numpy vector
    """

    def loss_and_grad(x):
        diff = A.dot(x) - y
        res = 0.5 * np.sum(diff ** 2)
        f = res + l1_reg*np.sum(x) + l2_reg*np.sum(x**2)
        grad = A.T.dot(diff) + l1_reg + l2_reg*x

        return f, grad

    xinit = np.zeros(A.shape[1])
    bnds = [(0,None)]*A.shape[1]
    xest,_,info = optimize.lbfgsb.fmin_l_bfgs_b(loss_and_grad,
                                                x0=xinit,
                                                pgtol=1e-4,
                                                bounds=bnds,
                                                maxiter=maxiter,
                                                m=1)
    xest[xest < 0] = 0.0
    return xest

def wnnls(W, A, y):
    xhat = lsmr(A, y)[0]
    yhat = W.dot(xhat)
    return nnls(W, yhat) 
