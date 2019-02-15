import numpy as np
from hdmm.matrix import EkteloMatrix

def convert_implicit(A):
    if isinstance(A, EkteloMatrix):
        return A
    return EkteloMatrix(A)

def expected_error(W, A, eps=np.sqrt(2), delta=0):
    """
    Given a strategy and a privacy budget, compute the expected squared error
    """
    assert delta == 0, 'delta must be 0'
    W, A = convert_implicit(W), convert_implicit(A)
    AtA1 = A.gram().pinv()
    WtW = W.gram()
    X = AtA1 @ WtW
    delta = A.sensitivity()
    trace = X.trace()
    var = 2.0 / eps**2
    return var * delta**2 * trace

def rootmse(W, A, eps=np.sqrt(2), delta=0):
    """ compute a normalized version of expected squared error """
    return np.sqrt(expected_error(W, A, eps, delta) / W.shape[0])

def squared_error(W, noise):
    """ 
    Given a noise vector (x - xhat), compute the squared error on the workload
    """
    W = convert_implicit(W)
    WtW = W.gram()
    return noisy.dot(WtW.dot(noise))

def average_error_ci(W, noises):
    """
    Given a list of noise vectors (x - xhat), compute a 95% confidence interval for the mean squared error.
    """
    samples = [squared_error(W, noise) for noise in noises]
    avg = np.mean(samples)
    pm = 1.96 * np.std(samples) / np.sqrt(len(samples))
    return (avg-pm, avg+pm)

def per_query_error(W, A, eps=np.sqrt(2), delta=0):
    W, A = convert_implicit(W), convert_implicit(A)
    delta = A.sensitivity()
    var = 2.0/eps**2
    X = W @ A.pinv()
    err = X.sqr().sum(axis=1)
    return var * delta * err
