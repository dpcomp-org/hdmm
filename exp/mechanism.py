import numpy as np
import optimize
import implicit

class ParametricMM:

    def __init__(self, W, x, eps, seed=0):
        self.domain = W.domain
        self.W = W
        self.x = x
        self.eps = eps
        self.prng = np.random.RandomState(seed)

    def optimize(self):
        pass
