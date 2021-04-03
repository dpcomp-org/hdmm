# Copyright (C) 2019-2021, Tumult Labs Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from hdmm import templates
import autograd.numpy as np
from autograd import grad
from autograd.extend import primitive, defvjp
from IPython import embed

@primitive
def mm_error(A, WtW):
    """ Compute the expected error of A on W, under the following assumptions:
            1. A is a sensitivity 1 strategy
            2. A supports W
    """
    AtA1 = np.linalg.pinv(np.dot(A.T, A))
    return np.trace(np.dot(AtA1, WtW))

def grad_error(A, WtW):
    AtA1 = np.linalg.pinv(np.dot(A.T, A))
    X = -np.dot(AtA1, np.dot(WtW, AtA1))
    return 2*np.dot(A, X) 

defvjp(mm_error, lambda ans, A, WtW: lambda g: g*grad_error(A, WtW), argnums=[0])

class CustomTemplate(templates.TemplateStrategy):
    """
    The CustomTemplate strategy is specified by a function mapping parameters theta to 
    a strategy A(theta).  Gradients + Optimization are handled automatically as long
    as the passed function is compatible with autograd.  
    """
    def __init__(self, strategy, theta0, normalize=True, seed=None):
        """
        :param strategy: a function mapping parameters theta to strategies A(theta)
        :param theta0: the initial parameters
        :param normalize: flag to determine if A(theta) should be normalized
            Note: if normalize=False, A(theta) must always have bounded sensitivity for any theta
        """
        super(CustomTemplate, self).__init__(seed)

        self.strategy = strategy
        self.set_params(theta0)
        self.normalize = True

    def A(self):
        A = self.strategy(self.get_params())
        if self.normalize:
            wgt = np.sum(np.abs(A), axis=0)
            return A / wgt
        return A

    def _loss(self, params):
        WtW = self.workload.WtW
        A = self.strategy(params)
        if self.normalize:
            A = A / np.sum(np.abs(A), axis=0)
        return mm_error(A, WtW)
    
    def _grad(self, params):
        return grad(self._loss)(params)
 
    def _loss_and_grad(self):
        params = self.get_params()
        return self._loss(params), self._grad(params)

class RowWeighted(templates.TemplateStrategy):
    """
    The RowWeighted template strategy is characterized by a 'base matrix' and parameterized by
    a vector of weights corresponding to the rows in the matrix.  The strategy consists of 
    the queries from the base matrix, weighted according to the weight vector, plus any leftover 
    privacy budget is used to answer the identity queries.
    """
    def __init__(self, base, seed=None):
        super(RowWeighted, self).__init__(seed)

        self.base = base
        theta0 = self.prng.rand(base.shape[0]) * 3
        self._params = theta0

    @property
    def A(self):
        B = self.base * self._params[:,None] 
        p, n = B.shape
        scale = 1.0 + np.sum(B, axis=0)
        B = B / np.max(scale)
        diag = 1.0 - np.sum(B, axis=0)
        return np.vstack([np.diag(diag), B])

    def _set_workload(self, W):
        self.WtW = W.gram().dense_matrix()

    # objective function not as well behaved as PIdentity's objective function
    # because of the max function (convergence may be slower)
    def _loss(self, params):
        WtW = self.WtW
        B = self.base * params[:,None] 
        p, n = B.shape
        scale = 1.0 + np.sum(B, axis=0)
        B = B / np.max(scale)
        diag = 1.0 - np.sum(B, axis=0)
        # A = [diag; B] - always has sensitivity 1
        #TODO(ryan): use woodbury identity to make more efficient
        A = np.vstack([np.diag(diag), B])
        AtA1 = np.linalg.inv(np.dot(A.T, A))
        #D2 = 1.0 / diag**2
        #C = np.eye(p) + np.dot(B, B.T)
        #C1 = np.linalg.inv(C)
        #X = np.dot(B.T, np.dot(C1, B))
        # inverse calculated using woodbury identity
        #AtA1 = np.diag(D2) - X*D2*D2[:,None] 
        return np.trace(np.dot(WtW, AtA1))

    def _grad(self, params):
        return grad(self._loss)(params)

    def _loss_and_grad(self, params):
        return self._loss(params), self._grad(params)
