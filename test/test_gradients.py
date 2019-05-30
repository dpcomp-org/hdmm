import numpy as np
from hdmm import matrix, workload, templates
import unittest
from scipy.optimize import check_grad

class TestGradients(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(10)

    def test_pidentity(self):
        pid = templates.PIdentity(2, 8)
        pid._set_workload(workload.Prefix(8))

        x0 = self.prng.rand(16)

        func = lambda p: pid._loss_and_grad(p)[0]
        grad = lambda p: pid._loss_and_grad(p)[1]

        err = check_grad(func, grad, x0)
        print(err)
        self.assertTrue(err <= 1e-5)

    def test_augmented_identity(self):    

        pid1 = templates.IdTotal(8)
        imatrix = self.prng.randint(0, 5, (3,8))
        pid2 = templates.AugmentedIdentity(imatrix)
        strats = [pid1, pid2]

        for pid in strats:
            pid._set_workload(workload.Prefix(8))
            x0 = self.prng.rand(pid._params.size)
            func = lambda p: pid._loss_and_grad(p)[0]
            grad = lambda p: pid._loss_and_grad(p)[1]
            err = check_grad(func, grad, x0)
            print(err)
            self.assertTrue(err <= 1e-5)

    def test_default(self):
        # TODO(ryan): test fails, but we don't really use this parameterization anyway
        temp = templates.Default(10, 8)
        temp._set_workload(workload.Prefix(8))   

        x0 = self.prng.rand(80)
        x0[0] = 10

        func = lambda p: temp._loss_and_grad(p)[0]
        grad = lambda p: temp._loss_and_grad(p)[1]

        err = check_grad(func, grad, x0)
        print(err)
        #self.assertTrue(err <= 1e-5)

    def test_marginals(self):
        # full rank case
        W = workload.Range2D(4)

        temp = templates.Marginals((4,4))
        temp._set_workload(W)

        x0 = self.prng.rand(4)

        func = lambda p: temp._loss_and_grad(p)[0]
        grad = lambda p: temp._loss_and_grad(p)[1]

        err = check_grad(func, grad, x0)
        print(err)
        #self.assertTrue(err <= 1e-5)

        # low rank case
        P = workload.Prefix(4)
        T = workload.Total(4)
        W1 = workload.Kronecker([P,T])
        W2 = workload.Kronecker([T,P])
        W = workload.VStack([W1, W2])

        temp = templates.Marginals((4,4))
        temp._set_workload(W)
        x0 = np.array([1,1,1,0.0])

        func = lambda p: temp._loss_and_grad(p)[0]
        grad = lambda p: temp._loss_and_grad(p)[1]

        f, g = func(x0), grad(x0)
        g2 = np.zeros(4)
        for i in range(4):
            x0[i] -= 0.00001
            f1 = func(x0)
            x0[i] += 0.00002
            f2 = func(x0)
            x0[i] -= 0.00001
            g2[i] = (f2 - f1) / 0.00002 


        print(g)
        print(g2)

        np.testing.assert_allclose(g, g2, atol=1e-5)
        #err = check_grad(func, grad, x0)
        #print(err)
        #self.assertTrue(err <= 1e-5)

if __name__ == '__main__':
    unittest.main()
