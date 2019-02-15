import numpy as np
from hdmm import matrix, workload, error
import unittest

class TestError(unittest.TestCase):
    """ make sure error evaluation doesn't change based on matrix represenation """
    
    def setUp(self):
        self.prng = np.random.RandomState(10)


    def test_per_query(self):
        W = workload.Prefix2D(4)
        A1 = matrix.EkteloMatrix(self.prng.rand(5,4))
        I = matrix.Identity(4)
        A = matrix.Kronecker([A1, A1])
        total = error.expected_error(W, A)
        pq = error.per_query_error(W, A)
        print(total, pq.sum())
        self.assertTrue(abs(total-pq.sum()) <= 1e-5)

    def test_representations(self):
        W = workload.Prefix2D(4)
        A = matrix.EkteloMatrix(self.prng.rand(5,4))
        A = matrix.Kronecker([A, A])

        W2 = matrix.EkteloMatrix(W.dense_matrix())
        A2 = matrix.EkteloMatrix(A.dense_matrix())

        total = error.expected_error(W, A)
        total2 = error.expected_error(W, A2)
        total3 = error.expected_error(W2, A)
        total4 = error.expected_error(W2, A2)

        self.assertTrue(abs(total-total2) <= 1e-5)
        self.assertTrue(abs(total-total3) <= 1e-5)
        self.assertTrue(abs(total-total4) <= 1e-5)

        W = workload.DimKMarginals((4,4), 1)
       
        total = error.expected_error(W, A)
        total2 = error.expected_error(W, A2)
        self.assertTrue(abs(total-total2) <= 1e-5) 



if __name__ == '__main__':
    unittest.main()
