from ektelo import workload
from hdmm.templates import DefaultKron, Marginals, DefaultUnionKron
from mbi import FactoredInference, Domain, Dataset
import numpy as np
from IPython import embed

# set up domain and workload
attributes = ['A','B','C'] #  should be the names of the columns, for now just using 0 and 1
sizes = [32, 32, 32]
dom = Domain(attributes, sizes)
#W = workload.Prefix2D(32)
W = workload.DimKMarginals(sizes, 1)
data = Dataset.synthetic(dom, 1000)

# optimize strategy using HDMM
#template = DefaultKron(sizes)
#template = Marginals(sizes)
template = DefaultUnionKron(sizes, 3)
template.optimize(W)
A = template.strategy()

def take_measurements(A, data):
    """ Efficiently take measurements from HDMM strategy and convert to a PGM-compatable form """
    A = workload.union_kron_canonical(A)
    measurements = []
    for Ai in A.matrices:
        w = Ai.weight
        proj = [ attributes[i] for i, B in enumerate(Ai.base.matrices) if type(B) != workload.Ones ]
        print(proj)
        matrix = workload.Kronecker([ B for B in Ai.base.matrices if type(B) != workload.Ones ])
        matrix = w * matrix.sparse_matrix()
        x = data.project(proj).datavector() # does Relation have this functionality?
        y = matrix.dot(x) + np.random.laplace(loc=0, scale=1, size=matrix.shape[0]) 
        measurements.append( (matrix, y, 1.0, proj) )
    return measurements

measurements = take_measurements(A, data)

engine = FactoredInference(dom)
model = engine.estimate(measurements)

df = model.synthetic_data().df
print(df.head())

# Then you can post-process to change category/bin ids with values
