from hdmm.templates import DefaultKron
from hdmm.workload import Prefix2D
from mbi import FactoredInference, Domain
import numpy as np

# set up domain and workload
attributes = [0, 1] #  should be the names of the columns, for now just using 0 and 1
sizes = [32, 32]
dom = Domain(attributes, sizes)
W = Prefix2D(32)

# optimize strategy using HDMM
template = DefaultKron(sizes)
template.optimize(W)
A = template.strategy()


# prepare HDMM measuremetns for PGM estimation + synthetic data
A = A.sparse_matrix() # this is not necessary but may be faster than using A as is
x = np.random.rand(np.prod(sizes)) * 100
y = A.dot(x) + np.random.laplace(loc=0, scale=1, size=A.shape[0])

measurements = [(A, y, 1.0, attributes)] # using 1 measurement over all attributes.  Note that we can always do this but may not be the most efficient if measurements can be expressed over lower dimensional marginals.

engine = FactoredInference(dom)
model = engine.estimate(measurements)

df = model.synthetic_data().df
print(df.head())

# Then you can post-process to change category/bin ids with values
