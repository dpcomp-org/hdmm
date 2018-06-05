import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed

data = pd.read_csv('results/togus_scalability.csv')

plt.plot(data.domain**3, 6*data.time, '.-k', linewidth=3, markersize=11)
plt.xlabel('Domain Size', fontsize='large')
plt.ylabel('Time (s)', fontsize='large')
plt.xscale('log')
plt.savefig('scalability_opt+.pdf')
plt.show()

marg = pd.read_csv('results/togus_scalability_marginals_uniform.csv')
marg = marg.groupby('dims').time.mean()[:-1]

plt.plot(10**marg.index, marg.values, '.-k', linewidth=3, markersize=11)
plt.xlabel('Domain Size', fontsize='large')
plt.ylabel('Time (s)', fontsize='large')
plt.xscale('log')
plt.savefig('scalability_optm.pdf')
plt.show()

