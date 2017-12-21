import numpy as np
import pandas as pd
from IPython import embed

results = pd.read_csv('results/dawa_results.csv')
print results


rmse = lambda x: np.sqrt(np.sum(x**2) / len(x))

avgs = results.groupby(['dataset', 'epsilon','domain'])[['GreedyH','Opt']].apply(rmse)

embed()

rel = avgs.div(avgs.Opt, axis=0)
#rel = avgs.div(avgs.min(axis=1), axis=0)

stats = rel.GreedyH.groupby(level=[1,2])

table = pd.DataFrame()
table['min'] = stats.min()
table['median'] = stats.median()
table['max'] = stats.max()

print table.round(decimals=2)
