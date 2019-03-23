import argparse
import pandas as pd
import numpy as np
from IPython import embed
import optimize
import workload
import time
import matplotlib.pyplot as plt
import census_workloads

"""
Compare optimization with different p values 
"""

if __name__ == '__main__':
   
    try:
        df = pd.read_csv('hyperparam.csv', index_col=0)
    except: 
        domain = 256
        restarts = 25 
        
        workloads = {}
        workloads['All Range'] = workload.AllRange(domain)
        workloads['Prefix'] = workload.Prefix(domain)
#        workloads['Census Age'] = census_workloads.CensusSF1().project_and_merge([[4]])
        workloads['1D Marginal'] = workload.Marginal(domain)
        workloads['2D Prefix'] = workload.PrefixIdentity2D(domain)

        df = pd.DataFrame(columns=['workload', 'p', 'error', 'time'])
        idx = 0
        I = np.eye(256)

        for name, W in workloads.items():
            if name == '2D Prefix':
                err = W.expected_error([I,I])
            else:
                err = W.expected_error(I)
            df.loc[idx] = [name, 0, err, 0]
            idx += 1
            p = 1 
            while p <= domain:
                print name, p
                t0 = time.time()
                if name == '2D Prefix':
                    A = optimize.restart_union_kron(W, restarts, [p,p])
                else:
                    A = optimize.restart_optimize(W, restarts, p)
                t1 = time.time()
                err = W.expected_error(A)
#                err = workload.Workload.expected_error(W, A)
                df.loc[idx] = [name, p, err, (t1-t0)/restarts]
                idx += 1
                p = p*2   

        df = df.set_index(['workload','p']).error
        df = np.sqrt(df / df.min(level=0))
        df = df.unstack(level=0)

        df.to_csv('hyperparam.csv')

    R = df['All Range']
    plt.plot(R.index, R.values, 'o')

    for p, e in zip(R.index[:-1], R.values[:-1]):
        plt.text(p*1.1, e+0.1, '(%d, %.2f)' % (p,e), rotation=30)
        #plt.annotate('(%d, %.2f)' % (p,e), xy=(p, e))

    plt.xscale('log')
    plt.xticks([1,16,256], ['1','16','256'])
    plt.xlabel('p')
    plt.ylabel('Relative Error')
    plt.savefig('hyperparam.pdf')
    plt.show()


     
#    df.plot(style='o')
#    plt.xticks([1, 16, 256])
#    plt.xlabel('p')
#    plt.ylabel('Relative Error')
#    plt.savefig('hyperparam.pdf')
#    plt.show()
