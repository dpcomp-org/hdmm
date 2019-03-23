"""Create strategy matrix for marginal queries (datacubes)
Based on "Differentially private data cubes: optimizing noise sources and consistency",
Ding et al. SIGMOD 2011"""

import math
import sys
import itertools
import workload

def datacube_strategy(W):
    d = len(W.domain)   
 
    L = []
    for key, value in W.weights.items():
        if value > 0:
            L.append(set([i for i in range(d) if key[i] == 1]))
    
    ans = datacube(L, W.domain, 2)

    theta = {}
    for marginal in ans:
        tpl = tuple([1 if i in marginal else 0 for i in range(d)])
        theta[tpl] = 1.0
    
    A = workload.Marginals(W.domain, theta)
    return A.weight_vector()


def datacube(L, dimensions, Lnorm=2):

	theta_L = 0
	theta_R = 2 * len(L)**2

	cur_result = None
	while abs(theta_L - theta_R) > 1:
		theta_M = ( theta_L + theta_R) / 2
		for c in range(len(L)):
			res = feasible(L, dimensions, theta_M, c+1, Lnorm)
			if res is not None:
				cur_result = list(res)
				theta_R = theta_M
			else:
				theta_L = theta_M

        return cur_result


def feasible(L, dimensions, theta, s, Lnorm):
	"""Datacube algorithm: given a list of cuboids in L ( each element of L is a set )
	size of dimesions in `dimesions', determine wether there is a size s subset
	of L covers L with error bounded by theta"""
	
	threshold = math.floor(theta / float(s**(3-Lnorm))) 
	cov = []
	for C in L:
		cov_C = []
		for c in range(len(L)):
			if C.issuperset(L[c]):
				err = 1
				for dim in C.difference(L[c]):
					err *= dimensions[dim]
					
				if err <= threshold:
					cov_C.append(c)
					
		cov.append(set(cov_C))
		
	base_cov = set([])
	strategy = []
	
	for c in range(s):
	# no more than s sets:

		max_size = -1
		set_id = -1
		for c1 in range(len(L)):
			cur_size = len(cov[c1].difference(base_cov))
			if cur_size > max_size:
				max_size = cur_size
				set_id = c1
				
		base_cov = base_cov.union(cov[set_id])
		strategy.append(L[set_id])
		
		if len(base_cov) == len(L):
			break
			
	#print theta, s,'\t', base_cov, '\t', strategy
		
	if len(base_cov) < len(L):
		return None
	else:
		return strategy

def main():
	a=[[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
	seta = map(set, a)
	datacube(seta, [2,2,2])

        W = workload.DimKMarginals([2]*3, [1,2,3])
        print datacube_strategy(W)

if __name__ == "__main__":
    main()
