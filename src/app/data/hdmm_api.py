from hdmm import workload, templates
from hdmm.workload import *
import math


class api(object):

	def __init__(self):
		self.

	def opt_error(self,work):
		summary = []
		workloads = []

		""" 
		Parsing workloads
		"""

		for wd in postRequest:
			blockinfo = {"columnNames":[],'buildingBlock':[],'p':[]}
			for bb in wd['data']:
				blockinfo['columnNames'].append(bb['name'])
				size = int((float(bb['maximum']) - float(bb['minimum']))/float(bb['bucketSize']))
				pv = math.ceil(size/16.0) if math.ceil(size/16.0) != 2 else math.ceil(size/16.0) - 1
				blockinfo['p'].append(pv)
				if bb['buildingBlock'] == 'identity':
					blockinfo['buildingBlock'].append(Identity(size))
				elif bb['buildingBlock'] == 'allrange':
					blockinfo['buildingBlock'].append(AllRange(size))
				elif bb['buildingBlock'] == 'prefix':
					blockinfo['buildingBlock'].append(Prefix(size))
				else:
					blockinfo['buildingBlock'].append(Total(size))
			wk = float(wd['weight']) * Kron(blockinfo['buildingBlock'])
			workloads.append(wk)
			self.kron = templates.KronPIdentity(wk.domain, blockinfo['p'])
			self.kron.optimize(wk)
			A = [sub.A for sub in self.kron.strategies]
			num_query = int(wk.queries)
			expected_error = wk.expected_error(A)
			summary.append({'wid':wd['wid'], 'expected_error':'{:.4f}'.format(expected_error), 'num_query':num_query, 'workloadString':wd['workloadString']})
		print(summary)
		return summary