import copy
from ektelo.data import Relation
from ektelo.matrix import EkteloMatrix 
from ektelo.matrix import Kronecker
from ektelo.matrix import Ones
from ektelo.plans.common import Base
from ektelo.private.measurement import Laplace
from ektelo.private.transformation import Vectorize
from mbi import FactoredInference, Domain
import numpy as np
import pickle
import argparse

class PSynDB(Base):

    def __init__(self, config, strategy):
        self.init_params = {}
        self.config = config
        self.strategy = strategy

    def _fields_by_type(self, field_type):
        return [k for k, v in self.config.items() if v['type'] == field_type]

    def _numerize(self, df):
        for field in self._fields_by_type('categorical'):
            df[field] = df[field].apply(lambda x: self.config[field]['values'].index(x.strip()))

    def _denumerize(self, df):
        for field in self._fields_by_type('categorical'):
            df[field] = df[field].apply(lambda x: self.config[field]['values'][x])
    
    def _sample_numerical(self, df):
        for field in self._fields_by_type('numerical'):
            mn = self.config[field]['domain'][0]
            mx = self.config[field]['domain'][1]
            bin_width = int((mx - mn + 1) / float(self.config[field]['bins']))
            df[field] = mn + bin_width * df[field] + np.random.randint(0, bin_width, df.shape[0])

    def synthesize(self, file_path, eps, seed):
        # setup random state
        prng = np.random.RandomState(seed)

        # load data vector
        relation = Relation(self.config)
        relation.load_csv(file_path)
        self._numerize(relation._df)

        # perform measurement
        attributes = [field_name for field_name in self.config.keys()]
        measurements = []
        w_sum = sum(Ai.weight for Ai in self.strategy.matrices)
        for Ai in self.strategy.matrices:
            w = Ai.weight
            proj=[attributes[i] for i,B in enumerate(Ai.base.matrices) if type(B).__name__ != 'Ones']
            matrix = [B for B in Ai.base.matrices if type(B).__name__ != 'Ones']
            matrix = EkteloMatrix(np.ones((1,1))) if len(matrix)==0 else Kronecker(matrix)
            proj_rel = copy.deepcopy(relation)
            proj_rel.project(proj)
            if proj_rel.df.shape[1] == 0:
                x = np.array([proj_rel.df.shape[0]])
            else:
                x = Vectorize('').transform(proj_rel).flatten()
            y = Laplace(matrix, w*eps/w_sum).measure(x, prng)
            measurements.append((matrix.sparse_matrix(), y, 1.0/w, proj))

        # generate synthetic data
        sizes = [field['bins'] for field in self.config.values()]
        dom = Domain(attributes, sizes)
        engine = FactoredInference(dom)
        model = engine.estimate(measurements)
        df = model.synthetic_data().df
        self._denumerize(df)
        self._sample_numerical(df)

        return df

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['epsilon'] = 1.0
    params['config'] = 'config.pickle'
    params['strategy'] = 'strategy.pickle'
    params['seed'] = None
    params['output'] = 'synthetic.csv'

    return params

if __name__ == '__main__':
    description = 'Run the PSynDB Ektelo plan to generate synthetic data'
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, help='dataset to use')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--config', type=str, help='strategy file')
    parser.add_argument('--strategy', type=str, help='strategy file')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--output', type=str, help='output file')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    config = pickle.load(open(args.config, 'rb'))
    strategy = pickle.load(open(args.strategy, 'rb'))

    plan = PSynDB(config, strategy)
    privatized_data = plan.synthesize(args.dataset, args.epsilon, args.seed)
    privatized_data.to_csv(args.output, index=False)
