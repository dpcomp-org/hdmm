import pandas as pd
import numpy as np
import json
import os

APP_HOME = os.path.join(os.environ['PRIV_HOME'], 'app')

class Parser(object):
    """Guessing the column type based on pandas."""

    def __init__(self):
        pass

    def meta(self, filePath, header=0, sep=','):
        """Guess the columns type
        Argument:
            filePath (string): file path for reading.
            If failed, return error code.

        Returns:
            json with each column name, type and num of distinct with error code.
        """
        meta = []
        try:
            df = pd.read_csv(os.path.join(APP_HOME, filePath), sep=sep+'\s+', delimiter=sep,
                                header=0, skipinitialspace=True)
        except:
            return json.dumps({'filename': os.path.basename(filePath),
                               'data':meta,
                               'error':1,
                               'msg':'Fail reading csv, please check your data frame.'})
        
        try:
            for col in df.columns:
                dis = len(np.unique(df[col]))
                missing = int(df[col].isna().sum())
                if str(df[col].dtype) in ['int64','float64']:
                    minimum = min(df[col].values.tolist())
                    maximum = max(df[col].values.tolist())
                    meta.append({'name':col,'type':'Numerical','distinct':dis,'missing':missing,'min':minimum,'max':maximum})
                elif str(df[col].dtype) == 'object':
                    meta.append({'name':col,'type':'Categorical','distinct':dis,
                                'missing':missing,'min':'N/A','max':'N/A','value':','.join(np.unique(df[col]))})
                else:
                    meta.append({'name':col,'type':'Unknown','distinct':dis,
                                'missing':missing,'min':'N/A','max':'N/A'})
            return json.dumps({'filename': os.path.basename(filePath), 'data':meta,'error':0,'msg':'success'})
        except:
            return json.dumps({'filename': os.path.basename(filePath), 
                               'data':meta,
                               'error':2,
                               'msg':'Fail parsing csv, please check your data frame.'})
