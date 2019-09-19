# -*- coding: utf-8 -*-
import os, json, math, argparse
import gc
from collections import OrderedDict
import copy
import ektelo
import ektelo.client.inference
import ektelo.client.mapper
import ektelo.client.selection
import ektelo.data
import ektelo.dataset
import ektelo.matrix
import ektelo.private.measurement
import ektelo.private.transformation
import ektelo.support
import numpy as np
from data.parser import Parser
import pickle
from psyndb import PSynDB
from hdmm import templates
from hdmm import error as hdmm_error
from ektelo.matrix import EkteloMatrix, Identity, Kronecker, VStack, Weighted
from ektelo.workload import AllRange, Prefix, Total, union_kron_canonical
import inspect
from mbi import FactoredInference, Domain
from flask import Flask, Response, render_template, redirect, url_for, request, session, abort, flash
from itertools import product
import os
from scipy import stats
import shutil
from ruamel import yaml
import uuid
from werkzeug.utils import secure_filename
import zipfile

parser = argparse.ArgumentParser(description="Priv Publish server")
parser.add_argument('--mode', choices=['debug','production'],help='Running mode.')
parser.add_argument('--ip', help='Server IP address.')
parser.add_argument('--port', default=8555, type=int, help='Server port number.')
args = parser.parse_args()
if args.mode == None or args.ip == None or args.port == None:
    print("Please specify running mode, server ip and port number \n e.g. \"python3 app.py --mode debug --ip 0.0.0.0\"")
    exit(0)
else:
    ENV =  {'mode':args.mode,'ip':args.ip,'port':args.port}

"""
Priv Publish server
"""

UPLOAD_FOLDER = os.environ['PRIV_DATA']
ALLOWED_EXTENSIONS = set(['txt', 'csv'])
#DATA_HOME = os.environ['PRIV_DATA']

app = Flask(__name__,template_folder="templates",static_folder='templates/components')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024 # 8MB
app.secret_key = "pRiV-PuBLish"
pd_parser = Parser()

# home page
@app.route("/", methods=["GET"])
def home():
    return render_template('homepage.html')

# synthetic page
@app.route("/data_generation", methods=["GET"])
def data_generation():
    prod_ip = ENV['ip']
    return render_template('data_generation.html', prod_ip=prod_ip)

@app.route("/local_mode", methods=["GET"])
def local_mode():
    return render_template('local_mode.html')

@app.route("/background", methods=["GET"])
def background():
    return render_template('background.html')

@app.route("/about", methods=["GET"])
def about():
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
  return render_template('404.html'), 404

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_sizes(mn, mx, bk_sz):
    return int(np.ceil((mx - mn + 1) / float(bk_sz)))

def format_meta(raw_meta, config_name='config'):
    assert type(raw_meta) == list, 'meta must be list'

    meta = {config_name: OrderedDict()}
    for raw_field in raw_meta:
        field = copy.deepcopy(raw_field)
        name = field['name']
        del field['name']
        mn = float(field['minimum'])
        del field['minimum']
        mx = float(field['maximum'])
        del field['maximum']
        field['domain'] = [mn, mx]
        field['bins'] = get_sizes(mn, mx, float(field['bucketSize']))
        del field['bucketSize']
        field['type'] = field['type'].lower()
        del field['buildingBlock']
        meta[config_name][name] = field

    return meta

@app.route("/api/export", methods=['GET'])
def export():
    response_dict = json.loads(request.args.get('data'))
    opt_type = response_dict['type']
    data_file_name = response_dict['data_file_name']
    code_file_name = response_dict['code_file_name']
    eps = float(response_dict['eps'])

    synth_filename = gen_synth(data_file_name, code_file_name, eps=eps, seed=0)

    file_contents =  None
    with open(synth_filename) as fd:
        file_contents = fd.read()

    os.unlink(synth_filename)

    return Response(file_contents,
                    mimetype="text/plain",
                    headers={"Content-Disposition":
                             "attachment;filename=synthetic_data.csv"})

@app.route("/api/code", methods=['GET'])
def code():
    response_dict = json.loads(request.args.get('data'))
    opt_type = response_dict['type']
    filepath = response_dict['code_file_name']

    file_contents =  None
    with open(filepath, 'rb') as fd:
        file_contents = fd.read()

    return Response(file_contents,
                    mimetype="text/plain",
                    headers={"Content-Disposition":
                             "attachment;filename=ektelo_code.zip"})

# csv parser
@app.route("/api/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        file_path = os.path.join(UPLOAD_FOLDER, 'adult.csv')
    if request.method == 'POST':
        if 'file' not in request.files:
            return {'error':-1,'msg':'No file part in upload request.'}
        file = request.files['file']
        if file.filename == '':
            return {'error':-1,'msg':'No file name.'}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

    meta = pd_parser.meta(file_path, 0, ',')
    print(file_path)
    if request.method == 'GET':
        reorder = json.loads(meta)
        adult_order = ['age', 'race', 'sex', 'workclass', 'capital-gain', 'education', 'hours-per-week', 'native-country', 'occupation', 'marital-status', 'income', 'relationship', 'capital-loss', 'fnlwgt']
        reorder['data'] = [[row for row in reorder['data'] if row['name'] == item][0] for item in adult_order]
        meta = json.dumps(reorder)
    return meta

"""
HDMM Optimization

TODO:
1. We could have memory problem here if workloads contained too many queries.
   Thus, some limits should be set and return oversize error if too large workloads.

2. The cdf and pdf graphs will look incorrect if workloads only had few queries. (see np.histogram)
   Thus, return tinysize error if workloads too small.

3. According to workloads type trying to return queries where max min error occurs.
"""

def domain(mat):
    # george is not sure about the max below, the Concat object used to have
    # a domain field so we used cat.domain instead
    if hasattr(mat, 'n'):
        return mat.n
    elif isinstance(mat, VStack):
        return domain(mat.matrices[0])
    elif isinstance(mat, Kronecker):
        return tuple([domain(m) for m in mat.matrices])
    elif isinstance(mat, Weighted):
        return domain(mat.base)
    else:
        raise NotImplementedError('objects of type %s do not support "domain" operation' % type(mat))

def parse_customized(bb, size):
    domainMatrix = np.zeros((len(bb['customizedQueries']),size))
    if bb['type'] == 'Numerical':
        for qid, query in enumerate(bb['customizedQueries']):
            index = [math.ceil(i) for i in np.divide(np.array(query) - float(bb['minimum']), float(bb['bucketSize']))]
            domainMatrix[qid, index] = 1
    else:
        for qid, query in enumerate(bb['customizedQueries']):
            index = [bb['values'].index(item) for item in query]
            domainMatrix[qid, index] = 1
    return domainMatrix

def process_workload(wd, eps):
    blockinfo = {"columnNames":[],'buildingBlock':[],'p':[]}
    for bb in wd['data']:
        blockinfo['columnNames'].append(bb['name'])
        size = int((float(bb['maximum']) - float(bb['minimum']))/float(bb['bucketSize']) + 1)
        pv = math.ceil(size/16.0) if math.ceil(size/16.0) != 2 else math.ceil(size/16.0) - 1
        if bb['buildingBlock'] == 'identity':
            blockinfo['buildingBlock'].append(Identity(size))
            pv = 1
        elif bb['buildingBlock'] == 'allrange':
            blockinfo['buildingBlock'].append(AllRange(size))
        elif bb['buildingBlock'] == 'prefix':
            blockinfo['buildingBlock'].append(Prefix(size))
        elif bb['buildingBlock'] == 'customized':
            domainMatrix = parse_customized(bb, size)
            blockinfo['buildingBlock'].append(EkteloMatrix(domainMatrix))
            pv = 1
        else:
            blockinfo['buildingBlock'].append(Total(size))
            pv = 1
        blockinfo['p'].append(pv)
        gc.collect()
    gc.collect()
    wgt = np.sqrt(float(wd['weight']))
    return wgt * Kronecker(blockinfo['buildingBlock']), blockinfo

def optimize_workload(wk, blockinfo):
    kron = templates.KronPIdentity(blockinfo['p'], domain(wk))
    kron.optimize(wk)
    gc.collect()

    return kron.strategy()

def gen_code(strategy, workload, meta):
    # create ektelo plan
    attributes = [m['name'] for m in meta]
    sizes = [get_sizes(float(m['minimum']), float(m['maximum']), float(m['bucketSize'])) for m in meta]
    config = format_meta(meta)['config']

    # export Ektelo data
    code_filename = export_ektelo(workload, strategy, config)

    return code_filename

def gen_synth(data_file_name, code_file_name, eps=0.1, seed=0):
    code_path = code_file_name.split('.')[0]

    # load PSynDB instance
    with open(os.path.join(code_path, 'strategy.pickle'), 'rb') as fd:
        strategy = pickle.load(fd)

    with open(os.path.join(code_path, 'config.pickle'), 'rb') as fd:
        config = pickle.load(fd)

    p_syn_db = PSynDB(config, strategy)

    # generate synthetic data
    data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_file_name)
    df = p_syn_db.synthesize(data_file_path, eps, seed)
    synth_filename = os.path.join('/tmp/priv_publish', str(uuid.uuid4()) + '.csv')
    df.to_csv(synth_filename, index=False)

    return synth_filename

def matrix_rewriter(M):
    # locate module
    if type(M).__name__ in dir(ektelo.matrix):
        base = 'ektelo.matrix.'
    elif type(M).__name__ in dir(ektelo.workload):
        base = 'ektelo.workload.'
    else:
        return M

    # get class
    ektelo_class = eval(base+type(M).__name__)
    if not inspect.isclass(ektelo_class):
        return M

    # load as ektelo matrix
    Mp = ektelo_class.__new__(ektelo_class)
    for k,v in M.__dict__.items():
        if type(v).__name__ in dir(ektelo.matrix) or type(v).__name__ in dir(ektelo.workload):
            setattr(Mp, k, matrix_rewriter(v))
        elif type(v) == list:
            setattr(Mp, k, [matrix_rewriter(entry) for entry in v])
        elif type(v) == tuple:
            setattr(Mp, k, tuple([matrix_rewriter(entry) for entry in v]))
        else:
            setattr(Mp, k, v)

    return Mp

def export_ektelo(W, A, config):
    code_dir = os.path.join('/tmp/priv_publish', str(uuid.uuid4()))
    os.mkdir(code_dir)

    config_file = os.path.join(code_dir, 'config.pickle')
    with open(config_file, 'wb') as fd:
        pickle.dump(config, fd)

    strategy_file = os.path.join(code_dir, 'strategy.pickle')
    with open(strategy_file, 'wb') as fd:
        #pickle.dump(matrix_rewriter(union_kron_canonical(A)), fd)
        pickle.dump(union_kron_canonical(A), fd)

    workload_file = os.path.join(code_dir, 'workload.pickle')
    with open(workload_file, 'wb') as fd:
        #pickle.dump(matrix_rewriter(W), fd)
        pickle.dump(W, fd)

    code_file = os.path.join(code_dir, 'psyndb.py')
    shutil.copyfile(os.path.join(os.environ['PRIV_HOME'], 'psyndb.py'), code_file)

    zip_file = code_dir + '.zip'
    zipf = zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED)
    zipf.write(config_file, os.path.join('ektelo_code', os.path.basename(config_file)))
    zipf.write(strategy_file, os.path.join('ektelo_code', os.path.basename(strategy_file)))
    zipf.write(workload_file, os.path.join('ektelo_code', os.path.basename(workload_file)))
    zipf.write(code_file, os.path.join('ektelo_code', os.path.basename(code_file)))
    zipf.close()

    return zip_file

def get_workload():
    postRequest = request.get_json(force=True)
    print(postRequest['workloadsType'])
    workloads = []
    workloadString = []
    eps = 1.0
    wds = list(postRequest['workloads'])

    num_query = expected_error = 0
    for wd in wds:
        wk, blockinfo = process_workload(wd, eps)
        workloads.append(wk)
        workloadString.append(wd['workloadString'])
    error_wd = {'wid':'Union','workloadString':wd['workloadString']}
    return VStack(workloads), wds

def calculate_workload_error_union(wk, strategy, eps):
    " Strategies that are a union still need special treatment """
    expected_error = 0
    per_query_error = np.array([])
    table = []
    weights = np.array([])
    effective_queries = 0
    for Wi, Ai in zip(wk.matrices, strategy.matrices):
        effective_queries += Wi.weight**2 * Wi.shape[0]
        err = hdmm_error.expected_error(Wi.base, Ai.base, eps=eps) / Ai.weight**2
        expected_error += Wi.weight**2 * err
        pqe = hdmm_error.per_query_error_sampling(Wi.base,Ai.base,eps=eps,normalize=True)/Ai.weight
        per_query_error = np.append(per_query_error, pqe)
        wgt = np.ones(pqe.size) * Wi.weight**2 * Wi.shape[0]
        weights = np.append(weights, wgt)

        rootmse = np.sqrt(err / Wi.shape[0])
        if np.var(pqe) < 1e-2:
            pdf = False
            pdf_x = False
        else:
            density = np.histogram(pqe, bins='auto', density=True)
            pdf = density[0].tolist()
            pdf_x = density[1][0:-1].tolist()
        res = {
                'pdf': pdf,
                'pdf_x': pdf_x,
                'method': 'HDMM',
                'expected_error': rootmse,
                'num_query': int(Wi.shape[0]),
                }
        table.append(res)

    rootmse = np.sqrt(expected_error / effective_queries)

    if np.var(per_query_error) < 1e-2:
        pdf = False
        pdf_x = False
    else:
        density = np.histogram(per_query_error, bins=100, weights=weights, density=True)
        pdf = density[0].tolist()
        pdf_x = density[1][0:-1].tolist()

    graph = {
            'pdf': pdf,
            'pdf_x': pdf_x,
            'method': 'HDMM',
            'expected_error': rootmse,
            'num_query': int(wk.shape[0]),
            }

    return table, graph

def calculate_workload_error_marginals(wk, strategy, eps):
    " Strategies that are a union still need special treatment """
    expected_error = 0
    table = []
    weights = np.array([])
    effective_queries = 0
    for Wi in wk.matrices:
        print(Wi)
        effective_queries += Wi.weight**2 * Wi.shape[0]
        err = hdmm_error.expected_error(Wi.base, strategy, eps=eps)
        expected_error += Wi.weight**2 * err

        rootmse = np.sqrt(err / Wi.shape[0])
        res = {
                'pdf': False,
                'pdf_x': False,
                'method': 'HDMM',
                'expected_error': rootmse,
                'num_query': int(Wi.shape[0]),
                }
        table.append(res)

    rootmse = np.sqrt(expected_error / effective_queries)

    graph = {
            'pdf': False,
            'pdf_x': False,
            'message' : "Optimized strategy is marginals, doesn't support density plots.",
            'method': 'HDMM',
            'expected_error': rootmse,
            'num_query': int(wk.shape[0]),
            }

    return table, graph

def calculate_workload_error_default(wk, strategy, eps):
    " Strategies that are a union still need special treatment """
    expected_error = 0
    per_query_error = np.array([])
    table = []
    weights = np.array([])
    effective_queries = 0
    for Wi in wk.matrices:
        print(Wi)
        effective_queries += Wi.weight**2 * Wi.shape[0]
        err = hdmm_error.expected_error(Wi.base, strategy, eps=eps)
        expected_error += Wi.weight**2 * err
        pqe = hdmm_error.per_query_error_sampling(Wi.base,strategy,eps=eps,normalize=True)
        per_query_error = np.append(per_query_error, pqe)
        wgt = np.ones(pqe.size) * Wi.weight**2 * Wi.shape[0]
        weights = np.append(weights, wgt)

        rootmse = np.sqrt(err / Wi.shape[0])
        if np.var(pqe) < 1e-2:
            pdf = False
            pdf_x = False
        else:
            density = np.histogram(pqe, bins='auto', density=True)
            pdf = density[0].tolist()
            pdf_x = density[1][0:-1].tolist()
        res = {
                'pdf': pdf,
                'pdf_x': pdf_x,
                'method': 'HDMM',
                'expected_error': rootmse,
                'num_query': int(Wi.shape[0]),
                }
        table.append(res)

    rootmse = np.sqrt(expected_error / effective_queries)

    if np.var(per_query_error) < 1e-2:
        pdf = False
        pdf_x = False
    else:
        density = np.histogram(per_query_error, bins=100, weights=weights, density=True)
        pdf = density[0].tolist()
        pdf_x = density[1][0:-1].tolist()

    graph = {
            'pdf': pdf,
            'pdf_x': pdf_x,
            'method': 'HDMM',
            'expected_error': rootmse,
            'num_query': int(wk.shape[0]),
            }

    return table, graph

def calculate_workload_error2(wk, strategy, eps):
    # Deprecated
    expected_error = hdmm_error.rootmse(wk, strategy, eps=eps)
    per_query_error = hdmm_error.per_query_error_sampling(wk,strategy,100000,eps,normalize=True)
    if np.var(per_query_error) < 1e-2:
        pdf = False
        pdf_x = False
    else:
        density = np.histogram(per_query_error, bins='auto', density=True)
        pdf = density[0].tolist()
        pdf_x = density[1][0:-1].tolist()
    res = {
            'pdf': pdf,
            'pdf_x': pdf_x,
            'method': 'HDMM',
            'expected_error': expected_error,
            'num_query': int(wk.shape[0]),
            }
    return res

@app.route("/api/hdmm3", methods=["POST"])
def HDMM3():
    eps = 1.0
    wk, wds = get_workload()
    ns, k = domain(wk), len(wk.matrices)
    temp = templates.BestHD(ns, k)
    temp.optimize(wk)

    A = temp.strategy()

    if type(temp.best) == templates.Union:
        table, graph = calculate_workload_error_union(wk, A, eps)
    elif type(temp.best) == templates.Marginals:
        table, graph = calculate_workload_error_marginals(wk, A, eps)
    else:
        table, graph = calculate_workload_error_default(wk, A, eps)
        #table = [calculate_workload_error2(Wi, A, eps) for Wi in wk.matrices]
        #graph = calculate_workload_error2(wk, A, eps)

    metrics = { }
    metrics['Table'] = table
    # for index, mts in enumerate(metrics['Table']):
    #     metrics['Table'][index] =
    metrics['Graph'] = graph
    metrics['stage'] = 'HDMM Optimization Complete'

    #data_file_name = wd['filename'] if 'filename' in wd else ''
    #code_filename = gen_code(strategy, cat, wd['meta'])

    #metrics = calculate_workload_error(error_wd, cat, strategy, eps, len(postRequest), 'HDMM2', 1000*len(postRequest), postRequest['workloadsType'])
    #metrics.update({'percent':'75','stage':'HDMM2'})
    #metrics.update({'data_file': data_file_name, 'code': code_filename})

    wd = wds[0]
    data_file_name = wd['filename'] if 'filename' in wd else ''
    code_filename = gen_code(A, wk, wd['meta'])
    metrics.update({'data_file': data_file_name, 'code_file': code_filename})

    return json.dumps(metrics)

@app.route("/api/identity", methods=["POST"])
def Laplace():
    eps = 1.0
    wk, wds = get_workload()
    identity = Kronecker([Identity(n) for n in domain(wk)])
    metrics = calculate_workload_error_default(wk, identity, eps)[1]
    metrics['stage'] = 'Identity Baseline Complete'
    metrics['method'] = 'Identity'
    return json.dumps(metrics)

if __name__ == '__main__':
    debug = ENV['mode'] == 'debug'
    app.run(host='0.0.0.0', port=ENV['port'], debug=debug, threaded=True)
