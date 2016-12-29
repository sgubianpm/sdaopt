import os
import sys
import argparse
import logging
#from jinja2 import Template, Environment, FileSystemLoader
from collections import OrderedDict
import rpy2.robjects as R
from rpy2.robjects.vectors import ListVector
#from rpy2.robjects.packages import importr
from .convert2r import GOClass2RConverter
from .go_func_utils import goclass

logger = logging.getLogger(__name__)

def csv(value):
    return value.split(",")

def benchmark(methods, nbruns, deltas, output_dir):
    script_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            'scripts')
    R.r['source'](os.path.join(script_path, 'bench.R'))
    od = OrderedDict()
    # Creating the R list with converted python go testing functions
    print('Converting SciPy test functions to R...')
    counter = 0
    for name, klass in goclass():
        try:
            gocc = GOClass2RConverter(klass)
            res = gocc.fun_no_context(gocc.xglob)
            od.update(((name, gocc.rlist),))
        except Exception:
            print('Ignoring function: {} that can not be translated...'.format(
                name))
            counter += 1
            continue
    print('{} functions have been ignored'.format(counter))
    bench_funs =  ListVector(od)
    print('Launching R benchmark...')
    RF_benchmark = R.r['bench.run']
    RF_save = R.r['save.res']
    results = {}
    r_deltas = R.FloatVector([float(x) for x in deltas])
    for method in methods:
        results[method] = RF_benchmark(benchfuns=bench_funs, meth=method,
                deltas=r_deltas, nbruns=int(nbruns))
        RF_save(results[method], filepath=os.path.join(output_dir,
            (method + '.rda')))

def main():
    parser = argparse.ArgumentParser(description='GenSA comparison runner')
    parser.add_argument('--methods', dest='methods', action='store',
            default="GenSA,rgenoud,DEoptim_BFGS", type=csv,
            help='List of method to use in: GenSA, DEoptim, DEoptim_LBFGS')
    parser.add_argument('--nbruns', dest='nbruns', action='store',
            default=2, help='Number of runs')
    parser.add_argument('--deltas', dest='deltas', action='store',
            #default="1e-5,1e-7,1e-9", type=csv,
            default="1e-8", type=csv,
            help='Delta values comma separated')
    parser.add_argument('--output-dir', dest='output_dir', action='store',
            default=os.path.join(os.getcwd(),'bench_robjects'),
            help='Delta values comma separated')

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        print('Using folder %s' % args.output_dir)
    else:
        print('Creating folder %s' % args.output_dir)
        os.mkdir(args.output_dir)

    benchmark(methods=args.methods, nbruns=args.nbruns,
            deltas=args.deltas, output_dir=args.output_dir)

if __name__ == "__main__":
    main()


