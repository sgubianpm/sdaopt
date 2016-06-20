import os
import sys
import argparse
import logging
from jinja2 import Template, Environment, FileSystemLoader
import rpy2.robjects as R
from rpy2.robjects.packages import importr

logger = logging.getLogger(__name__)

def csv(value):
    return value.split(",")


def benchmark(methods, nbruns, deltas, output_dir):
    script_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            '..', 'scripts')
    R.r['source'](os.path.join(script_path, 'bench.funcs.R'))
    R.r['source'](os.path.join(script_path, 'bench.R'))
    bench_funs = R.r['get']('bench.funs')
    for i in range(len(bench_funs)):
        name_index = bench_funs[i].names.index('name')
        dim_index = bench_funs[i].names.index('dim')
        run_index = bench_funs[i].names.index('run')
        if bench_funs[i][run_index][0]:
            print 'Function: %s dim: %s' % (bench_funs[i][name_index][0],
                    str(bench_funs[i][dim_index][0]))
    RF_benchmark = R.r['bench.run']
    RF_save = R.r['save.res']
    results = {}
    r_deltas = R.FloatVector([float(x) for x in deltas])
    for method in methods:
        results[method] = RF_benchmark(bench_funs=bench_funs, meth=method,
                deltas=r_deltas, nb_runs=int(nbruns))
        RF_save(results[method], filepath=os.path.join(output_dir,
            (method + '.rda')))
def main():
    parser = argparse.ArgumentParser(description='GenSA comparison runner')
    parser.add_argument('--methods', dest='methods', action='store',
            default="GenSA,DEoptim,DEoptim_BFGS", type=csv,
            help='List of method to use in: GenSA, DEoptim, DEoptim_LBFGS')
    parser.add_argument('--nbruns', dest='nbruns', action='store',
            default=100, help='Number of runs')
    parser.add_argument('--deltas', dest='deltas', action='store',
            default="1e-5,1e-7,1e-9", type=csv,
            help='Delta values comma separated')
    parser.add_argument('--output-dir', dest='output_dir', action='store',
            default=os.path.join(os.getcwd(),'bench_robjects'),
            help='Delta values comma separated')
    
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        print 'Using folder %s' % args.output_dir
    else:
        print 'Creating folder %s' % args.output_dir
        os.mkdir(args.output_dir)
    
    benchmark(methods=args.methods, nbruns=args.nbruns,
            deltas=args.deltas, output_dir=args.output_dir)
    
if __name__ == "__main__":
    main()
    

