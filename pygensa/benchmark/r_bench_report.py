import os
import sys
import argparse
import logging
from collections import OrderedDict
from cStringIO import StringIO
from jinja2 import Template, Environment, FileSystemLoader
import rpy2.robjects as R
from rpy2.robjects.packages import importr

logger = logging.getLogger(__name__)

def csv(value):
    return value.split(",")

nb_metric=6


def main():
    parser = argparse.ArgumentParser(description='GenSA comparison tables generator')
    parser.add_argument('--R-files', dest='rfiles', action='store', help='R Input file', type=csv, required=True) and None
    parser.add_argument('--output-dir', dest='output_dir', action='store', default=os.path.join(os.getcwd(),'bench_reports'), help='Folder path to where report is generated') and None
    parser.add_argument('--format', dest='frmat', action='store', default='text', help='Ouput format, tex or csv') and None
    args = parser.parse_args()
    create_report(args.output_dir, rfiles=args.rfiles, frmat=args.frmat)


def create_report(output_dir, rfiles=None, frmat='tex'):
    file_str = StringIO()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    results = OrderedDict()
    for file in rfiles:
        # Load R file
        print('Loading R file: %s' % file)
        R.r['load'](file)
        method_name = os.path.splitext(os.path.basename(file))[0].replace('_','-')
        results[method_name] = R.r['get']("obj")

    func_names = [x for x in results[method_name].names]
    delta_names = [x for x in results[method_name][0].names]
    metric_names = [x for x in results[method_name][0][0].names]

    deltas = {}
    deltas = deltas.fromkeys(delta_names)
    for key in deltas.keys():
        funs = {}
        funs = funs.fromkeys(func_names)
        deltas[key] = funs
        for key in funs.keys():
            methods = {}
            methods = methods.fromkeys(results.keys())
            funs[key] =methods
            for key in methods.keys():
                metrics = {}
                metrics = metrics.fromkeys(metric_names)
                methods[key] = metrics

    for method in results.keys():
        for func_name in results[method].names:
            name_idx = results[method].names.index(func_name)
            print('\tFunction: %s (%s)' % (func_name, str(name_idx)))
            for delta in results[method][name_idx].names:
                delta_idx = results[method][name_idx].names.index(delta)
                print('\t\tDelta: %s (%s)' % (delta,str(delta_idx)))
                for metric in results[method][name_idx][delta_idx].names:
                    metric_idx = results[method][name_idx][delta_idx].names.index(metric)
                    deltas[delta][func_name][method][metric] = results[method][name_idx][delta_idx][metric_idx][0]
                    #if deltas[delta][func_name][method][metric] == R.NA_Real or:
                    #    deltas[delta][func_name][method][metric] = float('nan')
                    if str(deltas[delta][func_name][method][metric]) == 'NA':
                        deltas[delta][func_name][method][metric] = float('nan')

    if frmat == 'tex':
        file_str.write('''
        \\documentclass{article}
        \\newcommand{\specialcell}[2][c]{\%
        \\begin{tabular}[#1]{@{}c@{}}#2\end{tabular}}
        \\begin{document}
        \\scriptsize
        ''')

        for delta in delta_names:
            file_str.write('\\section{For tolerance ' + delta + '} \n \\begin{tabular}{|c|c|c|c|c|} \\hline \n')
            file_str.write('func ')
            for method in results.keys():
                file_str.write('& '+ method)
            file_str.write('\\\ \\hline \n')
            for func_name in func_names:
                file_str.write(func_name)
                for method in results.keys():
                    file_str.write(' & ' + str(round(deltas[delta][func_name][method]['success'],1)) + '\%')
                file_str.write('\\\ \n')
                for method in results.keys():
                    file_str.write(' & %s (B)' % round(deltas[delta][func_name][method]['best'],1))
                file_str.write('\\\ \n')
                for method in results.keys():
                    file_str.write(' & %s (A)' % round(deltas[delta][func_name][method]['aveSucFC'],1))
                file_str.write('\\\ \n')
                for method in results.keys():
                    file_str.write(' & $\pm$ %s (S)' % round(deltas[delta][func_name][method]['se'],1))
                file_str.write('\\\ \n')
                for method in results.keys():
                    file_str.write(' & %s (W)' % round(deltas[delta][func_name][method]['worst'],1))
                file_str.write('\\\ \n')
                for method in results.keys():
                    file_str.write(' & %s (T)' % round(deltas[delta][func_name][method]['aveFC'],1))
                file_str.write('\\\ \n \\hline \n')
            file_str.write('\end{tabular}')

        file_str.write('\n\end{document}')
        file = open('results.tex', 'w')
        s = file_str.getvalue().replace('-inf','NA').replace('inf','NA').replace('nan','NA')
        #print(s)
        file.write(s)
        file.close()
    else:
        file_str.write('function name')
        for method in results.keys():
            file_str.write('\t{}'.format(method))
        file_str.write('\n')
        for func_name in func_names:
            file_str.write(func_name)
            for method in results.keys():
                file_str.write('\t' + str(round(deltas[delta][func_name][method]['success'],1)) + '%')
                file_str.write(' (%s)' % round(deltas[delta][func_name][method]['aveSucFC'],1))
            file_str.write('\n')
#            for method in results.keys():
#                file_str.write('\t%s (B)' % round(deltas[delta][func_name][method]['best'],1))
#            file_str.write('\n')
            #for method in results.keys():
            #    file_str.write(' (%s)' % round(deltas[delta][func_name][method]['aveSucFC'],1))
            #file_str.write('\n')
#            for method in results.keys():
#                file_str.write('\t(+/-) %s (S)' % round(deltas[delta][func_name][method]['se'],1))
#            file_str.write('\n')
#            for method in results.keys():
#                file_str.write('\t%s (W)' % round(deltas[delta][func_name][method]['worst'],1))
#            file_str.write('\n')
#            for method in results.keys():
#                file_str.write('\t%s (T)' % round(deltas[delta][func_name][method]['aveFC'],1))
#            file_str.write('\n')
        file = open('results.csv', 'w')
        s = file_str.getvalue().replace('-inf','NA').replace('inf','NA').replace('nan','NA')
        #print(s)
        file.write(s)
        file.close()
