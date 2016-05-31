#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Sylvain Gubian"
__license__ = "Simplified BSD license"
__version__ = "0.0.1"
__email__ = "Sylvain.Gubian@pmi.com"
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import Color
from special_bench_go import BenchStore
from special_bench_go import BenchUnit
import scipy.cluster.hierarchy as sch
import fastcluster

"""
fastcluster python package is used instead of the scipy one because I get an
issue in linkage call in scipy:
In [8]: z = scipy.cluster.hierarchy.linkage(a)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-8-2023ef72a731> in <module>()
----> 1 z = scipy.cluster.hierarchy.linkage(a)

/gpfshpc/software/python3-science/lib/python3.4/site-packages/scipy/cluster/hierarchy.py in linkage(y, method, metric)
    655
    656             if method == 'single':
--> 657                 _hierarchy.slink(dm, Z, n)
    658             else:
    659                 _hierarchy.linkage(dm, Z, n,

scipy/cluster/_hierarchy.pyx in scipy.cluster._hierarchy.slink (scipy/cluster/_hierarchy.c:11690)()

TypeError: slink() takes exactly 2 positional arguments (3 given)
"""


def get_data(path):
    data = BenchStore.report(path, raw_values=True)
    return data


def success_color_scale(start_color, end_color, nb):
    # Generating color map
    start = Color(start_color)
    colors = list(start.range_to(Color(end_color),nb))
    colors = [c.get_hex_l() for c in colors]
    return colors

def get_data_info(data, info='nbruns'):
    methods = data.keys()
    fnames = data[methods[0]].keys()
    metrics = data[methods[0]][fnames[0]].keys()
    nbruns = len(data[methods[0]][fnames[0]][metrics[0]])
    if info == 'fnames':
        return fnames
    elif info == 'methods':
        return methods
    elif info == 'metrics':
        return metrics
    else:
        return nbruns

def all_func_nb_call(data):
    # Plotting for all functions
    d = {}
    fnames = []
    nb_runs = get_data_info(data)
    for k in data:
        d[k] = np.empty([nb_runs, len(data[k])])
        for i, f in enumerate(data[k]):
            d[k][:,i] = data[k][f]['ncall']
            if f not in fnames:
                fnames.append(f)
        df = pd.DataFrame(d[k], columns=fnames)
        axes  = df.plot.box(title=k)
        axes.set_ylim([0, 10000])
        ax = axes.set_xticklabels(axes.xaxis.get_majorticklabels(),
                rotation=90)
        plt.show(k)

def overall_fncall(data):
    doa = {}
    reliability = {}
    nb_runs = get_data_info(data)
    fnames = get_data_info(data, info='fnames')
    mat = np.empty([nb_runs * len(fnames), len(data)])
    for k in data:
        reliability[k] = []
        doa[k] = []
        for i, f in enumerate(data[k]):
            values = data[k][f]['ncall']
            success = np.sum(
                data[k][f]['success']) * 100 / nb_runs
            if np.isnan(success):
                success = 0
            reliability[k].append(int(success))
            # Normalizing values
            values /= np.max(np.abs(values), axis=0)
            if i == 0:
                doa[k][:] = values
            else:
                doa[k] = np.append(doa[k], values)
        reliability[k] = int(np.round(np.mean(reliability[k])))
    df = pd.DataFrame(doa, columns=data.keys())
    axes  = df.plot.box(notch=True, title='Normalized Overall function calls')
    axes.set_ylim([-0.5, 1.1])
    # ax = axes.set_xticklabels(axes.xaxis.get_majorticklabels(), rotation=90)
    plt.show()

def heat_map_reliability(data):
    nb_runs = get_data_info(data)
    nb_func = len(get_data_info(data, 'fnames'))
    methods = get_data_info(data, 'methods')
    mat = np.empty([nb_func, len(methods)])
    for j, k in enumerate(data):
        for i, f in enumerate(data[k]):
            success = np.sum(
                data[k][f]['success']) * 100 / nb_runs
            if np.isnan(success):
                success = 0
            mat[i, j] = success
    #mat.sort(axis=0)
    fig = plt.figure()
    ax1 = fig.add_axes([0.7, 0.1, 0.18, 0.8])
    Y = fastcluster.linkage(mat, method='ward')
    Z1 = sch.dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    axmatrix = fig.add_axes([0.1, 0.1 , 0.6, 0.8])
    axmatrix.set_title(
            'Success rate across test functions (reliability over 200 runs)')
    im = axmatrix.matshow(mat[Z1['leaves'], :], aspect='auto', origin='lower',
            cmap=plt.cm.RdYlGn, )
    methods.insert(0, ' ')
    axmatrix.set_xticklabels(methods)
    # Reorder functions indexes:
    c = np.arange(0, nb_func)[Z1['leaves']]
    axmatrix.set_yticks(np.arange(0, nb_func, 10))
    axmatrix.set_yticklabels(c)
    axmatrix.set_ylabel('Test function number')
    axcolor = fig.add_axes([0.9,0.1,0.02,0.8])
    plt.colorbar(im, cax=axcolor)
    fig.show()

def main(data_path):
    print('Retrieving data...')
    data = get_data(data_path)
    #overall_fncall(data)
    heat_map_reliability(data)
    try:
        input('Press a key to end')
    except Exception:
        pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Path to benchmark data folder has to be provided as arg')
        sys.exit(-1)
    main(sys.argv[1])
