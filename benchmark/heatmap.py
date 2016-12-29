#!/usr/bin/env python
##############################################################################
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
##############################################################################
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .benchstore import BenchStore
import scipy.cluster.hierarchy as sch
import fastcluster

__author__ = "Sylvain Gubian"
__license__ = "Simplified BSD license"
__version__ = "0.0.1"
__email__ = "Sylvain.Gubian@pmi.com"


def get_data(path):
    data = BenchStore.report(path, kind='raw')
    return data


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
            d[k][:, i] = data[k][f]['ncall']
            if f not in fnames:
                fnames.append(f)
        df = pd.DataFrame(d[k], columns=fnames)
        axes = df.plot.box(title=k)
        axes.set_ylim([0, 10000])
        axes.set_xticklabels(
                axes.xaxis.get_majorticklabels(),
                rotation=90
                )
        plt.show(k)


def overall_fncall(data):
    doa = {}
    reliability = {}
    nb_runs = get_data_info(data)
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
    axes = df.plot.box(notch=True, title='Normalized Overall function calls')
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
    # mat.sort(axis=0)
    fig = plt.figure()
    ax1 = fig.add_axes([0.7, 0.1, 0.18, 0.8])
    Y = fastcluster.linkage(mat, method='ward')
    Z1 = sch.dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    axmatrix = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    axmatrix.set_title(
            'Success rate across test functions (reliability over 200 runs)')
    im = axmatrix.matshow(
            mat[Z1['leaves'], :], aspect='auto', origin='lower',
            cmap=plt.cm.RdYlGn,
            )
    methods.insert(0, ' ')
    axmatrix.set_xticklabels(methods)
    # Reorder functions indexes:
    c = np.arange(0, nb_func)[Z1['leaves']]
    axmatrix.set_yticks(np.arange(0, nb_func, 10))
    axmatrix.set_yticklabels(c)
    axmatrix.set_ylabel('Test function number')
    axcolor = fig.add_axes([0.9, 0.1, 0.02, 0.8])
    plt.colorbar(im, cax=axcolor)
    # fig.show()
    fig.save('heatmap.pdf', bbox_inches='tight', format='pdf')


def main(data_path):
    print('Retrieving data...')
    data = get_data(data_path)
    # overall_fncall(data)
    heat_map_reliability(data)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Path to benchmark data folder has to be provided as arg')
        sys.exit(-1)
    main(sys.argv[1])
