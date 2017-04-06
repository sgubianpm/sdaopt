##############################################################################
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
##############################################################################
# -*- coding: utf-8 > -*-
import numpy as np
import numpy.testing as npt
try:
    from scipy.optimize import gensa
except:
    from pygensa import gensa
__author__ = "Sylvain Gubian"
__copyright__ = "Copyright 2016, PMP SA"
__license__ = "GPL2.0"
__email__ = "Sylvain.Gubian@pmi.com"

A = 1
C = 39.432
EPSILON = 1
M = 6
K = 9


def sutton_chen(x):
    x = x.reshape((int(x.size/3), 3))
    idx = np.array(list(np.arange(0, x.shape[0])) * x.shape[0])
    jdx = np.concatenate([[a] * x.shape[0] for a in range(
        0, x.shape[0])])
    index = np.column_stack((idx, jdx))
    index = index[index[:, 0] < index[:, 1], :]
    rij = np.zeros(index.shape[0])
    for i in range(index.shape[0]):
        rij[i] = np.sqrt(np.sum((x[index[i, 0], :] - x[
            index[i, 1], :]) ** 2))
    f1s = np.zeros(index.shape[0])
    rhos = np.zeros(index.shape[0])
    for i in range(0, x.shape[0]):
        idx = np.logical_or(index[:, 0] == i, index[:, 1] == i)
        f1s[i] = 0.5 * (A**K) * np.sum(1/rij[idx] ** K)
        rhos[i] = (A**M) * sum(1/(rij[idx]) ** M)
    return np.sum(f1s - C * np.sqrt(rhos))


def test_sutton_chen():
    x_global = np.array([
        -0.2900181566, -0.2176381343, -0.2842129662,
        -0.4342237494, -0.1257555181, -0.9118061294,
        0.1138133902, 0.2919699092, -0.3023945653,
        -0.0303922026, 0.3838525254, -0.9299877285,
        -0.5060632976, 0.3614667414, -0.4868774344,
        0.1856529384, -0.1952523504, -0.7273232603])
    npt.assert_almost_equal(sutton_chen(x_global), -1163.9639632)


def main():
    n_particles = 6
    lw = [-0.7] * (3 * n_particles)
    up = [0.7] * (3 * n_particles)
    np.random.seed(123)
    ret = gensa(sutton_chen, None, bounds=(zip(lw, up)))
    # np.set_printoptions(precision=4)
    print('xmin =\n{}'.format(np.array2string(ret.x, max_line_width=40)))
    print("global minimum: f(xmin) = {}".format(ret.fun))

if __name__ == '__main__':
    main()
