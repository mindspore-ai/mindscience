# Copyright 2021 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""utils"""
from mindspore.ops import deepcopy
from mindspore import Tensor
import numpy as np
import mindspore as ms
import mindspore.ops as ops


def load_cfg(filename):
    """load_cfg
    Load configurations from yaml file and return a namespace object

    :param filename:
    """
    import yaml
    from argparse import Namespace
    with open(filename, "r") as f:
        cfg = yaml.safe_load(f)
    return Namespace(**cfg)

def collate_list_of_dicts(list_of_dicts, batch_info=None):
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}
    # collated = {k: Tensor(np.vstack(dict_of_lists[k]), dtype=ms.float32) for k in dict_of_lists}
    collated = {k : ops.cat(dict_of_lists[k], axis=0) for k in dict_of_lists}
    return collated

def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)
    while nested_list:
        sublist = nested_list.pop(0)
        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


# NOTE: The units of x, y, z here are assumed to be angstrom
#       I convert to bohr for gau2grid, but the grid remains in angstroms
def gau2grid_density_kdtree(x, y, z, data, ml_y, rs, isoOnlyFlag=0, small=1e-5):
    import numpy as np
    import gau2grid as g2g
    from scipy import spatial

    xyz = np.vstack([x, y, z])
    tree = spatial.cKDTree(xyz.T)

    ml_density = np.zeros_like(x)
    target_density = np.zeros_like(x)

    for coords, full_coeffs, iso_coeffs, ml_coeffs, alpha, norm in zip(
        data['pos_orig'].asnumpy(), data['full_c'].asnumpy(), data['iso_c'].asnumpy(), ml_y.asnumpy(),
        data['exp'].asnumpy(), data['norm'].asnumpy()
    ):
        center = coords
        counter = 0
        for mul, l in rs:
            for j in range(mul):
                normal = norm[counter]
                if normal != 0:
                    exp = [alpha[counter]]

                    angstrom2bohr = 1.8897259886
                    bohr2angstrom = 1 / angstrom2bohr

                    target_full_coeffs = full_coeffs[counter:counter + (2 * l + 1)]

                    pop_ml = ml_coeffs[counter:counter + (2 * l + 1)]
                    c_ml = pop_ml * normal / (2 * np.sqrt(2))
                    if isoOnlyFlag==1:
                        c_ml = 0.0 # ML coeff set to 0.0; only isolated atom used
                    ml_full_coeffs = c_ml + iso_coeffs[counter:counter + (2 * l + 1)]

                    target_max = np.amax(np.abs(target_full_coeffs))
                    ml_max = np.amax(np.abs(ml_full_coeffs))
                    max_c = np.amax(np.array([target_max, ml_max]))

                    cutoff = np.sqrt((-1 / exp[0]) * np.log(small / np.abs(max_c * normal))) * bohr2angstrom

                    close_indices = tree.query_ball_point(center, cutoff)
                    points = np.require(xyz[:, close_indices], requirements=['C', 'A'])

                    ret_target = g2g.collocation(points * angstrom2bohr, l, [1], exp, center * angstrom2bohr)
                    ret_ml = g2g.collocation(points * angstrom2bohr, l, [1], exp, center * angstrom2bohr)

                    # Now permute back to psi4 ordering
                    ##              s     p         d             f                 g
                    psi4_2_e3nn = [[0],[2,0,1],[4,2,0,1,3],[6,4,2,0,1,3,5],[8,6,4,2,0,1,3,5,7]]
                    e3nn_2_psi4 = [[0],[1,2,0],[2,3,1,4,0],[3,4,2,5,1,6,0],[4,5,3,6,2,7,1,8,0]]
                    target_full_coeffs = np.array([target_full_coeffs[k] for k in e3nn_2_psi4[l]])
                    ml_full_coeffs = np.array([ml_full_coeffs[k] for k in e3nn_2_psi4[l]])

                    scaled_components = (target_full_coeffs * normal * ret_target["PHI"].T).T
                    target_tot = np.sum(scaled_components, axis=0)

                    ml_scaled_components = (ml_full_coeffs * normal * ret_target["PHI"].T).T
                    ml_tot = np.sum(ml_scaled_components, axis=0)

                    target_density[close_indices] += target_tot
                    ml_density[close_indices] += ml_tot

                counter += 2 * l + 1

    return target_density, ml_density


def find_min_max(coords):
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    zmin = coords[:, 2].min()
    zmax = coords[:, 2].max()
    return xmin, xmax, ymin, ymax, zmin, zmax


def generate_grid(data, spacing=0.5, buffer=2.0):
    import numpy as np

    buf = buffer
    xmin, xmax, ymin, ymax, zmin, zmax = find_min_max(data['pos_orig'].asnumpy())

    x_points = int((xmax - xmin + 2 * buf) / spacing) + 1
    y_points = int((ymax - ymin + 2 * buf) / spacing) + 1
    z_points = int((zmax - zmin + 2 * buf) / spacing) + 1
    npoints = int((x_points + y_points + z_points) / 3)

    xlin = np.linspace(xmin - buf, xmax + buf, npoints)
    ylin = np.linspace(ymin - buf, ymax + buf, npoints)
    zlin = np.linspace(zmin - buf, zmax + buf, npoints)

    x_spacing = xlin[1] - xlin[0]
    y_spacing = ylin[1] - ylin[0]
    z_spacing = zlin[1] - zlin[0]
    vol = x_spacing * y_spacing * z_spacing

    # 使用 'ij' 索引模式生成网格
    x, y, z = np.meshgrid(xlin, ylin, zlin, indexing='ij')

    return x, y, z, vol, x_spacing, y_spacing, z_spacing


def get_scalar_density_comparisons(data, y_ml, Rs, spacing=0.5, buffer=2.0):
    import numpy as np
    # 生成网格
    x, y, z, vol, x_spacing, y_spacing, z_spacing = generate_grid(data, spacing=spacing, buffer=buffer)
    target_density, ml_density = gau2grid_density_kdtree(x.flatten(),y.flatten(),z.flatten(),data,y_ml,Rs)

    # 将单位转换为 Bohr
    angstrom2bohr = 1.8897259886
    bohr2angstrom = 1/angstrom2bohr

    ep = 100 * np.sum(np.abs(ml_density - target_density)) / np.sum(target_density)

    num_ele_target = np.sum(target_density) * vol * (angstrom2bohr ** 3)
    num_ele_ml = np.sum(ml_density) * vol * (angstrom2bohr ** 3)

    numer = np.sum((ml_density - target_density) ** 2)
    denom = np.sum(ml_density ** 2) + np.sum(target_density ** 2)
    bigI = numer / denom

    return num_ele_target, num_ele_ml, bigI, ep