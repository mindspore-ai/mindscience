# Copyright 2024 Huawei Technologies Co., Ltd
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
"""data utils file"""
import numpy as np
import pandas as pd
import scipy

import mindspore as ms

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from p_tqdm import p_umap

# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions
# Note that differing from OCP, we have 27 offsets here because we are in 3D
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    r"""Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, _ = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def lattice_polar_decompose(lattice: np.ndarray):
    """decompose the lattice to lattice_polar"""
    assert lattice.ndim == 2
    a, u = np.linalg.eigh(lattice @ lattice.T)
    a, u = np.real(a), np.real(u)
    a = np.diag(np.log(a)) / 2
    s = u @ a @ u.T

    k = np.array(
        [
            s[0, 1],
            s[0, 2],
            s[1, 2],
            (s[0, 0] - s[1, 1]) / 2,
            (s[0, 0] + s[1, 1] - 2 * s[2, 2]) / 6,
            (s[0, 0] + s[1, 1] + s[2, 2]) / 3,
        ]
    )
    return k

def lattice_polar_build(k: np.ndarray):
    """build lattice using lattice_polar"""
    assert k.ndim == 1
    s = np.array(
        [
            [k[3] + k[4] + k[5], k[0], k[1]],
            [k[0], -k[3] + k[4] + k[5], k[2]],
            [k[1], k[2], -2 * k[4] + k[5]],
        ]
    )  # (3, 3)
    exp_s = scipy.linalg.expm(s)  # (3, 3)
    return exp_s

class StandardScalerMS:
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = ms.Tensor(means, dtype=ms.float32)
        self.stds = ms.Tensor(stds, dtype=ms.float32)

    def transform(self, x):
        x = ms.Tensor(x, dtype=ms.float32)
        return (x - self.means) / self.stds

    def inverse_transform(self, x):
        x = ms.Tensor(x, dtype=ms.float32)
        return x * self.stds + self.means

    def copy(self):
        return StandardScalerMS(
            means=self.means.copy(),
            stds=self.stds.copy())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means}, "
            f"stds: {self.stds})"
        )

crystalnn = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_crystal

def build_crystal_graph(crystal, graph_method='crystalnn'):
    """build crystal graph especially for edge data from Structure of Pymatgen.
    Convert them to numpy arrays."""
    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, crystalnn)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]
    lattice_polar = lattice_polar_decompose(crystal.lattice.matrix)

    edge_indices, to_jimages = [], []

    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, lattice_polar, edge_indices, to_jimages, num_atoms

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)

def lattice_matrix_to_params(matrix):
    """Converts matrix to lattice from abc, angles.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    lengths = np.sqrt(np.sum(matrix ** 2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) /
                            (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma

def preprocess(input_file, num_workers, niggli, primitive, graph_method,
               prop_list, nrows=-1):
    """
    Read crystal data from a dataset CSV file and preprocess it

    Args:
        input_file (str): The path of dataset csv.
        num_workers (int): The numbers of cpus used for preprocessing the crystals.
        niggli (bool): Whether to use niggli algorithom to preprocess the choice of lattice.
        primitive (bool): Whether to represent the crystal in primitive cell.
        graph_method (str): If 'crystalnn', construct the graph by crystalnn algorithm,
            mainly effect the construct of edges. If 'none', don't construct any edge.
        prop_list (list[str]): Read the property of crystal as specified by the element of the list.
        nrows (int): If nrows > 0, read the first 'nrows' lines of csv file. If nrows = -1, read the whole csv file.
            This arg is mainly for debugging to quickly load a few crystals.

    Returns:
        List. Return the list of crystals. Each element is a Dict composed by:
        {
            'mp_id': int,
            'cif': crystal string,
            'graph_arrays': numpy arrays of frac_coords, atom_types,
                lengths, angles, edge_indices, to_jimages, num_atoms,
        }
    """
    if nrows == -1:
        df = pd.read_csv(input_file)
    # for debug
    else:
        df = pd.read_csv(input_file, nrows=nrows)

    def process_one(row, niggli, primitive, graph_method, prop_list):
        crystal_str = row['cif']

        crystal = build_crystal(
            crystal_str, niggli=niggli, primitive=primitive)
        # 晶体构建图，重点
        graph_arrays = build_crystal_graph(crystal, graph_method)

        properties = {k: row[k] for k in prop_list if k in row.keys()}
        result_dict = {
            'mp_id': row['material_id'],
            'cif': crystal_str,
            'graph_arrays': graph_arrays,
        }
        result_dict.update(properties)
        return result_dict

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        num_cpus=num_workers)

    mpid_to_results = {result['mp_id']: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
                       for idx in range(len(df))]
    return ordered_results

class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, x):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`x`.
        :param x: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        x = np.array(x).astype(float)
        self.means = np.nanmean(x, axis=0)
        self.stds = np.nanstd(x, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means).astype(float)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds).astype(float)

        return self

    def transform(self, x):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param x: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        x = np.array(x).astype(float)
        transformed_with_nan = (x - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan).astype(float)

        return transformed_with_none

    def inverse_transform(self, x):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param x: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        x = np.array(x).astype(float)
        transformed_with_nan = x * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan).astype(float)

        return transformed_with_none
