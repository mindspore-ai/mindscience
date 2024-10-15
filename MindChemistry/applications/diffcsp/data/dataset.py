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
"""dataset file"""
import os
from pathlib import Path

import numpy as np

from data.data_utils import StandardScaler, preprocess

if Path('./dataset/dataset_prop.txt').exists():
    with open('./dataset/dataset_prop.txt', 'r') as file:
        data = file.read()
    # pylint: disable=W0123
    scalar_dict = eval(data)
else:
    scalar_dict = {}

def fullconnect_dataset(name,
                        path,
                        niggli=True,
                        primitive=False,
                        graph_method='none',
                        preprocess_workers=30,
                        save_path='',
                        nrows=-1):
    """
    Read crystal data from a CSV file and convert each into a fully connected graph,
    where the nodes represent atoms within the unit cell and the edges connect every pair of nodes.

    Args:
        name (str): The name of dataset, mainly used to read the dataset
            property in './dataset/dataset_prop.txt'.
            It doesn't matter for crystal structure prediction task.
            Choices: [perov_5, carbon_24, mp_20, mpts_52].
            Users can also create custom datasets, by modify the
            './dataset/dataset_prop.txt'.
        path (str): The path of csv file of dataset.
        niggli (bool): Whether to use niggli algorithom to
            preprocess the choice of lattice. Default:
            ``True``.
        primitive (bool): Whether to represent the crystal in primitive cell. Default:
            ``False``.
        graph_method (str): If 'crystalnn', construct the graph by crystalnn
            algorithm, mainly effect the construct of edges.
            If 'none', don't construct any edge.  Default: ``none``.
        preprocess_workers (int): The numbers of cpus used for
            preprocessing the crystals. Default: ``None``.
        save_path (str): The path for saving the preprocessed data,
            aiming to load the dataset more quickly next time.
        nrows (int): If nrows > 0, read the first 'nrows' lines of csv file.
            If nrows = -1, read the whole csv file.
            This arg is mainly for debugging to quickly load a few crystals.
    Returns:
        x (list): List of Atom types. Shape of each element i.e. numpy array: (num_atoms, 1)
        frac_coord_list (list): List of Fractional Coordinates of atoms.
            Shape of each element i.e. numpy array: (num_atoms, 3)
        edge_attr (list): List of numpy arrays filled with ones,
            just used to better construct the dataloader,
            without numerical significance. Shape of each element
            i.e. numpy array: (num_edges, 3)
        edge_index (list): List of index of the beginning and end
            of edges. Each element is composed as [src, dst], where
            src and dst is numpy arrays with Shape (num_edges,).
        lengths_list (list): List of lengths of lattice. Shape of
            each element i.e. numpy array: (3,)
        angles_list (list): List of angles of lattice. Shape of
            each element i.e. numpy array: (3,)
        labels (list): List of property of crystal. Shape of
            each element i.e. numpy array: (1,)
    """
    x = []
    frac_coord_list = []
    edge_index = []
    edge_attr = []
    labels = []
    lengths_list = []
    angles_list = []

    if name in scalar_dict.keys():
        prop = scalar_dict[name]['prop']
        scaler = StandardScaler(scalar_dict[name]['scaler.means'],
                                scalar_dict[name]['scaler.stds'])
    else:
        print('No dataset property is specified, so no property reading is performed')
        prop = "None"
        scaler = None

    if os.path.exists(save_path):
        cached_data = np.load(save_path, allow_pickle=True)
    else:
        cached_data = preprocess(path,
                                 preprocess_workers,
                                 niggli=niggli,
                                 primitive=primitive,
                                 graph_method=graph_method,
                                 prop_list=[prop],
                                 nrows=nrows)

        np.save(save_path, cached_data)

    for idx in range(len(cached_data)):
        data_dict = cached_data[idx]
        (frac_coords, atom_types, lengths, angles, _, _,
         num_atoms) = data_dict['graph_arrays']

        indices = np.arange(num_atoms)
        dst, src = np.meshgrid(indices, indices)
        src = src.reshape(-1)
        dst = dst.reshape(-1)

        x.append(atom_types.reshape(-1, 1))
        frac_coord_list.append(frac_coords)
        edge_index.append(np.array([src, dst]))
        edge_attr.append(np.ones((num_atoms * num_atoms, 3)))
        lengths_list.append(lengths)
        angles_list.append(angles)
        if scaler is not None:
            labels.append(scaler.transform(data_dict[prop]))
        else:
            labels.append(0.0)

    return x, frac_coord_list, edge_attr, edge_index, lengths_list, angles_list, labels
