# Copyright 2025 Huawei Technologies Co., Ltd
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
"""create_dataset"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_umap
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from mindchemistry.utils.load_config import load_yaml_config_from_path
from mindchemistry.cell.gemnet.data_utils import get_scaler_from_data_list
from mindchemistry.cell.gemnet.data_utils import lattice_params_to_matrix
from mindchemistry.cell.dimenet.preprocess import PreProcess
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class CreateDataset:
    """Create Dataset for crystal structures

    Args:
        name (str): Name of the dataset
        path (str): Path to the dataset
        prop (str): Property to predict
        niggli (bool): Whether to convert to Niggli reduced cell
        primitive (bool): Whether to convert to primitive cell
        graph_method (str): Method to create graph
        preprocess_workers (int): Number of workers for preprocessing
        lattice_scale_method (str): Method to scale lattice
        num_samples (int): Number of samples to use, if None use all
    """

    def __init__(self, name, path,
                 prop, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, config_path,
                 num_samples=None):
        super().__init__()
        self.path = path
        self.name = name
        self.num_samples = num_samples
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.config = load_yaml_config_from_path(config_path).get("Encoder")
        self.preprocess = PreProcess(
            num_spherical=self.config.get("num_spherical"),
            num_radial=self.config.get("num_radial"),
            envelope_exponent=self.config.get("envelope_exponent"),
            otf_graph=False,
            cutoff=self.config.get("cutoff"),
            max_num_neighbors=self.config.get("max_num_neighbors"),)

        self.cached_data = data_preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            num_samples=self.num_samples
        )[:self.num_samples]
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        data = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data["graph_arrays"]
        data_res = self.preprocess.data_process(angles.reshape(1, -1), lengths.reshape(1, -1),
                                                np.array([num_atoms]), edge_indices.T, frac_coords,
                                                edge_indices.shape[0], to_jimages, atom_types, prop)
        return data_res

    def __repr__(self):
        return f"CrystDataset({self.name}, {self.path})"

    def get_dataset_size(self):
        return len(self.cached_data)


# match element with its chemical symbols
chemical_symbols = [
    # 0
    "X",
    # 1
    "H", "He",
    # 2
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    # 3
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    # 4
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    # 5
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    # 6
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn",
    # 7
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og"
]

# used for crystal matching
CRYSTALNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt="cif")

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
    # match is gaurantteed because cif only uses lattice params & frac_coords
    assert canonical_crystal.matches(crystal)
    return canonical_crystal


def build_crystal_graph(crystal, graph_method="crystalnn"):
    """build crystal graph"""

    if graph_method == "crystalnn":
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CRYSTALNN)
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms


def save_data(dataset, is_train, dataset_name):
    """save created dataset to npy"""
    processed_data = dict()
    data_parameters = ["atom_types", "dist", "angle", "idx_kj", "idx_ji",
                       "edge_j", "edge_i", "pos", "batch", "lengths",
                       "num_atoms", "angles", "frac_coords",
                       "num_bonds", "num_triplets", "sbf", "y"]
    for j, name in enumerate(data_parameters):
        if j == 16:
            # Here, y is mindspore.Tensor, while others are all numpy.array, so need to change the type first.
            processed_data[name] = [i[j].astype(np.float32) for i in dataset]
        elif j == 14:
            # Here, we need the sum of num_triplets, so get the summary before we save it.
            processed_data[name] = [i[j].sum() for i in dataset]
        else:
            processed_data[name] = [i[j] for i in dataset]

    if not os.path.exists(f"./data/{dataset_name}/{is_train}"):
        os.makedirs(f"./data/{dataset_name}/{is_train}")
        logging.info("%s has been created",
                     f"./data/{dataset_name}/{is_train}")
    if is_train == "train":
        np.savetxt(f"./data/{dataset_name}/{is_train}/scaler_mean.csv",
                   dataset.scaler.means.reshape(-1))
        np.savetxt(f"./data/{dataset_name}/{is_train}/scaler_std.csv",
                   dataset.scaler.stds.reshape(-1))
        np.savetxt(
            f"./data/{dataset_name}/{is_train}/lattice_scaler_mean.csv", dataset.lattice_scaler.means)
        np.savetxt(
            f"./data/{dataset_name}/{is_train}/lattice_scaler_std.csv", dataset.lattice_scaler.stds)
    np.save(
        f"./data/{dataset_name}/{is_train}/processed_data.npy", processed_data)


def process_one(row, niggli, primitive, graph_method, prop_list):
    """process one one sample"""
    crystal_str = row["cif"]
    crystal = build_crystal(
        crystal_str, niggli=niggli, primitive=primitive)
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: row[k] for k in prop_list if k in row.keys()}
    result_dict = {
        "mp_id": row["material_id"],
        "cif": crystal_str,
        "graph_arrays": graph_arrays,
    }
    result_dict.update(properties)
    return result_dict


def data_preprocess(input_file, num_workers, niggli, primitive, graph_method, prop_list, num_samples):
    """process data"""
    df = pd.read_csv(input_file)[:num_samples]

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        num_cpus=num_workers)

    mpid_to_results = {result["mp_id"]: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]["material_id"]]
                       for idx in range(len(df))]

    return ordered_results


def add_scaled_lattice_prop(data_list, lattice_scale_method):
    """add scaled lattice prop to dataset"""
    for data in data_list:
        graph_arrays = data["graph_arrays"]
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == "scale_length":
            lengths = lengths / float(num_atoms)**(1 / 3)

        data["scaled_lattice"] = np.concatenate([lengths, angles])


def create_dataset(args):
    """create dataset"""
    config_data_path = f"./conf/data/{args.dataset}.yaml"
    config_path = f"./conf/configs.yaml"
    config_data = load_yaml_config_from_path(config_data_path)
    prop = config_data.get("prop")
    niggli = config_data.get("niggli")
    primitive = config_data.get("primitive")
    graph_method = config_data.get("graph_method")
    lattice_scale_method = config_data.get("lattice_scale_method")
    preprocess_workers = config_data.get("preprocess_workers")
    path_train = f"./data/{args.dataset}/train.csv"
    train_dataset = CreateDataset("Formation energy train", path_train, prop,
                                  niggli, primitive, graph_method,
                                  preprocess_workers, lattice_scale_method,
                                  config_path, args.num_samples_train)
    lattice_scaler = get_scaler_from_data_list(
        train_dataset.cached_data,
        key="scaled_lattice")
    scaler = get_scaler_from_data_list(
        train_dataset.cached_data,
        key=train_dataset.prop)
    train_dataset.lattice_scaler = lattice_scaler
    train_dataset.scaler = scaler
    save_data(train_dataset, "train", args.dataset)

    path_val = f"./data/{args.dataset}/val.csv"
    val_dataset = CreateDataset("Formation energy val", path_val, prop,
                                niggli, primitive, graph_method,
                                preprocess_workers, lattice_scale_method, args.num_samples_val)
    val_dataset.lattice_scaler = lattice_scaler
    val_dataset.scaler = scaler
    save_data(val_dataset, "val", args.dataset)

    path_test = f"./data/{args.dataset}/test.csv"
    test_dataset = CreateDataset("Formation energy test", path_test, prop,
                                 niggli, primitive, graph_method,
                                 preprocess_workers, lattice_scale_method,
                                 args.num_samples_test)
    test_dataset.lattice_scaler = lattice_scaler
    test_dataset.scaler = scaler
    save_data(test_dataset, "test", args.dataset)


def main(args):
    create_dataset(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="perov_5")
    parser.add_argument("--num_samples_train", default=300, type=int)
    parser.add_argument("--num_samples_val", default=300, type=int)
    parser.add_argument("--num_samples_test", default=300, type=int)
    main_args = parser.parse_args()
    main(main_args)
