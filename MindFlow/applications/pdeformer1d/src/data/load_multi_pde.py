# Copyright 2023 Huawei Technologies Co., Ltd
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
r"""Loading custom dataset consisting of multiple PDEs (multi_pde)."""
import os
from typing import Tuple, Dict, Any, List

import numpy as np
from numpy.typing import NDArray
import h5py
from omegaconf import DictConfig
from mindspore.dataset import BatchDataset

from .env import float_dtype, int_dtype
from .pde_dag import PDEAsDAG, ModNodeSwapper
from .utils_multi_pde import dag_info_file_path, get_pde_latex
from .utils_dataload import Dataset, datasets2loader


class CustomPDEformerDataset(Dataset):
    r"""Dataset to be fed into PDEformer."""
    DATA_COLUMN_NAMES = ['node_type', 'node_scalar', 'node_function',
                         'in_degree', 'out_degree', 'attn_bias',
                         'spatial_pos', 'coordinate', 'u_label']

    def __init__(self,
                 config: DictConfig,
                 filename: str,
                 n_samples: int,
                 test: bool = False,
                 deterministic: bool = False) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.test = test
        self.disconn_attn_bias = float(config.data.pde_dag.disconn_attn_bias)
        if deterministic:
            self.data_augment = False
            self.num_tx_samp_pts = -1
        else:
            self.num_tx_samp_pts = config.train.num_tx_samp_pts
            self.data_augment = config.data.augment

        # main data file
        u_filepath = os.path.join(config.data.path, filename + ".hdf5")
        self.h5_file_u = h5py.File(u_filepath, "r")
        self.u_dset = self.h5_file_u["u_sol_all"]

        # turn off data augmentation for non-periodic PDEs
        pde_type_id = np.array(self.h5_file_u["pde_info/pde_type_id"])
        if pde_type_id in [1, 2]:
            periodic = np.array(self.h5_file_u["pde_info/args/periodic"])
        elif pde_type_id == 3:
            periodic = True
        elif pde_type_id == 4:
            # 'node_function' include varying scalars, and the temporal domain is not periodic.
            periodic = False
        self.data_augment = self.data_augment and periodic

        # spatial-temporal coordinates
        x_coord = self.h5_file_u["x_coord"][:]
        t_coord = self.h5_file_u["t_coord"][:]
        x_extend, t_extend = np.meshgrid(x_coord, t_coord)
        # We have t_x_coord[idx_t, idx_x, :] == [t[idx_t], x[idx_x]]
        t_x_coord = np.dstack((t_extend, x_extend))
        self.t_x_coord = t_x_coord.astype(float_dtype)

        # auxiliary data file containing dag_info
        dag_filepath = dag_info_file_path(config, filename)
        if not os.path.exists(dag_filepath):
            raise FileNotFoundError(
                f"The file {dag_filepath} does not exist. Consider running the"
                " following command before training (as shown in"
                " scripts/run_distributed_train.sh): \n\n\t"
                "python3 preprocess_data.py --config_file_path CONFIG_PATH")
        h5_file_dag = h5py.File(dag_filepath, "r")
        self.dag_info = h5_file_dag

        # load data to memory if needed: keep data as NumPy arrays instead of HDF5 objects
        if config.data.load_to_ram:
            if self.test:
                self.u_dset = self.u_dset[-n_samples:].astype(float_dtype)
                # We assume h5_file_dag is generated according to float_dtype
                self.dag_info = {key: h5_file_dag[key][-n_samples:]
                                 for key in h5_file_dag}
            else:
                self.u_dset = self.u_dset[:n_samples].astype(float_dtype)
                self.dag_info = {key: h5_file_dag[key][:n_samples]
                                 for key in h5_file_dag}
            self.h5_file_u.close()
            h5_file_dag.close()

        # prepare to swap INR modulation nodes in the DAG for multi-component PDEs
        self.n_vars = self.u_dset.shape[-1]
        if self.n_vars > 1:
            uf_num_mod = config.model.inr.num_layers - 1
            self.mod_node_swapper = ModNodeSwapper(uf_num_mod, self.n_vars)

    def __getitem__(self, idx_data: int) -> Tuple[NDArray]:
        idx_pde, idx_var = divmod(idx_data, self.n_vars)
        if self.test:
            idx_pde = -idx_pde - 1

        # Shape is [n_t_grid, n_x_grid, n_vars].
        u_label = self.u_dset[idx_pde].astype(float_dtype)
        dag_info = self.dag_info
        # Shape is [n_function_node, n_x_grid, 2].
        node_function = dag_info["node_function_all"][idx_pde]

        # optional parallel transport for periodic BCs
        if self.data_augment:
            roll_x = np.random.randint(u_label.shape[1])
            u_label = np.roll(u_label, roll_x, axis=1)
            node_function_x = node_function[:, :, :-1]
            node_function_fx = np.roll(
                node_function[:, :, -1:], roll_x, axis=1)
            node_function = np.concatenate(
                [node_function_x, node_function_fx], axis=-1)

        # spatial-temporal subsampling
        # Shape is [n_t_grid, n_x_grid, 3].
        t_x_u = np.concatenate([self.t_x_coord, u_label[:, :, [idx_var]]],
                               axis=-1)
        t_x_u = t_x_u.reshape((-1, t_x_u.shape[-1]))
        if self.num_tx_samp_pts > 0:  # subsample spatial-temporal coordinates
            num_tx_pts = t_x_u.shape[0]
            tx_sample_idx = np.random.randint(
                0, num_tx_pts, self.num_tx_samp_pts)
            t_x_u = t_x_u[tx_sample_idx, :]
        coordinate = t_x_u[:, :2]
        u_label = t_x_u[:, 2:]

        # generate spatial_pos
        spatial_pos = dag_info["spatial_pos_all"][idx_pde].astype(int_dtype)
        if idx_var > 0:
            self.mod_node_swapper.apply_(spatial_pos, idx_var)

        # generate attn_bias
        node_type = dag_info["node_type_all"][idx_pde]
        attn_bias = PDEAsDAG.get_attn_bias(
            node_type, spatial_pos, self.disconn_attn_bias)

        input_tuple = (node_type,  # [n_node, 1]
                       dag_info["node_scalar_all"][idx_pde],  # [n_scalar_node, 1]
                       node_function,  # [n_function_node, n_x_grid, 2]
                       dag_info["in_degree_all"][idx_pde],  # [n_node]
                       dag_info["out_degree_all"][idx_pde],  # [n_node]
                       attn_bias,  # [n_node + 1, n_node + 1] if USE_GLOBAL_NODE
                       spatial_pos,  # [n_node, n_node]
                       coordinate,  # [n_t_grid * n_x_grid, 2] if not subsampled
                       u_label,  # [n_t_grid * n_x_grid, 1] if not subsampled
                       )
        return input_tuple

    def __len__(self) -> int:
        return self.n_samples * self.n_vars

    def get_pde_info(self, idx_data: int) -> Dict[str, Any]:
        r"""Get a dictionary containing the information of the current PDE indexed by `idx_data`."""
        idx_pde, idx_var = divmod(idx_data, self.n_vars)
        if self.test:
            idx_pde = -idx_pde - 1

        if repr(self.h5_file_u) == "<Closed HDF5 file>":
            raise NotImplementedError(
                "Method 'get_pde_info' does not support 'load_to_ram'.")
        pde_latex, coef_list = get_pde_latex(self.h5_file_u, idx_pde, idx_var)
        n_t_grid, n_x_grid, _ = self.t_x_coord.shape
        data_info = {"pde_latex": pde_latex, "coef_list": coef_list,
                     "idx_pde": idx_pde, "idx_var": idx_var,
                     "n_t_grid": n_t_grid, "n_x_grid": n_x_grid,
                     "n_tx_pts": n_t_grid * n_x_grid}
        return data_info


def gen_dataloader(config: DictConfig,
                   n_samples: int,
                   datafile_dict: Dict[str, List[str]],
                   batch_size: int,
                   test: bool = False) -> BatchDataset:
    r"""
    Generate the dataloader (`BatchDataset` class object in MindSpore) for
    the training dataset.
    """
    deterministic = False
    shuffle = not deterministic

    # {pde_type: [filename]} -> [filename] -> dataloader
    file_all = []
    for file_list in datafile_dict.values():
        file_all.extend(file_list)

    datasets = [CustomPDEformerDataset(
        config, fname, n_samples, test, deterministic) for fname in file_all]
    return datasets2loader(datasets, batch_size, shuffle,
                           config.data.num_workers, create_iter=False)


def gen_loader_dict(config: DictConfig,
                    n_samples: int,
                    datafile_dict: Dict[str, List[str]],
                    batch_size: int,
                    test: bool = False) -> Dict[str, Dict[str, Tuple]]:
    r"""
    Generate a nested dictionary containing the dataloaders (tuple_iterators in
    MindSpore) for the training or testing datasets.
    """
    deterministic = True
    shuffle = not deterministic

    def dataloader_from_file(filename):
        dataset = CustomPDEformerDataset(
            config, filename, n_samples, test, deterministic)
        dataloader = datasets2loader(
            [dataset], batch_size, shuffle, config.data.num_workers,
            create_iter=True)
        return (dataloader, dataset)

    # {pde_type: [filename]} -> {pde_type: {filename: (dataloader, dataset)}}
    loader_dict = {}
    for pde_type, file_list in datafile_dict.items():
        loader_dict[pde_type] = {filename: dataloader_from_file(filename)
                                 for filename in file_list}
    return loader_dict


def multi_pde_dataset(config: DictConfig) -> Tuple:
    r"""
    Generate dataloaders for the custom multi_pde datasets.

    Args:
        config (DictConfig): Training configurations.

    Returns:
        dataloader_train (BatchDataset): Data loader for the training dataset.
        train_loader_dict (Dict[str, Dict[str, Tuple]]): A nested
            dictionary containing the data iterator instances of the training
            dataset for the evaluation, in which the random operations (data
            augmentation, spatio-temporal subsampling, etc) are disabled. Here,
            to be more specific, 'Tuple' actually refers to
            'Concatenate[TupleIterator, Dataset]'.
        test_loader_dict (Dict[str, Dict[str, Tuple]]): Similar to
            `train_loader_dict`, but for the testing dataset.
        data_info (Dict[str, Any]): A dictionary containing basic information
            about the current dataset.
    """
    num_samples_train = config.data.num_samples_per_file.train
    num_samples_test = config.data.num_samples_per_file.test
    train_file_dict = config.data.multi_pde.train
    if "test" in config.data.multi_pde.keys():
        test_file_dict = config.data.multi_pde.test
    else:
        if not num_samples_train + num_samples_test <= 10000:
            raise ValueError(
                "When the test set is not specified, the sum of "
                f"'num_samples_train' ({num_samples_train}) and "
                f"'num_samples_test' ({num_samples_test}) should not exceed "
                "10000.")
        test_file_dict = train_file_dict

    dataloader_train = gen_dataloader(
        config, num_samples_train, train_file_dict,
        batch_size=config.train.total_batch_size)
    train_loader_dict = gen_loader_dict(
        config, num_samples_train, train_file_dict,
        batch_size=config.test.total_batch_size)
    test_loader_dict = gen_loader_dict(
        config, num_samples_test, test_file_dict,
        batch_size=config.test.total_batch_size, test=True)

    n_t_grid = 101
    data_info = {'n_t_grid': n_t_grid, 'n_x_grid': 256,
                 'n_tx_pts': n_t_grid * 256}
    out_tuple = (dataloader_train, train_loader_dict, test_loader_dict, data_info)
    return out_tuple
