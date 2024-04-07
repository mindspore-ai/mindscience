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
r"""Loading datasets containing one specific PDE (single_pde), mainly PDEBench datasets."""
import os
from typing import Tuple, Union, List, Dict, Any

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from omegaconf import DictConfig

from .env import float_dtype
from .pde_dag import PDENodesCollector
from .utils_dataload import Dataset, datasets2loader, concat_datasets
from .load_multi_pde import CustomPDEformerDataset


class PostProcessHDF5Loader:
    r"""
    Wrapped loader of an HDF5 object. The PDE solution data `u` will be loaded with a specific stride
    in the `x` direction, and then interpolated in the `t` direction.
    """

    def __init__(self,
                 h5_object,
                 t_orig: float,
                 t_new: float,
                 x_stride: int = 1,
                 newaxis: bool = False) -> None:
        self.h5_object = h5_object
        self.t_orig = t_orig
        self.t_new = t_new
        self.x_stride = x_stride
        self.newaxis = newaxis

    def __getitem__(self, idx_pde: int) -> NDArray[float]:
        u_label = self.h5_object[idx_pde, :, ::self.x_stride]
        if self.newaxis:
            u_label = np.expand_dims(u_label, axis=-1)
        interp = RegularGridInterpolator((self.t_orig,), u_label)
        # Note that for scipy 1.10.1, 'interp' would change dtype to float64,
        # so we have to change it back to the default float32.
        u_interp = interp(self.t_new).astype(float_dtype)
        return u_interp


class SinglePDEDataset(Dataset):
    r"""Base class for loading the single_pde data."""

    def __init__(self,
                 config: DictConfig,
                 pde_type: str,
                 pde_param: Union[float, List[float]],
                 n_samples: int,
                 test: bool = False,
                 deterministic: bool = False) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.test = test
        self.input_pde_param = config.data.single_pde.get(
            "input_pde_param", False)
        if deterministic:
            self.data_augment = False
            self.num_tx_samp_pts = -1
        else:
            self.data_augment = config.data.augment
            self.num_tx_samp_pts = config.train.num_tx_samp_pts

        # filename and pde_param
        data_info = self._pde_type_info(pde_type, pde_param)
        if 'coef_list' not in data_info:
            raise KeyError(f"'coef_list' not found in data_info for {pde_type}.")
        pde_param_list = [value for (_, value) in data_info["coef_list"]]
        self.pde_param = np.array(pde_param_list, dtype=float_dtype)

        # main data file
        if 'filename' not in data_info:
            raise KeyError(f"'filename' not found in data_info for {pde_type}.")
        filepath = os.path.join(config.data.path, data_info["filename"])
        self.h5_file_u = h5py.File(filepath + ".hdf5", "r")

        # spatial-temporal coordinates
        t_coord_orig, t_coord, x_stride, t_x_coord = self._gen_t_x_coord(
            pde_type, self.h5_file_u)
        n_t_grid, n_x_grid, _ = t_x_coord.shape
        data_info.update({"n_t_grid": n_t_grid, "n_x_grid": n_x_grid,
                          "n_tx_pts": n_t_grid * n_x_grid})
        self.t_x_coord = t_x_coord
        self.data_info = data_info

        # post-processed PDE solution
        if pde_type in ["burgers_nu2", "adv_beta", "reacdiff_nu_rho"]:
            u_dset_h5 = self.h5_file_u["tensor"]  # PDEBench
            newaxis = True
        else:
            u_dset_h5 = self.h5_file_u["u_sol_all"]  # custom dataset
            newaxis = False
        self.u_dset = PostProcessHDF5Loader(
            u_dset_h5, t_coord_orig, t_coord, x_stride, newaxis)

        # keep data as NumPy arrays instead of hdf5 objects
        if config.data.load_to_ram:
            if self.test:
                u_dset = u_dset_h5[-n_samples:, :, ::x_stride]
            else:
                u_dset = u_dset_h5[:n_samples, :, ::x_stride]
            self.h5_file_u.close()
            if newaxis:
                u_dset = np.expand_dims(u_dset, axis=-1)
            # [n_pde, n_t_grid_old, n_x_grid, 1] -> [n_t_grid_old, n_pde, n_x_grid, 1]
            u_dset = u_dset.transpose((1, 0, 2, 3))
            interp = RegularGridInterpolator((t_coord_orig,), u_dset)
            # [n_t_grid, n_pde, n_x_grid, 1] -> [n_pde, n_t_grid, n_x_grid, 1]
            u_dset = interp(t_coord).transpose((1, 0, 2, 3))
            self.u_dset = u_dset.astype(float_dtype)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx) -> None:
        pass

    @staticmethod
    def _pde_type_info(pde_type: str,
                       pde_param: Union[float, List[float]]) -> Dict[str, Any]:
        r"""Generate information of the current PDE."""
        # pre-training custom_v4, nu: [5e-4, 5], cU5
        if pde_type == "burgers_nu2":
            nu_value = pde_param
            filename = f"1D_Burgers_Sols_Nu{nu_value}"
            # 2.0 rather than 0.5: time is rescaled (x2), flux computing issue
            # in the PDEBench data generation code till Oct 2023 (x2)
            pde_latex = r"$u_t+(2u^2-\kappa u_x)_x=0$"
            # need to be multiplied by 2 because time is rescaled
            coef_list = [(r"\kappa", 2 * nu_value / np.pi)]
        elif pde_type == "adv_beta":
            beta_value = pde_param
            filename = f"1D_Advection_Sols_beta{beta_value}"
            pde_latex = r"$u_t+(\beta u)_x=0$"
            # need to be multiplied by 4 due to rescaled time and space
            coef_list = [(r"\beta", 4 * beta_value)]
        elif pde_type == "reacdiff_nu_rho":
            nu_value, rho_value = pde_param
            filename = f"ReacDiff_Nu{nu_value}_Rho{rho_value}"
            pde_latex = r"$u_t-\rho u+\rho u^2-\nu u_{xx}=0$"
            # need to be multiplied by 4 due to rescaled space
            coef_list = [(r"\nu", 4 * nu_value), (r"\rho", rho_value)]
        elif pde_type == "cosFlux_nu":
            nu_value = pde_param
            filename = f"custom_cosFlux_nu{nu_value}_seed1"
            pde_latex = r"$u_t+(\cos(u)-\nu u_x)_x=0$"
            coef_list = [(r"\nu", nu_value)]
        else:
            raise NotImplementedError
        data_info = {"pde_latex": pde_latex,
                     "coef_list": coef_list, "filename": filename}
        return data_info

    @staticmethod
    def _gen_t_x_coord(pde_type: str, h5_file_u) -> Tuple:
        r"""Generate the spatial-temporal coordinates."""
        if pde_type in ["burgers_nu2", "adv_beta", "reacdiff_nu_rho"]:
            # subsample PDEBench x-coordinates from 1024 to 256
            x_stride = 4
            x_coord = h5_file_u["x-coordinate"][::x_stride]
            # The last time step stored in t-coordinate is not used.
            t_coord_orig = h5_file_u["t-coordinate"][:-1]
            # spatio-temporal rescaling to fit t\in[0,1], x\in[-1,1]
            if pde_type in ["burgers_nu2", "adv_beta"]:
                t_coord_orig = 0.5 * t_coord_orig
            if pde_type in ["adv_beta", "reacdiff_nu_rho"]:
                x_coord = 2 * x_coord - 1
        else:  # custom dataset
            x_stride = 1
            x_coord = h5_file_u["x_coord"][:]
            # for custom dataset, the t-coordinate is not of very high
            # precision, and the last time-step is a bit smaller than 1
            # if float64 is used
            t_coord_orig = h5_file_u["t_coord"][:].astype(np.float32)

        t_coord_orig = t_coord_orig.astype(float_dtype)
        x_coord = x_coord.astype(float_dtype)
        t_coord = np.linspace(0, 1, x_coord.shape[0], dtype=float_dtype)
        x_extend, t_extend = np.meshgrid(x_coord, t_coord)
        # We have t_x_coord[idx_t, idx_x, :] == [t[idx_t], x[idx_x]]
        t_x_coord = np.dstack((t_extend, x_extend))

        coords = (t_coord_orig, t_coord, x_stride, t_x_coord)
        return coords

    def get_pde_info(self, _) -> Dict[str, Any]:
        r"""
        Get a dictionary containing the information of the current PDE. As the data samples in the
        single_pde dataset differ only in initial onditions, the input `idx_pde` will not be used.
        """
        return self.data_info


class HWCSinglePDEDataset(SinglePDEDataset):
    r"""Loading single_pde dataset in the HWC format. Used by FNO."""
    DATA_COLUMN_NAMES = ["grid_in", "u_label"]
    CHW = False

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        if self.test:
            idx_pde = -idx_pde - 1

        # generate t_x_u
        u_label = self.u_dset[idx_pde]
        n_t_grid, n_x_grid, _ = u_label.shape
        if self.data_augment:
            roll_x = np.random.randint(n_x_grid)
            u_label = np.roll(u_label, roll_x, axis=1)

        init_cond = u_label[0, :, :]  # [n_x_grid, 1]
        if self.input_pde_param:
            # [n_param] -> [n_x_grid, n_param]
            tiled_param = np.tile(self.pde_param, (n_x_grid, 1))
            # Shape is [n_x_grid, 1 + n_param].
            grid_in = np.concatenate((init_cond, tiled_param), axis=1)
        else:
            grid_in = init_cond
        # Shape is [n_t_grid, n_x_grid, 1 (+ n_param)].
        grid_in = np.tile(grid_in, (n_t_grid, 1, 1))
        if self.CHW:
            # Shape is [1 (+ n_param), n_t_grid, n_x_grid].
            grid_in = grid_in.transpose((2, 0, 1))
            u_label = u_label.transpose((2, 0, 1))  # [1, n_t_grid, n_x_grid]
        return grid_in, u_label


class CHWSinglePDEDataset(HWCSinglePDEDataset):
    r"""Loading single_pde dataset in the CHW format. Used by U-Net."""
    CHW = True


class INRSinglePDEDataset(SinglePDEDataset):
    r"""Base class for loading the single_pde data for INRs."""
    DATA_COLUMN_NAMES = ["init_cond", "coordinate", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        if self.test:
            idx_pde = -idx_pde - 1

        u_label = self.u_dset[idx_pde]  # [n_t_grid, n_x_grid, n_vars]
        # optional parallel transport for periodic BCs
        if self.data_augment:
            roll_x = np.random.randint(u_label.shape[1])
            u_label = np.roll(u_label, roll_x, axis=1)
        init_cond = u_label[0, :, 0]

        # spatial-temporal subsampling
        # Shape is [n_t_grid, n_x_grid, 3].
        t_x_u = np.concatenate([self.t_x_coord, u_label], axis=-1)
        t_x_u = t_x_u.reshape((-1, t_x_u.shape[-1]))
        if self.num_tx_samp_pts > 0:
            num_tx_pts = t_x_u.shape[0]
            tx_sample_idx = np.random.randint(
                0, num_tx_pts, self.num_tx_samp_pts)
            t_x_u = t_x_u[tx_sample_idx, :]
        coordinate = t_x_u[:, :2]  # [n_tx_pts, 2]
        u_label = t_x_u[:, 2:]  # [n_tx_pts, 1]

        return init_cond, coordinate, u_label


class DeepONetSinglePDEDataset(INRSinglePDEDataset):
    r"""Loading single_pde dataset for DeepONet."""
    DATA_COLUMN_NAMES = ["trunk_in", "branch_in", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        init_cond, coordinate, u_label = super().__getitem__(idx_pde)
        trunk_in = coordinate
        if self.input_pde_param:
            # Shape is [n_param + n_x_grid].
            branch_in = np.concatenate([self.pde_param, init_cond])
        else:
            branch_in = init_cond  # [n_x_grid]
        return trunk_in, branch_in, u_label


class PDEformerSinglePDEDataset(INRSinglePDEDataset):
    r"""Loading single_pde dataset for PDEformer."""
    DATA_COLUMN_NAMES = ['node_type', 'node_scalar', 'node_function',
                         'in_degree', 'out_degree', 'attn_bias',
                         'spatial_pos', 'coordinate', 'u_label']

    def __init__(self,
                 config: DictConfig,
                 pde_type: str,
                 pde_param: Union[float, List[float]],
                 n_samples: int,
                 test: bool = False,
                 deterministic: bool = False) -> None:
        super().__init__(config, pde_type, pde_param, n_samples, test, deterministic)
        x_coord = self.t_x_coord[0, :, 1]
        pde = self._gen_pde_nodes(pde_type, self.pde_param, x_coord)
        self.pde_dag = pde.gen_dag(config)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        init_cond, coordinate, u_label = super().__getitem__(idx_pde)

        # node_function
        # Shape is [n_function_nodes, n_x_grid, 2].
        node_function = np.copy(self.pde_dag.node_function)
        node_function[0, :, 1] = init_cond  # [n_x_grid]

        input_tuple = (self.pde_dag.node_type,  # [n_node, 1]
                       self.pde_dag.node_scalar,  # [n_scalar_node, 1]
                       node_function,  # [n_function_node, n_x_grid, 2]
                       self.pde_dag.in_degree,  # [n_node]
                       self.pde_dag.out_degree,  # [n_node]
                       self.pde_dag.attn_bias,  # [n_node, n_node]
                       self.pde_dag.spatial_pos,  # [n_node, n_node]
                       coordinate,  # [n_t_grid * n_x_grid, 2] if not subsampled
                       u_label,  # [n_t_grid * n_x_grid, 1] if not subsampled
                       )
        return input_tuple

    @staticmethod
    def _gen_pde_nodes(pde_type: str,
                       pde_param_list: List[float],
                       x_coord: NDArray[float]) -> PDENodesCollector:
        r"""Generate the PDE nodes."""
        pde = PDENodesCollector()
        u_node = pde.new_uf()
        ic_field = np.zeros_like(x_coord) + np.nan
        pde.set_ic(u_node, x_coord, ic_field)
        if pde_type == "burgers_nu2":
            kappa = pde.new_coef(pde_param_list[0])
            # 2 rather than 0.5: time is rescaled (x2), flux computing issue
            # in the PDEBench data generation code till Oct 2023 (x2)
            flux = 2 * pde.square(u_node) - kappa * pde.dx(u_node)
            pde.sum_eq0(pde.dt(u_node), pde.dx(flux))
        elif pde_type == "adv_beta":
            beta = pde.new_coef(pde_param_list[0])
            pde.sum_eq0(pde.dt(u_node), pde.dx(beta * u_node))
        elif pde_type == "reacdiff_nu_rho":
            nu_node = pde.new_coef(pde_param_list[0])
            rho = pde.new_coef(pde_param_list[1])
            mrho = pde.new_coef(-pde_param_list[1])
            pde.sum_eq0(pde.dt(u_node), pde.dx(-(nu_node * pde.dx(u_node))),
                        mrho * u_node, rho * pde.square(u_node))
        elif pde_type == "cosFlux_nu":
            nu_node = pde.new_coef(pde_param_list[0])
            f1u = pde.new_coef(1.0) * pde.cos(pde.new_coef(1.0) * u_node)
            pde.sum_eq0(u_node.dt, pde.dx(f1u - nu_node * u_node.dx))
        return pde


def get_dataset(config: DictConfig,
                pde_type: str,
                pde_param: Union[float, List[float]],
                n_samples: int,
                test: bool,
                deterministic: bool) -> Dataset:
    r"""Obtain single_pde dataset for the current network model."""
    model_type = config.model_type.lower()
    if model_type in ["pdeformer", "pf", "puser"]:
        data_cls = PDEformerSinglePDEDataset
    elif model_type == "deeponet":
        data_cls = DeepONetSinglePDEDataset
    elif model_type == "fno":
        data_cls = HWCSinglePDEDataset
    elif model_type in ["unet", "u-net"]:
        data_cls = CHWSinglePDEDataset
    else:
        raise NotImplementedError(f"unknown model_type: {model_type}")

    return data_cls(config, pde_type, pde_param, n_samples, test, deterministic)


def gen_loader_dict(config: DictConfig,
                    n_samples: int,
                    pde_param_list: Union[List[float], List[List[float]]],
                    batch_size: int,
                    test: bool = False) -> Dict[str, Dict[str, Tuple]]:
    r"""
    Generate a dictionary containing the dataloaders (`BatchDataset` class
    objects in MindSpore) for the training or testing datasets.
    """
    deterministic = True
    shuffle = not deterministic
    pde_type = config.data.single_pde.param_name

    def dataloader_from_param(pde_param):
        dataset = get_dataset(config, pde_type, pde_param,
                              n_samples, test, deterministic)
        dataloader = datasets2loader(
            [dataset], batch_size, shuffle, config.data.num_workers,
            create_iter=True)
        return (dataloader, dataset)

    param_loader_dict = {pde_param: dataloader_from_param(pde_param)
                         for pde_param in pde_param_list}
    return {pde_type: param_loader_dict}


class RegularizedFineTuneDataset(Dataset):
    r"""
    To avoid overfitting when fine-tuning PDEformer on small datasets, we
    include the pre-training (multi_pde) dataset during the fine-tuning stage
    as a regularization.
    Each sample in `dataset`, with probability `regularize_ratio`, is replaced
    by a randomly selected sample from `regularize_dataset`.
    """
    DATA_COLUMN_NAMES = ['node_type', 'node_scalar', 'node_function',
                         'in_degree', 'out_degree', 'attn_bias',
                         'spatial_pos', 'coordinate', 'u_label']

    def __init__(self,
                 datasets: List[Dataset],
                 regularize_datasets: List[Dataset],
                 regularize_ratio: float) -> None:
        self.dataset = concat_datasets(datasets)
        self.regularize_dataset = concat_datasets(regularize_datasets)
        self.regularize_ratio = regularize_ratio
        self.num_regularize_data = len(self.regularize_dataset)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        if np.random.rand() < self.regularize_ratio:
            idx_reg = np.random.randint(self.num_regularize_data)
            return self.regularize_dataset[idx_reg]
        return self.dataset[idx_pde]

    def __len__(self) -> int:
        return len(self.dataset)


def single_pde_dataset(config: DictConfig) -> Tuple:
    r"""
    Generate dataloaders for the single_pde datasets.

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
    if not num_samples_train + num_samples_test <= 10000:
        raise ValueError(
            "For PDEBench dataset, the sum of 'num_samples_train' "
            f"({num_samples_train}) and 'num_samples_test' ({num_samples_test})"
            " should not exceed 10000.")

    train_datasets = [get_dataset(
        config, config.data.single_pde.param_name, pde_param,
        num_samples_train, test=False, deterministic=False,
    ) for pde_param in config.data.single_pde.train]
    regularize_ratio = config.data.single_pde.get("regularize_ratio", 0.)
    if regularize_ratio > 0 and config.model_type == "pdeformer":
        # include regularization dataset (custom multi_pde utilized in
        # pre-training) during the fine-tuning stage
        if not config.train.num_tx_samp_pts > 0:
            raise ValueError("When 'regularize_ratio' is positive, "
                             "'num_tx_samp_pts' should be positive as well.")
        regularize_filenames = []
        for file_list in config.data.multi_pde.train.values():
            # train dict form: {pde_type1: [file1, ..], pde_type2: [files, ..]}
            regularize_filenames.extend(file_list)
        regularize_datasets = [CustomPDEformerDataset(
            config, fname, config.data.num_samples_per_file.regularize,
            test=False, deterministic=False,
        ) for fname in regularize_filenames]
        finetune_dataset = RegularizedFineTuneDataset(
            train_datasets, regularize_datasets, regularize_ratio)
        train_datasets = [finetune_dataset]
    dataloader_train = datasets2loader(
        train_datasets, config.train.total_batch_size, True,
        config.data.num_workers, create_iter=False)
    train_loader_dict = gen_loader_dict(
        config, num_samples_train, config.data.single_pde.train,
        batch_size=config.test.total_batch_size)
    test_loader_dict = gen_loader_dict(
        config, num_samples_test, config.data.single_pde.test,
        batch_size=config.test.total_batch_size, test=True)

    n_t_grid = 256
    data_info = {'n_t_grid': n_t_grid, 'n_x_grid': 256,
                 'n_tx_pts': n_t_grid * 256}

    out_tuple = (dataloader_train, train_loader_dict, test_loader_dict, data_info)
    return out_tuple
