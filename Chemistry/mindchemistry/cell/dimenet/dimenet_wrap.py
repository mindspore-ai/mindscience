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
"""dimenet wrap"""

from ...utils.load_config import load_yaml_config_from_path
from .dimenet import DimeNetPlusPlus
from .preprocess import PreProcess


class DimeNetWrap:
    r"""
    Wrapper class for the DimeNet model.
    The DimeNet Model is used in CDVAE.
    Warning: This is not the original DimeNet model, but a modified version for the CDVAE model.
    Pay attention to the differece between the original DimeNet model and this model.

    Args:
        config_path (str): Path to the configuration file.
        data_config_path (str): Path to the data configuration file.
    Inputs:
        - **angles** (np.ndarray) - The shape of ndarray is :math:`(batch\_size, 3)`.
        - **lengths** (np.ndarray) - The shape of ndarray is :math:`(batch\_size, 3)`.
        - **num_atoms** (np.ndarray) - The shape of ndarray is :math:`(batch\_size,)`.
        - **edge_index** (np.ndarray) - The shape of ndarray is :math:`(2, total\_edges)`.
        - **frac_coords** (np.ndarray) - The shape of ndarray is :math:`(total\_atoms, 3)`.
        - **num_bonds** (np.ndarray) - The shape of ndarray is :math:`(batch\_size,)`.
        - **to_jimages** (np.ndarray) - The shape of ndarray is :math:`(total\_edges,)`.
        - **atom_types** (np.ndarray) - The shape of ndarray is :math:`(total\_atoms,)`.
        - **y** (np.ndarray) - The shape of ndarray is :math:`(batch\_size,)`.
    Outputs:
        - **energy** (np.ndarray) - The shape of ndarray is :math:`(batch\_size,)`.

    Raises:
        TypeError: If predict_property is not bool.
        TypeError: If teacher_forcing_lattice is not bool.
        ValueError: If lattice_scale_method is not 'scale_length'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindchemistry.cell.dimenet import DimeNetWrap
        >>> ms.set_context(device_target="Ascend", device_id=0, mode="PYNATIVE")
        >>> config_path = "./configs.yaml"
        >>> data_config_path = "./perov_5.yaml"
        >>> atom_types = Tensor([6, 7, 6, 8], ms.int32)
        >>> dimenet = DimeNetWrap(config_path, data_config_path, num_targets=1)
        >>> batch_size = 2
        >>> atom_types = np.array([6, 7, 6, 8], np.int32)
        >>> edge_index = np.array([[0, 1, 1, 0, 2, 3, 3, 2],
        ...                        [1, 0, 0, 1, 3, 2, 2, 3]], np.int32)
        >>> lengths = np.array([[2.5, 2.5, 2.5],
        ...                     [2.5, 2.5, 2.5]], np.float32)
        >>> angles = np.array([[90, 90, 90],
        ...                    [90, 90, 90]], np.float32)
        >>> num_atoms = np.array([2, 2], np.int32)
        >>> num_bonds = np.array([4, 4], np.int32)
        >>> to_jimages = np.zeros((edge_index[1], 3), np.int32)
        >>> frac_coords = np.array([[0.0, 0.0, 0.0],
        ...                         [0.5, 0.5, 0.5],
        ...                         [0.7, 0.7, 0.7],
        ...                         [0.5, 0.5, 0.5]], np.float32)
        >>> y = np.array([0.08428, 0.01353], np.float32)
        >>> total_atoms = 4
        >>> out = dimenet.evaluation(angles, lengths, num_atoms, edge_index,
                                        frac_coords, num_bonds, to_jimages, atom_types, y)
        >>> print(out.shape)
        (2,)
    """

    def __init__(self, config_path, data_config_path, num_targets=None):
        super().__init__()
        configs = load_yaml_config_from_path(config_path)
        dimenet_config = configs.get("Encoder")
        data_config = load_yaml_config_from_path(data_config_path)
        self.preprocess = PreProcess(
            num_spherical=dimenet_config.get("num_spherical"),
            num_radial=dimenet_config.get("num_radial"),
            envelope_exponent=dimenet_config.get("envelope_exponent"),
            otf_graph=False,
            cutoff=dimenet_config.get("cutoff"),
            max_num_neighbors=dimenet_config.get("max_num_neighbors"),
            task="dimenet"
        )
        self.latent_dim = configs.get(
            "latent_dim") if num_targets is None else num_targets
        self.dimenet = DimeNetPlusPlus(
            num_targets=self.latent_dim,
            hidden_channels=dimenet_config.get("hidden_channels"),
            num_blocks=dimenet_config.get("num_blocks"),
            int_emb_size=dimenet_config.get("int_emb_size"),
            basis_emb_size=dimenet_config.get("basis_emb_size"),
            out_emb_channels=dimenet_config.get("out_emb_channels"),
            num_spherical=dimenet_config.get("num_spherical"),
            num_radial=dimenet_config.get("num_radial"),
            cutoff=dimenet_config.get("cutoff"),
            envelope_exponent=dimenet_config.get("envelope_exponent"),
            num_before_skip=dimenet_config.get("num_before_skip"),
            num_after_skip=dimenet_config.get("num_after_skip"),
            num_output_layers=dimenet_config.get("num_output_layers"),
            readout=data_config.get("readout")
        )

    def evaluation(self, angles, lengths, num_atoms, edge_index, frac_coords, num_bonds, to_jimages, atom_types):
        """
        Perform evaluation using the DimeNet model.
        """
        total_atoms = int(num_atoms.sum())
        batch_size = num_atoms.shape[0]
        (atom_types, dist, idx_kj, idx_ji, edge_j, edge_i,
         batch, sbf) = self.preprocess.data_process(angles, lengths, num_atoms,
                                                    edge_index, frac_coords, num_bonds,
                                                    to_jimages, atom_types)
        energy = self.dimenet(atom_types, dist, idx_kj, idx_ji, edge_i, edge_j,
                              batch, total_atoms, batch_size, sbf)
        return energy
