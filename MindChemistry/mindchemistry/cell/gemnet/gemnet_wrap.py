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
"""gemnet wrap"""

import mindspore as ms
from ...utils.load_config import load_yaml_config_from_path
from .gemnet import GemNetT
from .preprocess import GemNetPreprocess


class GemNetWrap:
    r"""
    Wrapper class for the GemNet model.
    The GemNet Model is used in CDVAE.
    Warning: This is not the original GemNet model, but a modified version for the CDVAE model.
    Pay attention to the differece between the original GemNet model and this model.

    Args:
        config_path (str): Path to the config file.

    Inputs:
        - **atom_types** (Tensor) - The shape of the tensor is :math:`(total\_atoms,)`.
        - **idx_s** (Tensor) - The shape of the tensor is :math:`(total\_edges,)`.
        - **idx_t** (Tensor) - The shape of the tensor is :math:`(total\_edges,)`.
        - **id3_ca** (Tensor) - The shape of the tensor is :math:`(total\_triplets,)`.
        - **id3_ba** (Tensor) - The shape of the tensor is :math:`(total\_triplets,)`.
        - **id3_ragged_idx** (Tensor) - The shape of the tensor is :math:`(total\_triplets,)`.
        - **id3_ragged_idx_max** (int) - The maximum of id3_ragged_idx.
        - **y_l_m** (Tensor) - The shape of the tensor is :math:`(total\_triplets,)`.
        - **d_st** (Tensor) - The shape of the tensor is :math:`(total\_edges,)`.
        - **v_st** (Tensor) - The shape of the tensor is :math:`(total\_edges, 3)`.
        - **id_swap** (Tensor) - The shape of the tensor is :math:`(total\_triplets,)`.
        - **batch** (Tensor) - The shape of the tensor is :math:`(total\_edges,)`.
        - **total_atoms** (int) - The total number of atoms.
        - **batch_size** (int) - The batch size.

    Outputs:
        - **h** (Tensor) - Used only in CDVAE. The shape of Tensor is :math:`(total\_atoms, emb_size_atom)`.
        - **f_j** (Tensor) - The shape of tensor is :math:`(total\_atoms, 3)`.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindchemistry.cell.gemnet import GemNetWrap
        >>> config_path = "./configs.yaml"
        >>> gemnet_model = GemNetWrap(config_path)
        >>> # input data
        >>> batch_size = 2
        >>> atom_types = Tensor([6, 7, 6, 8], ms.int32)
        >>> lengths = Tensor([[2.5, 2.5, 2.5],
        ...                   [2.5, 2.5, 2.5]], ms.float32)
        >>> angles = Tensor([[90.0, 90.0, 90.0],
        ...                  [90.0, 90.0, 90.0]], ms.float32)
        >>> frac_coords = Tensor([[0.0, 0.0, 0.0],
        ...                       [0.5, 0.5, 0.5],
        ...                       [0.7, 0.7, 0.7],
        ...                       [0.5, 0.5, 0.5]], ms.float32)
        >>> num_atoms = Tensor([2, 2], ms.int32)
        >>> total_atoms = 4
        >>> h, f_j = gemnet.evaluation(atom_types, num_atoms, frac_coords,
        ...                            lengths, angles, batch, total_atoms, batch_size)
        >>> print(h.shape)
        (4, 128)
        >>> print(f_j.shape)
        (4, 3)
    """

    def __init__(self, config_path):
        configs = load_yaml_config_from_path(config_path)
        gemnet_config = configs.get('Decoder')
        self.preprocess = GemNetPreprocess(otf_graph=True)
        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=configs.get('latent_dim'),
            emb_size_atom=gemnet_config.get("hidden_dim"),
            emb_size_edge=gemnet_config.get("hidden_dim"),
            regress_forces=True,
            cutoff=configs.get("radius"),
            max_neighbors=configs.get("max_neighbors"),
            config_path=config_path
        )

    def evaluation(self, frac_coords, num_atoms, lengths, angles,
                   atom_types, batch, total_atoms, batch_size):
        """Perform evaluation using the GemNet model.

        Args:
            frac_coords (numpy.ndarray): The fractional coordinates of the atoms.
            num_atoms (int): The number of atoms.
            lengths (numpy.ndarray): The lengths of the cell.
            angles (numpy.ndarray): The angles of the cell.
            atom_types (numpy.ndarray): The types of the atoms.
            batch (numpy.ndarray): The batch indices.
            total_atoms (int): The total number of atoms in the system.
            batch_size (int): The batch size.

        Returns:
            tuple: A tuple containing the energy and force predictions.
                - E_a (ms.Tensor): The predicted energies.
                - F_j (ms.Tensor): The predicted forces.
        """
        (_, idx_s, idx_t, id3_ca, id3_ba,
         id3_ragged_idx, id3_ragged_idx_max, _, d_st, v_st,
         id_swap, y_l_m) = self.preprocess.graph_generation(
             frac_coords, num_atoms, lengths, angles,
             edge_index=None, to_jimages=None, num_bonds=None,)
        atom_types = ms.Tensor(atom_types, ms.int32)
        h, f_j = self.gemnet(
            atom_types, idx_s, idx_t, id3_ca, id3_ba, id3_ragged_idx,
            id3_ragged_idx_max, y_l_m, d_st, v_st, id_swap, batch,
            None, total_atoms, batch_size
        )
        return h, f_j
