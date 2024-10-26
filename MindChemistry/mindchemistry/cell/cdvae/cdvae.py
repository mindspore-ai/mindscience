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
"""CDVAE model
"""

import numpy as np
import mindspore as ms
import mindspore.mint as mint
from tqdm import tqdm
from ...utils.load_config import load_yaml_config_from_path
from ...graph.graph import AggregateNodeToGlobal, LiftGlobalToNode
from ..dimenet import DimeNetPlusPlus
from ..gemnet.data_utils import (
    cart_to_frac_coords, frac_to_cart_coords, min_distance_sqr_pbc,
    cart_to_frac_coords_numpy, frac_to_cart_coords_numpy)
from ..gemnet.layers.embedding_block import MAX_ATOMIC_NUM
from ..gemnet.layers.base_layers import MLP
from ..gemnet.preprocess import GemNetPreprocess
from .decoder import GemNetTDecoder


class CDVAE(ms.nn.Cell):
    r"""
    CDVAE Model

    Args:
        config_path (str): Path to the config file.
        data_config_path (str): Path to the data config file.

    Inputs:
        - **atom_types** (Tensor) - The shape of tensor is :math:`(total\_atoms,)`.
        - **dist** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **idx_kj** (Tensor) - The index of the first edge in the triples.
          The shape of tensor is :math:`(total\_triplets,)`.
        - **idx_ji** (Tensor) - The index of the sechond edge in the triples.
          The shape of tensor is :math:`(total\_triplets,)`.
        - **edge_j** (Tensor) - The index of the first atom of the edges.
          The shape of tensor is :math:`(total\_edges,)`.
        - **edge_i** (Tensor) - The index of the second atom of the edges.
          The shape of tensor is :math:`(total\_edges,)`.
        - **batch** (Tensor) - The shape of Tensor is :math:`(total\_atoms,)`.
        - **length** (Tensor) - The lattice constant of each crystal. The shape of Tensor is :math:`(batch\_size, 3)`.
        - **angles** (Tensor) - The lattice agnle of each crystal. The shape of Tensor is :math:`(batch\_size, 3)`.
        - **num_atoms** (Tensor) - Num_atoms of each crystal. The shape of Tensor is :math:`(batch\_size,)`.
        - **frac_coords** (Tensor) - Position of each atoms. The shape of Tensor is :math:`(total\_atoms,3)`.
        - **y** (Tensor) - Position of each atoms. The shape of Tensor is :math:`(batch\_size,)`.
        - **batch_size** (int) - Batchsize.
        - **sbf** (Tensor) - The shape of Tensor is :math:`(total\_triplets, num\_spherical * num\_radial)`.
        - **total_atoms** (int) - Total atoms
        - **teacher_forcing** (bool) - if teacher_forcing: True, else: False
        - **training** (bool) - If training: True, else: False

    Outputs:
        - **loss** (Tensor) - Scaler

    Raises:
        TypeError: If predict_property is not bool.
        TypeError: If teacher_forcing_lattice is not bool.
        ValueError: If lattice_scale_method is not 'scale_length'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import context, Tensor
        >>> from mindchemistry.cell import CDVAE
        >>> from mindchemistry.cell.cdvae.data_utils import StandardScalerMindspore
        >>> os.environ["MS_JIT_MODULES"] = "mindchemistry"
        >>> context.set_context(mode=context.PYNATIVE_MODE)
        >>> config_path = "./configs.yaml"
        >>> data_config_path = "./perov_5.yaml"
        >>> cdvae_model = CDVAE(config_path, data_config_path)
        >>> # input data
        >>> batch_size = 2
        >>> atom_types = Tensor([6, 7, 6, 8], ms.int32)
        >>> dist = Tensor([1.4, 1.7, 1.8, 1.9, 2.0, 2.1, 1.8, 1.6], ms.float32)
        >>> idx_kj = Tensor([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 7, 6, 6, 7], ms.int32)
        >>> idx_ji = Tensor([1, 0, 3, 2, 5, 4, 4, 5, 2, 3, 0, 1, 6, 7, 7, 6], ms.int32)
        >>> edge_j = Tensor([0, 1, 1, 0, 2, 3, 3, 2], ms.int32)
        >>> edge_i = Tensor([1, 0, 0, 1, 3, 2, 2, 3], ms.int32)
        >>> batch = Tensor([0, 0, 1, 1], ms.int32)
        >>> lengths = Tensor([[2.5, 2.5, 2.5],
        ...                   [2.5, 2.5, 2.5]], ms.float32)
        >>> angles = Tensor([[90.0, 90.0, 90.0],
        ...                  [90.0, 90.0, 90.0]], ms.float32)
        >>> frac_coords = Tensor([[0.0, 0.0, 0.0],
        ...                       [0.5, 0.5, 0.5],
        ...                       [0.7, 0.7, 0.7],
        ...                       [0.5, 0.5, 0.5]], ms.float32)
        >>> num_atoms = Tensor([2, 2], ms.int32)
        >>> y = Tensor([0.08428, 0.01353], ms.float32)
        >>> total_atoms = 4
        >>> sbf = Tensor(np.random.randn(16, 42), ms.float32)
        >>> cdvae_model.lattice_scaler = StandardScalerMindspore(
        ...     Tensor([2.5, 2.5, 2.5, 90.0, 90.0, 90.0], ms.float32),
        ...     Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ms.float32))
        >>> cdvae_model.scaler = StandardScalerMindspore(
        ...     Tensor([2.62], ms.float32),
        ...     Tensor([1.0], ms.float32))
        >>> out = cdvae_model(atom_types, dist,
        ...                   idx_kj, idx_ji, edge_j, edge_i,
        ...                   batch, lengths, num_atoms,
        ...                   angles, frac_coords, y, batch_size,
        ...                   sbf, total_atoms, False, True)
        >>> print("out:", out)
        out: 27.780727
    """

    def __init__(self, config_path, data_config_path):
        super().__init__()
        self.configs = load_yaml_config_from_path(config_path)
        decoder_configs = self.configs.get("Decoder")
        encoder_configs = self.configs.get("Encoder")
        data_configs = load_yaml_config_from_path(data_config_path)
        self.latent_dim = self.configs.get("latent_dim")
        self.hidden_dim = self.configs.get("hidden_dim")
        self.fc_num_layers = self.configs.get("fc_num_layers")
        if isinstance(self.configs.get("predict_property"), bool):
            self.set_predict_property = self.configs.get("predict_property")
        else:
            raise TypeError("predict_property should be bool.")
        self.sigma_begin = self.configs.get("sigma_begin")
        self.sigma_end = self.configs.get("sigma_end")
        self.num_noise_level = self.configs.get("num_noise_level")
        self.type_sigma_begin = self.configs.get("type_sigma_begin")
        self.type_sigma_end = self.configs.get("type_sigma_end")
        if isinstance(self.configs.get("teacher_forcing_lattice"), bool):
            self.teacher_forcing_lattice = self.configs.get(
                "teacher_forcing_lattice")
        else:
            raise TypeError("teacher_forcing_lattice should be bool.")
        if data_configs.get("lattice_scale_method") in ["scale_length"]:
            self.lattice_scale_method = data_configs.get(
                "lattice_scale_method")
        else:
            raise ValueError(
                "For lattice scale method, supported methods: 'scale_length'.")
        self.max_atoms = data_configs.get("max_atoms")

        self.encoder = DimeNetPlusPlus(
            num_targets=self.latent_dim,
            hidden_channels=encoder_configs.get("hidden_channels"),
            num_blocks=encoder_configs.get("num_blocks"),
            int_emb_size=encoder_configs.get("int_emb_size"),
            basis_emb_size=encoder_configs.get("basis_emb_size"),
            out_emb_channels=encoder_configs.get("out_emb_channels"),
            num_spherical=encoder_configs.get("num_spherical"),
            num_radial=encoder_configs.get("num_radial"),
            cutoff=encoder_configs.get("cutoff"),
            envelope_exponent=encoder_configs.get("envelope_exponent"),
            num_before_skip=encoder_configs.get("num_before_skip"),
            num_after_skip=encoder_configs.get("num_after_skip"),
            num_output_layers=encoder_configs.get("num_output_layers"),
            readout=data_configs.get("readout"))

        self.decoder = GemNetTDecoder(
            hidden_dim=decoder_configs.get("hidden_dim"),
            latent_dim=self.configs.get("latent_dim"),
            max_neighbors=self.configs.get("max_neighbors"),
            radius=self.configs.get("radius"),
            config_path=config_path
        )
        self.fc_mu = mint.nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = mint.nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_num_atoms = MLP(self.latent_dim, self.hidden_dim,
                                self.fc_num_layers, self.max_atoms + 1,
                                activation='ReLU')
        self.fc_lattice = MLP(self.latent_dim, self.hidden_dim,
                              self.fc_num_layers, 6,
                              activation='ReLU')
        self.max_atomic_num = MAX_ATOMIC_NUM
        self.fc_composition = MLP(self.latent_dim, self.hidden_dim,
                                  self.fc_num_layers, self.max_atomic_num,
                                  activation='ReLU')
        # for property prediction.
        if self.set_predict_property:
            self.fc_property = MLP(self.latent_dim, self.hidden_dim,
                                   self.fc_num_layers, 1,
                                   activation='ReLU')
        self.sigmas = np.exp(np.linspace(
            np.log(self.sigma_begin),
            np.log(self.sigma_end),
            self.num_noise_level))

        self.type_sigmas = np.exp(np.linspace(
            np.log(self.type_sigma_begin),
            np.log(self.type_sigma_end),
            self.num_noise_level))

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None
        self.decoder_preprocess = GemNetPreprocess(otf_graph=True)
        self.aggregate_mean = AggregateNodeToGlobal(mode="mean")
        self.lift_global = LiftGlobalToNode()

    def reparameterize(self, mu, logvar):
        r"""
        Reparameterization trick to sample to N(mu, var) from N(0,1).

        Args:
            mu (Tensor): Mean of the latent Gaussian.
                The shape of tensor is :math:`(batch\_size, latent\_dim)`.
            logvar (Tensor): Standard deviation of the latent Gaussian.
                The shape of tensor is :math:`(batch\_size, latent\_dim)`.

        Returns:
            (Tensor) Randomly generated latent parameter.
            The shape of tensor is :math:`(batch\_size, latent\_dim)`.
        """

        std = mint.exp(mint.mul(0.5, logvar))
        eps = ms.Tensor(np.random.randn(
            std.shape[0], std.shape[1]), ms.float32)
        return eps * std + mu

    def encode(self, atom_types, dist, idx_kj, idx_ji, edge_j, edge_i, batch, total_atoms, batch_size, sbf):
        r"""
        encode crystal structures to latents.

        Args:
            atom_types (Tensor): Atom types of each atom.
                The shape of tensor is :math:`(total\_atoms,)`.
            dist (Tensor): Distance between atoms.
                The shape of tensor is :math:`(total\_edges,)`.
            idx_kj (Tensor): The index of the first edge in the triples.
                The shape of tensor is :math:`(total\_triplets,)`.
            idx_ji (Tensor): The index of the sechond edge in the triples.
                The shape of tensor is :math:`(total\_triplets,)`.
            edge_j (Tensor): The index of the first atom of the edges.
                The shape of tensor is :math:`(total\_edges,)`.
            edge_i (Tensor): The index of the second atom of the edges.
                The shape of tensor is :math:`(total\_edges,)`.
            batch (Tensor): The shape of tensor is :math:`(total\_atoms,)`.
            total_atoms (int): Total atoms.
            batch_size (int): Batch size.
            sbf (Tensor): The shape of tensor is :math:`(total\_triplets, num\_spherical * num\_radial)`.

        Returns:
            mu (Tensor): Mean of the latent Gaussian.
            The shape of tensor is :math:`(batch\_size, latent\_dim)`.
            log_var (Tensor): Standard deviation of the latent Gaussian.
            The shape of tensor is :math:`(batch\_size, latent\_dim)`.
            z (Tensor): Randomly generated latent parameter.
            The shape of tensor is :math:`(batch\_size, latent\_dim)`.
        """
        hidden = self.encoder(atom_types, dist, idx_kj, idx_ji, edge_j, edge_i,
                              batch, total_atoms, batch_size, sbf)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(self, z, batch=None, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        r"""
        Decode key statistics from latent embeddings.

        This method decodes key statistics from the given latent embeddings.

        Args:
            z (Tensor): Randomly generated latent parameter.
                The shape of the tensor is :math:`(batch\_size, latent\_dim)`.
            batch (Tensor, optional): The shape of the tensor is :math:`(total\_atoms,)`.
            gt_num_atoms (Tensor, optional): Ground truth number of atoms.
            gt_lengths (Tensor, optional): Ground truth lattice constant.
            gt_angles (Tensor, optional): Ground truth lattice angle.
            teacher_forcing (bool): If `True`, teacher forcing is used during training;
                otherwise, `False`. Default is `False`.

        Returns:
            tuple of Tensor including: num_atoms, lengths_and_angles, lengths, angles,
            composition_per_atom, z_per_atom and batch.
            num_atoms (Tensor): The predicted number of atoms. The shape of the tensor is :math:`(batch\_size,)`.
            lengths_and_angles (Tensor): The predicted lattice constants and angles.
            The shape of the tensor is :math:`(batch\_size, 6)`.
            lengths (Tensor): The predicted lattice constants. The shape of the tensor is :math:`(batch\_size, 3)`.
            angles (Tensor): The predicted lattice angles. The shape of the tensor is :math:`(batch\_size, 3)`.
            composition_per_atom (Tensor): The predicted composition per atom.
            The shape of the tensor is :math:`(total\_atoms, max_atomic\_num)`.
            z_per_atom (Tensor): The lifted global latent embeddings per atom.
            The shape of the tensor is :math:`(total\_atoms, latent\_dim)`.
            batch (Tensor): The batch tensor used to index which sample is the node belonging.
            The shape of the tensor is :math:`(total\_atoms,)`.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            assert batch is not None
            z_per_atom = self.lift_global(z, batch)
            composition_per_atom = self.predict_composition(z_per_atom)
            if self.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = mint.argmax(
                self.predict_num_atoms(z), dim=-1).astype(ms.int32)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            batch = ms.ops.repeat_interleave(mint.arange(
                num_atoms.shape[0], dtype=ms.int32), num_atoms)
            z_per_atom = self.lift_global(z, batch)
            composition_per_atom = self.predict_composition(z_per_atom)
        # set the max and min values for lengths and angles.
        angles = mint.clamp(angles, -180, 180)
        lengths = mint.clamp(lengths, 0.5, 20)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom, z_per_atom, batch

    def langevin_dynamics(self, z, ld_kwargs, batch_size, total_atoms=None, gt_num_atoms=None, gt_atom_types=None):
        r"""
        decode crystral structure from latent embeddings.

        Args:
            ld_kwargs: args for doing annealed langevin dynamics sampling:
                n_step_each (int): number of steps for each sigma level.
                step_lr (int): step size param.
                min_sigma (int): minimum sigma to use in annealed langevin dynamics.
                save_traj (bool): if True, save the entire LD trajectory.
                disable_bar (bool): disable the progress bar of langevin dynamics.
            gt_num_atoms (Tensor, optional): if not <None>, use the ground truth number of atoms.
            gt_atom_types (Tensor, optional): if not <None>, use the ground truth atom types.

        Returns:
            (dict): Including num_atoms, lengths, anlges, frac_coords, atom_types, is_traj
                num_atoms (Tensor): Number of atoms in each crystal.
                The shape of tensor is :math:`(batch\_size,)`.
                lengths (Tensor): Lattice constant of each crystal.
                The shape of tensor is :math:`(batch\_size, 3)`.
                angles (Tensor): Lattice angle of each crystal.
                The shape of tensor is :math:`(batch\_size, 3)`.
                frac_coords (Tensor): Fractional coordinates of each atom.
                The shape of tensor is :math:`(total\_atoms, 3)`.
                atom_types (Tensor): Atom types of each atom.
                The shape of tensor is :math:`(total\_atoms,)`.
                is_traj (bool): If True, save the entire LD trajectory.
        """

        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []
        # obtain key stats.
        num_atoms, _, lengths, angles, composition_per_atom, z_per_atom, batch = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms
        else:
            total_atoms = num_atoms.sum().item()
        # obtain atom types.
        composition_per_atom = mint.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms, batch, batch_size, total_atoms)
        else:
            cur_atom_types = gt_atom_types
        # init coords.
        cur_frac_coords = np.random.rand(total_atoms, 3)

        # annealed langevin dynamics.
        for sigma in tqdm(self.sigmas, total=self.sigmas.shape[0], disable=ld_kwargs.disable_bar, position=0):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2
            step_size_ms = ms.Tensor(step_size, ms.float32)

            for _ in range(ld_kwargs.n_step_each):
                noise_cart = np.random.randn(cur_frac_coords.shape[0],
                                             cur_frac_coords.shape[1]) * np.sqrt(step_size * 2)
                noise_cart = ms.Tensor(noise_cart, ms.float32)
                (_, idx_s, idx_t, id3_ca, id3_ba,
                 id3_ragged_idx, id3_ragged_idx_max, _, d_st, v_st,
                 id_swap, y_l_m) = self.decoder_preprocess.graph_generation(
                     cur_frac_coords, num_atoms.asnumpy(),
                     lengths.asnumpy(), angles.asnumpy(),
                     edge_index=None,
                     to_jimages=None,
                     num_bonds=None)
                cur_frac_coords = ms.Tensor(cur_frac_coords, ms.float32)
                batch = ms.ops.repeat_interleave(
                    mint.arange(num_atoms.shape[0]), num_atoms, 0)

                #### decoder ####
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    cur_atom_types, idx_s, idx_t, id3_ca, id3_ba, id3_ragged_idx, id3_ragged_idx_max,
                    y_l_m, d_st, v_st, id_swap, batch, z_per_atom, total_atoms, batch_size)

                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, batch, self.lift_global)
                pred_cart_coord_diff = mint.div(
                    pred_cart_coord_diff, ms.Tensor(sigma, ms.float32))
                cur_cart_coords = cur_cart_coords + \
                    mint.mul(step_size_ms, pred_cart_coord_diff) + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, batch, self.lift_global)

                if gt_atom_types is None:
                    cur_atom_types = mint.argmax(pred_atom_types, dim=1) + 1
                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)
                cur_frac_coords = cur_frac_coords.asnumpy()

        output_dict = {
            "num_atoms": num_atoms, "lengths": lengths, "angles": angles,
            "frac_coords": ms.Tensor(cur_frac_coords), "atom_types": cur_atom_types,
            "is_traj": False
        }
        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=mint.stack(all_frac_coords, dim=0),
                all_atom_types=mint.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=mint.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=mint.stack(all_noise_cart, dim=0),
                is_traj=True))
        return output_dict

    def construct(self, atom_types, dist, idx_kj, idx_ji, edge_j, edge_i,
                  batch, lengths, num_atoms, angles, frac_coords, y, batch_size, sbf, total_atoms,
                  teacher_forcing=True, training=True):
        """CDVAE construct"""
        ########### encoder ############
        mu, log_var, z = self.encode(atom_types, dist, idx_kj, idx_ji, edge_j, edge_i,
                                     batch, total_atoms, batch_size, sbf)
        ########### decode stats ############
        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom, z_per_atom, batch) = self.decode_stats(
             z, batch, num_atoms, lengths, angles, teacher_forcing)

        out = self.add_noise(atom_types, num_atoms, pred_lengths, pred_angles, frac_coords,
                             pred_composition_per_atom)
        (num_atoms_numpy, pred_lengths_numpy, pred_angles_numpy, noisy_frac_coords,
         used_sigmas_per_atom, used_type_sigmas_per_atom, rand_atom_types) = out

        (_, idx_s, idx_t, id3_ca, id3_ba,
         id3_ragged_idx, id3_ragged_idx_max, _, d_st, v_st,
         id_swap, y_l_m) = self.decoder_preprocess.graph_generation(
             noisy_frac_coords, num_atoms_numpy, pred_lengths_numpy, pred_angles_numpy,
             edge_index=None, to_jimages=None, num_bonds=None,)
        # switch to ms.Tensor
        noisy_frac_coords = ms.Tensor(noisy_frac_coords, ms.float32)
        used_sigmas_per_atom = ms.Tensor(used_sigmas_per_atom, ms.float32)
        used_type_sigmas_per_atom = ms.Tensor(
            used_type_sigmas_per_atom, ms.float32)
        rand_atom_types = ms.Tensor(rand_atom_types, ms.int32)

        ################ decoder ############
        pred_cart_coord_diff, pred_atom_types = self.decoder(
            rand_atom_types, idx_s, idx_t, id3_ca, id3_ba, id3_ragged_idx, id3_ragged_idx_max,
            y_l_m, d_st, v_st, id_swap, batch, z_per_atom, total_atoms, batch_size)

        ################ compute loss ############
        num_atom_loss = self.num_atom_loss(pred_num_atoms, num_atoms)
        lattice_loss = self.lattice_loss(
            pred_lengths_and_angles, lengths, num_atoms, angles)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, atom_types, batch, batch_size)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom,
            batch, lengths, angles, frac_coords, batch_size)
        type_loss = self.type_loss(pred_atom_types, atom_types,
                                   used_type_sigmas_per_atom, batch, batch_size)
        kld_loss = self.kld_loss(mu, log_var)

        if self.set_predict_property:
            property_loss = self.property_loss(z, y)
        else:
            property_loss = ms.Tensor([0], ms.float32)
        outputs = {
            "num_atom_loss": num_atom_loss,
            "lattice_loss": lattice_loss,
            "composition_loss": composition_loss,
            "coord_loss": coord_loss,
            "type_loss": type_loss,
            "kld_loss": kld_loss,
            "property_loss": property_loss,
            "pred_num_atoms": pred_num_atoms,
            "pred_lengths_and_angles": pred_lengths_and_angles,
            "pred_lengths": pred_lengths,
            "pred_angles": pred_angles,
            "pred_cart_coord_diff": pred_cart_coord_diff,
            "pred_atom_types": pred_atom_types,
            "pred_composition_per_atom": pred_composition_per_atom,
            "target_frac_coords": frac_coords,
            "target_atom_types": atom_types,
            "rand_frac_coords": noisy_frac_coords,
            "rand_atom_types": rand_atom_types,
            "z": z,
        }
        loss = self.compute_stats(batch, outputs, batch_size, training)
        return loss

    def add_noise(self, atom_types, num_atoms, pred_lengths, pred_angles, frac_coords,
                  pred_composition_per_atom):
        r"""
        Adds noise to the given input parameters and returns the modified values.

        Args:
            atom_types (Tensor): Array of atom types. The shape of tensor is :math:`(total\_atoms,)`.
            num_atoms (Tensor): Array of number of atoms. The shape of tensor is :math:`(batch\_size,)`.
            pred_lengths (Tensor): Array of predicted lengths. The shape of tensor is :math:`(batch\_size, 3)`.
            pred_angles (Tensor): Array of predicted angles. The shape of tensor is :math:`(batch\_size, 3)`.
            frac_coords (Tensor): Array of fractional coordinates. The shape of tensor is :math:`(total\_atoms, 3)`.
            pred_composition_per_atom (Tensor): Array of predicted composition probabilities per atom.
                The shape of tensor is :math:`(total\_atoms,)`.

        Returns:
            Tuple of ndarray, including: num_atoms_numpy, pred_lengths_numpy, pred_angles_numpy, noisy_frac_coords,
            used_sigmas_per_atom, used_type_sigmas_per_atom and rand_atom_types.
        """
        one_hot_res = mint.nn.functional.one_hot(
            mint.sub(atom_types, 1), self.max_atomic_num)
        one_hot_res = one_hot_res.asnumpy()
        pred_composition_probs = mint.softmax(
            pred_composition_per_atom, dim=-1)
        pred_composition_probs = pred_composition_probs.asnumpy()
        num_atoms_numpy = num_atoms.asnumpy()
        pred_lengths_numpy = pred_lengths.asnumpy()
        pred_angles_numpy = pred_angles.asnumpy()
        frac_coords_numpy = frac_coords.asnumpy()
        # sample noise levels.
        noise_level = np.random.randint(
            0, self.sigmas.shape[0], (1, num_atoms.shape[0]))
        used_sigmas_per_atom = np.repeat(
            self.sigmas[noise_level], num_atoms_numpy)
        type_noise_level = np.random.randint(
            0, self.type_sigmas.shape[0], (1, num_atoms.shape[0]))
        # test num_atoms
        used_type_sigmas_per_atom = np.repeat(self.type_sigmas[type_noise_level],
                                              num_atoms_numpy)
        # add noise to atom types and sample atom types.
        atom_type_probs = (one_hot_res + pred_composition_probs *
                           used_type_sigmas_per_atom[:, None])

        rand_atom_types = np.zeros(atom_types.shape[0])
        for i in range(atom_types.shape[0]):
            rand_atom_types[i] = np.random.choice(
                100, 1, p=atom_type_probs[i] / atom_type_probs[i].sum()) + 1

        coord_rand = np.random.rand(frac_coords.shape[0], frac_coords.shape[1])
        cart_noises_per_atom = (
            coord_rand * used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords_numpy(
            frac_coords_numpy, pred_lengths_numpy, pred_angles_numpy, num_atoms_numpy)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords_numpy(
            cart_coords, pred_lengths_numpy, pred_angles_numpy, num_atoms_numpy)
        return (num_atoms_numpy, pred_lengths_numpy, pred_angles_numpy, noisy_frac_coords,
                used_sigmas_per_atom, used_type_sigmas_per_atom, rand_atom_types)

    def sample_composition(self, composition_prob, num_atoms, batch, batch_size, total_atoms):
        r"""
        Samples composition such that it exactly satisfies composition_prob

        Args:
            composition_prob (Tensor): The shape of tensor is :math:`(total\_atoms, max_atomic_num)`.
            num_atoms (Tensor): The shape of tensor is :math:`(batch\_size,)`.
            batch (Tensor): The shape of tensor is :math:`(total\_atoms,)`.
            batch_size (int): Batch size.
            total_atoms (int): Total atoms.

        Returns:
            (Tensor): Sampled composition.
        """
        assert composition_prob.shape[0] == total_atoms == batch.shape[0]
        out = mint.zeros((batch_size, composition_prob.shape[1]))
        composition_prob = self.aggregate_mean(composition_prob, batch, out)
        all_sampled_comp = []
        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = ms.ops.round(comp_prob * num_atom).astype(ms.int32)
            if mint.max(comp_num) != 0:
                atom_type = mint.nonzero(comp_num)[:, 0] + 1
            else:
                atom_type = (mint.argmax(
                    mint.mul(comp_prob, num_atom)) + 1).view(1, -1)
                comp_num[atom_type - 1] = 1
            atom_num = comp_num[atom_type - 1].view(-1)

            sampled_comp = ms.ops.repeat_interleave(
                atom_type, atom_num).astype(ms.int32)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.shape[0] < num_atom:
                left_atom_num = num_atom - sampled_comp.shape[0]
                left_comp_prob = mint.div(
                    comp_prob - comp_num.float(), num_atom)
                # left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp_prob = mint.where(
                    left_comp_prob < 0., 0., left_comp_prob)
                left_comp = ms.ops.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = mint.cat((sampled_comp, left_comp), dim=0)
            # sampled_comp[:num_atom])
            sampled_comp = ms.ops.shuffle(mint.narrow(
                sampled_comp, 0, 0, num_atom.item()))
            all_sampled_comp.append(sampled_comp)
        all_sampled_comp = mint.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.shape[0] == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_lattice(self, z, num_atoms):
        """predict lattice constants and angles"""
        pred_lengths_and_angles = self.fc_lattice(z)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths, pred_angles = mint.split(scaled_preds, 3, 1)
        if self.lattice_scale_method == "scale_length":
            pred_lengths = mint.mul(pred_lengths, mint.pow(
                num_atoms.view(-1, 1), (1 / 3)))
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z_per_atom):
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, num_atoms):
        """compute num atom loss"""
        return ms.ops.cross_entropy(pred_num_atoms, num_atoms)

    def property_loss(self, z, y):
        """compute property loss"""
        return ms.ops.mse_loss(self.fc_property(z), y)

    def lattice_loss(self, pred_lengths_and_angles, lengths, num_atoms, angles):
        """compute lattice loss"""
        assert self.lattice_scale_method == "scale_length"
        target_lengths = lengths / \
            mint.pow(num_atoms.view(-1, 1), (1 / 3))
        target_lengths_and_angles = mint.cat(
            (target_lengths, angles), dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return ms.ops.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch, batch_size):
        """compute composition loss"""
        target_atom_types = target_atom_types - 1
        loss = ms.ops.cross_entropy(pred_composition_per_atom,
                                    target_atom_types, reduction="none")
        out = mint.zeros(batch_size)
        return mint.mean(self.aggregate_mean(loss, batch, out))

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch, lengths, angles,
                   frac_coords, batch_size):
        r"""
        comput coord loss

        Args:
            pred_cart_coord_diff (Tensor): The shape of tensor is :math:`(total\_atoms, 3)`..
            noisy_frac_coords (Tensor): The shape of tensor is :math:`(total\_atoms, 3)`.
            used_sigmas_per_atom (Tensor): The shape of tensor is :math:`(total\_atoms,)`.
            batch (Tensor): The shape of tensor is :math:`(total\_atoms,)`.
            lengths (Tensor): The shape of tensor is :math:`(batch\_size, 3)`.
            angles (Tensor): The shape of tensor is :math:`(batch\_size, 3)`.
            frac_coords (Tensor): The shape of tensor is :math:`(total\_atoms, 3)`.
            batch_size (int): Batch size.

        Returns:
            (Tensor): Loss.
        """
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, lengths, angles, batch, self.lift_global)
        target_cart_coords = frac_to_cart_coords(
            frac_coords, lengths, angles, batch, self.lift_global)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, lengths, angles,
            batch, batch_size, self.lift_global, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            ms.ops.pow(used_sigmas_per_atom.view(-1, 1), 2)
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom.view(-1, 1)

        loss_per_atom = mint.sum(
            ms.ops.pow((target_cart_coord_diff - pred_cart_coord_diff), 2), dim=1)

        loss_per_atom = 0.5 * loss_per_atom * \
            ms.ops.pow(used_sigmas_per_atom, 2)
        out = mint.zeros(batch_size)
        return mint.mean(self.aggregate_mean(loss_per_atom, batch, out))

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch, batch_size):
        """compute type loss"""
        target_atom_types = target_atom_types - 1
        loss = ms.ops.cross_entropy(
            pred_atom_types, target_atom_types, reduction="none")
        # rescale loss according to noise
        loss = mint.div(loss, used_type_sigmas_per_atom)
        out = mint.zeros(batch_size)
        return mint.mean(self.aggregate_mean(loss, batch, out))

    def kld_loss(self, mu, log_var):
        """compute kld loss"""
        kld_loss = mint.mean(
            -0.5 * mint.sum(mint.sub(mint.sub(mint.add(1, log_var), ms.ops.pow(mu, 2)),
                                     mint.exp(log_var)), dim=1), dim=0)
        return kld_loss

    def compute_stats(self, batch, outputs, batch_size, prefix):
        r"""compute stats
        Args:
            batch (Tensor): The shape of tensor is :math:`(total\_atoms,)`.
            outputs (dict): The output dict.
            batch_size (int): Batch size.
            prefix (bool): If True, return training loss,
                else, return validation loss.
        Returns:
            (Tensor): Loss.
        """
        num_atom_loss = outputs["num_atom_loss"]
        lattice_loss = outputs["lattice_loss"]
        coord_loss = outputs["coord_loss"]
        type_loss = outputs["type_loss"]
        kld_loss = outputs["kld_loss"]
        composition_loss = outputs["composition_loss"]
        property_loss = outputs["property_loss"]

        cost_natom = self.configs.get("cost_natom")
        cost_coord = self.configs.get("cost_coord")
        cost_type = self.configs.get("cost_type")
        cost_lattice = self.configs.get("cost_lattice")
        cost_composition = self.configs.get("cost_composition")
        cost_property = self.configs.get("cost_property")
        beta = self.configs.get("beta")

        loss = mint.sum(mint.stack((
            mint.mul(cost_natom, num_atom_loss),
            mint.mul(cost_lattice, lattice_loss),
            mint.mul(cost_coord, coord_loss),
            mint.mul(cost_type, type_loss),
            mint.mul(beta, kld_loss),
            mint.mul(cost_composition, composition_loss),
            mint.mul(cost_property, property_loss))))

        if prefix is False:
            # validation/test loss only has coord and type
            loss = (
                cost_coord * coord_loss +
                cost_type * type_loss)

            # evaluate atom type prediction.
            pred_atom_types = outputs["pred_atom_types"]
            target_atom_types = outputs["target_atom_types"]
            type_accuracy = pred_atom_types.argmax(
                axis=-1) == (target_atom_types - 1)
            type_accuracy = type_accuracy.astype(ms.float32)
            out = mint.zeros_like(type_accuracy[:batch_size])
            type_accuracy = mint.mean(
                self.aggregate_mean(type_accuracy, batch, out))

        return loss
