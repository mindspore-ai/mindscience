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
# ==============================================================================
"""allegro
"""

import mindspore as ms
from mindspore import Tensor, ops, float16, float32
from mindspore.nn import Cell, CellList, Identity

from ...cell.basic_block import MLPMixPrecision as MLP
from ...e3.o3 import Irreps, Irrep, Linear, SphericalHarmonics
from ...graph.graph import AggregateEdgeToNode, LiftNodeToEdge
from .strided import MakeWeightedChannels, Contractor, Allegro_Linear


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    """tp_path_exists

    Args:
        irreps_in1: irreps_in1
        irreps_in2: irreps_in2
        ir_out: ir_out

    Returns:
        bool: tp_path_exists or not
    """
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1.data:
        for _, ir2 in irreps_in2.data:
            if ir_out in ir1 * ir2:
                return True

    return False


class Allegro(Cell):
    r"""
    Allegro model.

    Args:
        l_max (int): max irreps order of spherical_harmonics embeddings. Default: ``1``.
        parity_setting (string): the parity settings. Default: ``"o3_full"``.
        num_layers (int): layer number of allegro network. Default: ``1``.
        env_embed_multi (int): the number of channels of the feature in the network. Default: ``8``.
        avg_num_neighbor (float): average number of neighborhood atoms. Default: ``1.0``.
        two_body_kwargs (dict): arguments of two body latent MLP. Default: ``None``.
        latent_kwargs (dict): arguments of latent MLP. Default: ``None``.
        env_embed_kwargs (dict): arguments of environment embedded MLP. Default: ``None``.
        irreps_in (Irreps): the irreps dims of input arguments. Default: ``None``.
        enable_mix_precision (bool): whether use mix precision. Default: ``False``.

    Inputs:
        - **embedding_out** (tuple(Tensor)) - Tuple of tensor.
        - **edge_index** (Tensor) - The shape of Tensor is :math:`(2, edge\_num)`.
        - **atom_types** (Tensor) - Tensor.

    Outputs:
        - **output** (Tensor) - The shape of Tensor is :math:`(edge\_num, final\_latent\_out)`.

    Raises:
        ValueError: If irreps_in is None.
        ValueError: If required fields not in irreps_in.
        ValueError: If wrong mul in input_irreps.
        ValueError: If env_embed_irreps not start with scalars.
        ValueError: If new_tps_irreps not have equal length with tps_irreps.
        ValueError: If order of tps_irreps not zero.
        ValueError: If order of full_out_irreps not zero.
        ValueError: If order of out_irreps not zero.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import os
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import context, Tensor
        >>> from mindchemistry.cell.allegro import Allegro
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> allegro_model = Allegro(
        ...     l_max=3,
        ...     irreps_in={'pos': '1x1o', 'edge_index': None, 'node_attrs': '4x0e', 'node_features': '4x0e',
        ...                 'edge_embedding': '8x0e'},
        ...     avg_num_neighbor=11.0,
        ...     num_layers=3,
        ...     env_embed_multi=128,
        ...     two_body_kwargs={'hidden_dims': [128, 256, 512, 1024], 'activation': 'silu', 'weight_init': 'uniform'},
        ...     latent_kwargs={'hidden_dims': [1024, 1024, 1024], 'activation': 'silu', 'weight_init': 'uniform'},
        ...     env_embed_kwargs={'hidden_dims': [], 'activation': None, 'weight_init': 'uniform'}
        ...     )
        >>> edges = 660
        >>> final_latent_out = 1024
        >>> embedding_out = (
        ...     Tensor(np.random.rand(60, 4), ms.float32),
        ...     Tensor(np.random.rand(60, 4), ms.float32),
        ...     Tensor(np.random.rand(660, 3), ms.float32),
        ...     Tensor(np.random.rand(660), ms.float32),
        ...     Tensor(np.random.rand(660, 8), ms.float32),
        ...     Tensor(np.random.rand(660), ms.float32),
        ...     Tensor(np.ones(660), ms.bool_)
        ...     )
        >>> edge_index = Tensor(np.ones((2, 660)), ms.int32)
        >>> atom_types = Tensor(np.ones((60, 1)), ms.int32)
        >>> out = allegro_model(embedding_out, edge_index, atom_types)
        >>> print(out.shape)
        (660, 1024)
    """

    # pylint: disable=W0102
    # pylint: disable=C0111
    def __init__(
            self,
            l_max: int = 1,
            parity_setting="o3_full",
            num_layers: int = 1,
            env_embed_multi: int = 8,
            avg_num_neighbor: float = 1.0,
            two_body_kwargs=None,
            latent_kwargs=None,
            env_embed_kwargs=None,
            irreps_in=None,
            enable_mix_precision=False,
    ):
        literal_hidden_dims = 'hidden_dims'
        literal_activation = 'activation'
        literal_weight_init = 'weight_init'
        literal_wrong_irreps = "wrong irreps"

        super().__init__()
        scalar = Irrep("0e")
        self.num_layers = num_layers
        self.env_embed_mul = env_embed_multi
        self.scatter = AggregateEdgeToNode(dim=1)
        self.lift = LiftNodeToEdge(dim=1)
        self.enable_mix_precision = enable_mix_precision
        if self.enable_mix_precision:
            self.mlp_dtype = float16
            self.linear_dtype = float16
            self.contract_dtype = float16
        else:
            self.mlp_dtype = float32
            self.linear_dtype = float32
            self.contract_dtype = float32

        # need modified
        self.field = "edge_attrs"
        self.latent_out_field = "edge_features"
        self.edge_invariant_field = "edge_embedding"
        self.node_invariant_field = "node_attrs"
        self.embed_initial_edge = True
        self.env_sum_norm = Tensor([avg_num_neighbor] * num_layers).rsqrt()
        self.linear_after_env_embed = False
        self.nonscalars_include_parity = True
        self.latent_resnet = True

        # add field of edge_attrs
        irreps_edge_sh = repr(Irreps.spherical_harmonics(l_max, p=(1 if parity_setting == "o3" else -1)))
        irreps_in.update({self.field: irreps_edge_sh})

        # init SphericalHarmonics
        self.sh = SphericalHarmonics(irreps_edge_sh, normalize=True, normalization="component")

        # irreps check
        self.irreps_in = irreps_in
        if irreps_in is None:
            raise ValueError(literal_wrong_irreps)
        required_irreps_in = [self.field, self.edge_invariant_field, self.node_invariant_field]
        for k in required_irreps_in:
            if k not in self.irreps_in:
                raise ValueError(f"Field {k} need to be in irreps_in!")

        # layers
        self.latents = CellList([])
        self.env_embed_mlps = CellList([])
        self.tps = CellList([])
        self.linears = CellList([])
        self.env_linears = CellList([])

        # check the irreps of field
        input_irreps = Irreps(self.irreps_in[self.field])
        if not all(mul == 1 for mul, ir in input_irreps.data):
            raise ValueError("wrong mul in input_irreps")
        env_embed_irreps = Irreps([(1, ir) for _, ir in input_irreps.data])
        if env_embed_irreps.data[0].ir != scalar:
            raise ValueError("env_embed_irreps must start with scalars")

        if self.embed_initial_edge:
            arg_irreps = env_embed_irreps
        else:
            arg_irreps = input_irreps

        # init irreps for the tps
        tps_irreps = [arg_irreps]
        # create irreps for other layers
        self.create_irreps(num_layers, env_embed_irreps, arg_irreps, tps_irreps)

        # remove unneed paths
        out_irreps = tps_irreps[-1]
        new_tps_irreps = [out_irreps]
        self.remove_unneed_paths(env_embed_irreps, tps_irreps, out_irreps, new_tps_irreps)

        if len(new_tps_irreps) != len(tps_irreps):
            raise ValueError(literal_wrong_irreps)
        tps_irreps = list(reversed(new_tps_irreps))
        del new_tps_irreps

        if tps_irreps[-1].lmax != 0:
            raise ValueError(literal_wrong_irreps)

        tps_irreps_in = tps_irreps[:-1]
        tps_irreps_out = tps_irreps[1:]
        del tps_irreps

        # Environment build
        self._env_weighter = MakeWeightedChannels(irreps_in=input_irreps, multiplicity_out=env_embed_multi)

        # build TP
        self._n_scalar_outs = []
        # pylint: disable=R1702
        for layer_idx, (arg_irreps, out_irreps) in enumerate(zip(tps_irreps_in, tps_irreps_out)):
            # env embed linear
            if self.linear_after_env_embed:
                self.env_linears.append(
                    Linear(
                        [(self.env_embed_mul, ir) for _, ir in env_embed_irreps.data],
                        [(self.env_embed_mul, ir) for _, ir in env_embed_irreps.data],
                    )
                )
            else:
                self.env_linears.append(Identity())

            # make TP
            n_scalar_outs = 0
            full_out_irreps = []
            instr = []
            tmp_i_out = 0
            for _, (_, ir_out) in enumerate(out_irreps.data):
                for i_1, (_, ir_1) in enumerate(arg_irreps.data):
                    for i_2, (_, ir_2) in enumerate(env_embed_irreps.data):
                        if ir_out in ir_1 * ir_2:
                            if ir_out == scalar:
                                n_scalar_outs += 1
                            full_out_irreps.append((env_embed_multi, ir_out))
                            instr.append((i_1, i_2, tmp_i_out))
                            tmp_i_out += 1
            full_out_irreps = Irreps(full_out_irreps)
            self._n_scalar_outs.append(n_scalar_outs)
            if not all(ir == scalar for _, ir in full_out_irreps.data[:n_scalar_outs]):
                raise ValueError(literal_wrong_irreps)

            # make tp
            tp = Contractor(
                irreps_in1=Irreps([(
                    (env_embed_multi if layer_idx > 0 or self.embed_initial_edge else 1),
                    ir,
                ) for _, ir in arg_irreps]),
                irreps_in2=Irreps([(env_embed_multi, ir) for _, ir in env_embed_irreps]),
                irreps_out=Irreps([(env_embed_multi, ir) for _, ir in full_out_irreps]),
                shared_weights=False,
                has_weight=False,
                connection_mode=("uuu" if layer_idx > 0 or self.embed_initial_edge else "uvv"),
                pad_to_alignment=1,
                sparse_mode=None,
                instr=instr,
                normalization="component",
                dtype=self.contract_dtype,
            )
            self.tps.append(tp)

            if out_irreps.data[0].ir != scalar:
                raise ValueError(literal_wrong_irreps)

            # make env embed mlp
            generate_n_weights = (self._env_weighter.weight_numel)
            if layer_idx == 0 and self.embed_initial_edge:
                generate_n_weights += self._env_weighter.weight_numel

            # make linear
            linear = Allegro_Linear(
                irreps_in=full_out_irreps,
                irreps_out=[(env_embed_multi, ir) for _, ir in out_irreps],
                instructions=None,
                pad_to_alignment=1,
                dtype=self.linear_dtype
            )
            self.linears.append(linear)

            if layer_idx == 0:
                self.latents.append(
                    MLP(
                        input_dim=2 * Irreps(self.irreps_in[self.node_invariant_field]).num_irreps +
                        Irreps(self.irreps_in[self.edge_invariant_field]).num_irreps,
                        hidden_dims=two_body_kwargs.get(literal_hidden_dims, None),
                        activation_fn=two_body_kwargs.get(literal_activation, None),
                        weight_init=two_body_kwargs.get(literal_weight_init, None),
                        has_bias=False,
                        dtype=self.mlp_dtype
                    )
                )
            else:
                self.latents.append(
                    MLP(
                        input_dim=(self.latents[-1].dims[-1] + env_embed_multi * n_scalar_outs),
                        hidden_dims=latent_kwargs.get(literal_hidden_dims, None),
                        activation_fn=latent_kwargs.get(literal_activation, None),
                        weight_init=latent_kwargs.get(literal_weight_init, None),
                        has_bias=False,
                        dtype=self.mlp_dtype
                    )
                )

            self.env_embed_mlps.append(
                MLP(
                    input_dim=self.latents[-1].dims[-1],
                    hidden_dims=env_embed_kwargs.get(literal_hidden_dims, None) + [generate_n_weights],
                    activation_fn=env_embed_kwargs.get(literal_activation, None),
                    weight_init=env_embed_kwargs.get(literal_weight_init, None),
                    has_bias=False,
                    dtype=self.mlp_dtype
                )
            )

        self.final_latent = MLP(
            input_dim=self.latents[-1].dims[-1] + env_embed_multi * n_scalar_outs,
            hidden_dims=latent_kwargs.get(literal_hidden_dims, None),
            activation_fn=latent_kwargs.get(literal_activation, None),
            weight_init=latent_kwargs.get(literal_weight_init, None),
            has_bias=False,
            dtype=self.mlp_dtype
        )
        self.latent_dim = self.final_latent.dims[-1]
        # -- end build modules --

        # initialization
        self._zero = ops.Zeros()
        self._ones = ops.Ones()

        # update weights of resnet layer
        self._latent_resnent_update_params = self._zero(self.num_layers, ms.common.dtype.float32)

    def remove_unneed_paths(self, env_embed_irreps, tps_irreps, out_irreps, new_tps_irreps):
        for arg_irreps in reversed(tps_irreps[:-1]):
            new_arg_irreps = []
            for mul, arg_ir in arg_irreps.data:
                for _, env_ir in env_embed_irreps.data:
                    if any(i in out_irreps for i in arg_ir * env_ir):
                        new_arg_irreps.append((mul, arg_ir))
                        break
            new_arg_irreps = Irreps(new_arg_irreps)
            new_tps_irreps.append(new_arg_irreps)
            out_irreps = new_arg_irreps

    def create_irreps(self, num_layers, env_embed_irreps, arg_irreps, tps_irreps):
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                ir_out = []
                for (_, ir) in env_embed_irreps.data:
                    if self.nonscalars_include_parity:
                        ir_out.append((1, (ir.l, 1)))
                        ir_out.append((1, (ir.l, -1)))
                    else:
                        ir_out.append((1, ir))
                ir_out = Irreps(ir_out)

            if layer_idx == self.num_layers - 1:
                ir_out = Irreps([(1, (0, 1))])

            ir_out = Irreps([(mul, ir) for mul, ir in ir_out.data if tp_path_exists(arg_irreps, env_embed_irreps, ir)])

            arg_irreps = ir_out
            tps_irreps.append(ir_out)

    def construct(self, embedding_data, edge_index, atom_types):
        """construct
        """
        node_emb, _, edge_vec, _, edge_embedding, edge_cutoff, edge_mask = embedding_data

        edge_cutoff = edge_cutoff.unsqueeze(-1)
        edge_mask = edge_mask.unsqueeze(-1).astype(edge_vec.dtype)

        edge_center = edge_index[1]
        edge_neighbor = edge_index[0]
        edge_attrs = self.sh(edge_vec)

        # mask edge_attr
        edge_attrs = edge_attrs * edge_mask
        features = edge_attrs
        num_edges = edge_attrs.shape[0]

        # pre declare
        scalars = self._zero((1,), edge_attrs.dtype)
        coefficient_old = scalars
        coefficient_new = scalars

        # initialize state
        latents = self._zero((num_edges, self.latent_dim), edge_attrs.dtype)
        latent_inputs = [node_emb[edge_center], node_emb[edge_neighbor], edge_embedding]
        layer_update_coefficients = ops.sigmoid(self._latent_resnent_update_params)

        # layer
        layer_index = 0
        for latent, env_embed_mlp, env_linear, tp, linear in zip(
                self.latents, self.env_embed_mlps, self.env_linears, self.tps, self.linears
        ):
            latent_concat = ops.concat(latent_inputs, axis=-1)
            latent_mask = latent_concat * edge_mask

            new_latents = latent(latent_mask)
            new_latents = edge_cutoff * new_latents

            if self.latent_resnet and layer_index > 0:
                this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
                coefficient_old = ops.rsqrt(this_layer_update_coeff.square() + Tensor(1.0))
                coefficient_new = this_layer_update_coeff * coefficient_old
                latents = coefficient_old * latents + coefficient_new * new_latents
            else:
                latents = new_latents

            # embedding MLP
            weights = env_embed_mlp(latents)
            w_index = 0
            if self.embed_initial_edge and layer_index == 0:
                env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                w_index += self._env_weighter.weight_numel
                features = self._env_weighter(features, env_w)

            # build the local environments
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel
            local_weight_sum = self._env_weighter(edge_attrs, env_w)
            tmp_shape = list(local_weight_sum.shape)
            tmp_shape[0] = node_emb.shape[0]
            tmp = self._zero(tuple(tmp_shape), local_weight_sum.dtype)
            local_env_per_edge = self.scatter(edge_attr=local_weight_sum, edge_index=edge_index, out=tmp)

            if self.env_sum_norm.ndim < 2:
                norm_const = self.env_sum_norm[layer_index]
            else:
                norm_const = self.env_sum_norm[layer_index, atom_types]
                norm_const = norm_const.unsqueeze(-1)
            local_env_per_edge = local_env_per_edge * norm_const
            local_env_per_edge = env_linear(local_env_per_edge)
            local_env_per_edge = self.lift(local_env_per_edge, edge_index)

            # do the tp
            features = tp(features, local_env_per_edge)

            # get invariants
            scalars = features[:, :, :self._n_scalar_outs[layer_index]].reshape(features.shape[0], -1)

            # do the linear
            features = linear(features)

            # concat latents and scalars
            latent_inputs = [latents, scalars]

            layer_index += 1

        latent_concat = ops.concat(latent_inputs, axis=-1)
        latent_mask = latent_concat * edge_mask
        new_latents = self.final_latent(latent_mask)
        new_latents = edge_cutoff * new_latents

        if self.latent_resnet:
            this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
            coefficient_old = ops.rsqrt(this_layer_update_coeff.square() + Tensor(1.0))
            coefficient_new = this_layer_update_coeff * coefficient_old
            latents = coefficient_old * latents + coefficient_new * new_latents
        else:
            latents = new_latents

        edge_features = latents

        return edge_features
