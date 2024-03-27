# Copyright 2022 Huawei Technologies Co., Ltd
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
r"""PDEformer model."""
from omegaconf import DictConfig
from mindspore import dtype as mstype
from mindspore import nn, Tensor, ops

from .inr_with_hypernet import SirenWithHypernet, PolyINRWithHypernet, MFNNetWithHypernet
from .graphormer.graphormer_encoder import GraphormerEncoder
from ..basic_block import MLP
from .function_encoder import DeepSetFuncEncoder, WeightedDeepSetFuncEncoder, PatchedFuncEncoder


class PDEEncoder(nn.Cell):
    r"""
    PDEEncoder is used for encoding the input graph and function into a fixed-size representation.
    It consists of a GraphormerEncoder, a scalar encoder and a function encoder.
    The GraphormerEncoder encodes the PDE formulation into a fixed-size representation,
    and the scalar encoder and function encoder encode the scalar and function information
    into a fixed-size representation.

    Args:
        config_model (Dict): Configurations.
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
        - **node_type** (Tensor) - The type of each node, shape :math:`(n\_graph, n\_node, 1)`.
        - **node_scalar** (Tensor) - The scalar value of each node, shape :math:`(n\_graph, num\_scalar, 1)`.
        - **node_function** (Tensor) - The function value of each node,
          shape :math:`(n\_graph, num\_function, num\_points\_function, 2)`.
        - **in_degree** (Tensor) - The in-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **out_degree** (Tensor) - The out-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **attn_bias** (Tensor) - The attention bias of the graph, shape :math:`(n\_graph, n\_node, n\_node)`.
        - **spatial_pos** (Tensor) - The spatial position from each node to each other node,
          shape :math:`(n\_graph, n\_node, n\_node)`.

    Outputs:
        The output representation of teh PDE, shape :math:`(n\_node, n\_graph, embed\_dim)`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, config_model: DictConfig, compute_dtype=mstype.float16) -> None:
        super().__init__()

        graphormer_config = config_model.graphormer
        self.graphormer = GraphormerEncoder(
            num_node_type=graphormer_config["num_node_type"],
            num_in_degree=graphormer_config["num_in_degree"],
            num_out_degree=graphormer_config["num_out_degree"],
            num_spatial=graphormer_config["num_spatial"],
            num_encoder_layers=graphormer_config["num_encoder_layers"],
            embed_dim=graphormer_config["embed_dim"],
            ffn_embed_dim=graphormer_config["ffn_embed_dim"],
            num_heads=graphormer_config["num_heads"],
            pre_layernorm=graphormer_config["pre_layernorm"],
            compute_dtype=compute_dtype
        )

        self.scalar_encoder = MLP(
            1, graphormer_config["embed_dim"],
            dim_hidden=config_model.scalar_encoder.dim_hidden,
            num_layers=config_model.scalar_encoder.num_layers,
            compute_dtype=compute_dtype)

        function_encoder_type = config_model.function_encoder.type.lower()
        if function_encoder_type == "deepset":
            self.function_encoder = DeepSetFuncEncoder(
                2, graphormer_config["embed_dim"] * config_model.function_encoder.num_branches,
                config_model.function_encoder.dim_hidden,
                config_model.function_encoder.num_layers,
                config_model.function_encoder.deepset_point_fn.lower(),
                compute_dtype=compute_dtype)
        elif function_encoder_type == "weighted_deepset":
            self.function_encoder = WeightedDeepSetFuncEncoder(
                2, graphormer_config["embed_dim"] * config_model.function_encoder.num_branches,
                config_model.function_encoder.dim_hidden,
                config_model.function_encoder.num_layers,
                config_model.function_encoder.deepset_point_fn.lower(),
                compute_dtype=compute_dtype)
        elif function_encoder_type == "patched":
            patch_len, residual = divmod(256, config_model.function_encoder.num_branches)
            if residual != 0:
                raise ValueError(
                    f"'num_branches' ({config_model.function_encoder.num_branches}) "
                    "should be a factor of 256.")
            self.function_encoder = PatchedFuncEncoder(
                2, graphormer_config["embed_dim"],
                config_model.function_encoder.dim_hidden,
                config_model.function_encoder.num_layers,
                patch_len,
                compute_dtype=compute_dtype)
        else:
            raise NotImplementedError(
                "function_encoder_type should be in ['deepset', 'weighted_deepset', 'patched'], "
                f"but got {config_model.function_encoder.type}")

    def construct(self,
                  node_type: Tensor,
                  node_scalar: Tensor,
                  node_function: Tensor,
                  in_degree: Tensor,
                  out_degree: Tensor,
                  attn_bias: Tensor,
                  spatial_pos: Tensor) -> Tensor:
        r"""construct"""
        node_scalar_feature = self.scalar_encoder(node_scalar)  # [n_graph, num_scalar, embed_dim]

        (n_graph, num_function, num_points_function, _) = node_function.shape
        node_function = node_function.reshape(n_graph * num_function, num_points_function, 2)
        # Tensor shape: [n_graph*num_function, num_branches*embed_dim] for deepset,
        # [n_graph*num_function*num_branches, embed_dim] for patched.
        node_function_feature = self.function_encoder(node_function)
        node_function_feature = node_function_feature.reshape(
            n_graph, -1, node_scalar_feature.shape[-1])  # [n_graph, num_function*num_branches, embed_dim]

        # Shape is [n_graph, num_scalar+num_function*num_branches, embed_dim].
        node_input_feature = ops.cat((node_scalar_feature, node_function_feature), axis=1)

        out = self.graphormer(node_type, node_input_feature, in_degree,
                              out_degree, attn_bias, spatial_pos)  # [n_node, n_graph, embed_dim]

        return out  # [n_node, n_graph, embed_dim]


class INRWithHypernet(nn.Cell):
    r"""
    INRWithHypernet is used for representing the solution of the PDE equation at each point.
    It consists of an INR and a Hypernet. The Hypernet takes the PDE feature as input,
    and outputs the modulations to each INR  hidden layer. The INR takes the modulated PDE
    feature and the coordinate of each point as input, and outputs the solution of the PDE
    equation at each point.

    Args:
        config_model (Dict): Configurations.
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
         - **coordination** (Tensor) - The coordinate of each point, shape :math:`(n\_graph, num\_points, 2)`.
         - **hyper_input** (Tensor) - The PDE feature, shape :math:`(n\_inr_node, n\_graph, embed\_dim)`.

    Outputs:
        The solution of the PDE equation at each point, shape :math:`(n\_graph, num\_points, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, config_model: DictConfig, compute_dtype=mstype.float16) -> None:
        super().__init__()
        embed_dim = config_model.graphormer.embed_dim

        # INR with hypernet
        base_kwargs = dict(
            inr_dim_in=config_model.inr.dim_in,
            inr_dim_out=config_model.inr.dim_out,
            inr_dim_hidden=config_model.inr.dim_hidden,
            inr_num_layers=config_model.inr.num_layers,
            hyper_dim_in=embed_dim,
            compute_dtype=compute_dtype,
        )

        # keys contained: hyper_dim_hidden, hyper_num_layers, share_hypernet
        hypernet_kwargs = config_model.hypernet
        inr_type = config_model.inr.type.lower()
        if inr_type == "siren":
            self.inr_with_hypernet = SirenWithHypernet(
                **base_kwargs,
                **hypernet_kwargs,
                num_pos_enc=config_model.inr.siren.num_pos_enc,
                enable_scale=config_model.inr.siren.enable_scale,
            )
        elif inr_type == "mfn":
            self.inr_with_hypernet = MFNNetWithHypernet(
                **base_kwargs,
                **hypernet_kwargs,
                **config_model.inr.mfn,
            )
        elif inr_type == "poly_inr":
            self.inr_with_hypernet = PolyINRWithHypernet(
                **base_kwargs,
                **hypernet_kwargs,
                **config_model.inr.poly_inr,
            )
        else:
            raise ValueError(
                f"inr_type should be in ['siren','mfn', 'poly_inr'], but got {config_model.inr.type}")

    def construct(self,
                  coordinate: Tensor,
                  hyper_in: Tensor) -> Tensor:
        r"""construct"""
        out = self.inr_with_hypernet(coordinate, hyper_in)  # [n_graph, num_points, dim_out]
        return out


class PDEformer(nn.Cell):
    r"""
    PDEformer consists of a PDEEncoder and an INRWithHypernet. The PDEEncoder encodes the PDE into
    a fixed-size representation, and the INRWithHypernet represents the solution of the PDE equation
    at each point. The PDEformer takes the PDE formulation, the scalar and function information as input,
    and outputs the fixed-size representation of the PDE. In addition, the PDE formulation is represented
    by the graph structure. The INRWithHypernet takes the fixed-size representation of the PDE and the
    coordinate of each point as input, and outputs the solution of the PDE equation at each point.

    Args:
        config_model (Dict): Configurations.
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
        - **node_type** (Tensor) - The type of each node, shape :math:`(n\_graph, n\_node, 1)`.
        - **node_scalar** (Tensor) - The scalar value of each node, shape :math:`(n\_graph, num\_scalar, 1)`.
        - **node_function** (Tensor) - The function value of each node,
          shape :math:`(n\_graph, num\_function, num\_points\_function, 2)`.
        - **in_degree** (Tensor) - The in-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **out_degree** (Tensor) - The out-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **attn_bias** (Tensor) - The attention bias of the graph, shape :math:`(n\_graph, n\_node, n\_node)`.
        - **spatial_pos** (Tensor) - The spatial position from each node to each other node,
          shape :math:`(n\_graph, n\_node, n\_node)`.
        - **coordinate** (Tensor) - The coordinate of each point, shape :math:`(n\_graph, num\_points, 2)`.

    Outputs:
        The solution of the PDE equation at each point, shape :math:`(n\_graph, num\_points, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, config_model: DictConfig, compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.n_inr_nodes = config_model.inr.num_layers - 1

        self.pde_encoder = PDEEncoder(config_model, compute_dtype=compute_dtype)
        self.inr = INRWithHypernet(config_model, compute_dtype=compute_dtype)

    def construct(self,
                  node_type: Tensor,
                  node_scalar: Tensor,
                  node_function: Tensor,
                  in_degree: Tensor,
                  out_degree: Tensor,
                  attn_bias: Tensor,
                  spatial_pos: Tensor,
                  coordinate: Tensor) -> Tensor:
        r"""construct"""
        pde_feature = self.pde_encoder(node_type, node_scalar, node_function, in_degree,
                                       out_degree, attn_bias, spatial_pos)

        # Shape is [n_graph, num_points, dim_out].
        out = self.inr(coordinate, pde_feature[:self.n_inr_nodes])
        return out
