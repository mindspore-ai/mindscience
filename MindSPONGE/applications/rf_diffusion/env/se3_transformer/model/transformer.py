# Modified from se3-transformer-public (https://github.com/FabianFuchsML/se3-transformer-public)
# Original license: MIT License
#
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

import logging
from typing import Dict, Optional

import mindspore as ms
from mindspore import Tensor, nn
from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.layers.attention import AttentionBlockSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.norm import NormSE3
from se3_transformer.runtime.utils import str2bool
from sharker.data import Graph


class Sequential(nn.SequentialCell):
    """Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features."""

    def construct(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


def get_populated_edge_features(
    relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None
):
    """Add relative positions to existing edge features"""
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if "0" in edge_features:
        edge_features["0"] = ms.mint.cat([edge_features["0"], r[..., None]], dim=1)
    else:
        edge_features["0"] = r[..., None]

    return edge_features


class SE3Transformer(nn.Cell):
    def __init__(
        self,
        num_layers: int,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        num_heads: int,
        channels_div: int,
        fiber_edge: Fiber = Fiber({}),
        return_type: Optional[int] = None,
        norm: bool = True,
        use_layer_norm: bool = True,
        tensor_cores: bool = False,
        low_memory: bool = False,
        **kwargs
    ):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param return_type:         Return only features of this type
        :param pooling:             'avg' or 'max' graph pooling before MLP layers
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param tensor_cores:        True if using Tensor Cores (affects the use of fully fused convs, and padded bases)
        :param low_memory:          If True, will use slower ops that use less memory
        """
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.max_degree = max(
            *fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees
        )
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        if low_memory and not tensor_cores:
            logging.warning("Low memory mode will have no effect with no Tensor Cores")

        # Fully fused convolutions when using Tensor Cores (and not low memory mode)
        fuse_level = (
            ConvSE3FuseLevel.FULL
            if tensor_cores and not low_memory
            else ConvSE3FuseLevel.PARTIAL
        )

        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(
                AttentionBlockSE3(
                    fiber_in=fiber_in,
                    fiber_out=fiber_hidden,
                    fiber_edge=fiber_edge,
                    num_heads=num_heads,
                    channels_div=channels_div,
                    use_layer_norm=use_layer_norm,
                    max_degree=self.max_degree,
                    fuse_level=fuse_level,
                )
            )
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(
            ConvSE3(
                fiber_in=fiber_in,
                fiber_out=fiber_out,
                fiber_edge=fiber_edge,
                self_interaction=True,
                use_layer_norm=use_layer_norm,
                max_degree=self.max_degree,
            )
        )
        self.graph_modules = Sequential(*graph_modules)

    def construct(
        self,
        graph: Graph,
        node_feats: Dict[str, Tensor],
        edge_feats: Optional[Dict[str, Tensor]] = None,
        basis: Optional[Dict[str, Tensor]] = None,
    ):
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = basis or get_basis(
            graph.edge_rel_pos,
            max_degree=self.max_degree,
            use_pad_trick=self.tensor_cores and not self.low_memory,
        )
        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(
            basis,
            self.max_degree,
            use_pad_trick=self.tensor_cores and not self.low_memory,
            fully_fused=self.tensor_cores and not self.low_memory,
        )

        edge_feats = get_populated_edge_features(graph.edge_rel_pos, edge_feats)

        node_feats = self.graph_modules(
            node_feats, edge_feats, graph=graph, basis=basis
        )

        if self.return_type is not None:
            return node_feats[str(self.return_type)]
        return node_feats

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--num_layers",
            type=int,
            default=7,
            help="Number of stacked Transformer layers",
        )
        parser.add_argument(
            "--num_heads", type=int, default=8, help="Number of heads in self-attention"
        )
        parser.add_argument(
            "--channels_div",
            type=int,
            default=2,
            help="Channels division before feeding to attention layer",
        )
        parser.add_argument(
            "--norm",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Apply a normalization layer after each attention block",
        )
        parser.add_argument(
            "--use_layer_norm",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Apply layer normalization between MLP layers",
        )
        parser.add_argument(
            "--low_memory",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="If true, will use fused ops that are slower but that use less memory "
            "(expect 25 percent less memory). "
            "Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs",
        )

        return parser
