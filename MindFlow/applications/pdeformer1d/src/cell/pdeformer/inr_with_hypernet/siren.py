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
r"""Siren model."""
from typing import Optional
import math

from mindspore import dtype as mstype
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import initializer, Uniform

from ...basic_block import MLP, CoordPositionalEncoding, Sine


class Siren(nn.Cell):
    r"""
    SIREN model. For SIREN's details, please refer to: https://www.vincentsitzmann.com/siren/.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        omega0 (float, optional): Omega 0 from SIREN paper. Default: ``30.0``.
        omega0_initial (float, optional): Omega 0 for first layer. Default: ``30.0``.
        weight_scale (float, optional): "c" value from SIREN paper used for weight initialization. Default: ``6.0``.
        num_pos_enc (int, optional): Number of positional embedding frequencies for the coordinate input.
            Default: ``5``.
        compute_dtype (ms.dtype, optional): Floating point data type of the network. Default: ``ms.dtype.float16``.

    Inputs:
        - **x** (Tensor): Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.
        - **scale_modulations** (Tensor, optional): Tensor of shape :math:`(num\_layers - 1, batch\_size, dim\_hidden)`.
          Modulation for scaling the output of each layer. Default: ``None``.
        - **shift_modulations** (Tensor, optional): Tensor of shape :math:`(num\_layers - 1, batch\_size, dim\_hidden)`.
          Modulation for shifting the output of each layer. Default: ``None``.

    Outputs:
        Output Tensor of shape :math:`(batch\_size, num\_points, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>> from src.cell.pdeformer.inr.siren import Siren
        >>> inr_dim_in = 2
        >>> inr_dim_out = 1
        >>> inr_dim_hidden = 64
        >>> inr_num_layers = 4
        >>> siren = Siren(inr_dim_in, inr_dim_hidden, inr_dim_out, inr_num_layers, compute_dtype=ms.float32)
        >>> x = ms.Tensor(np.random.rand(16, 100, 2), ms.float32)
        >>> out = siren(x)
        >>> print(out.shape)
        (16, 100, 1)
    """

    def __init__(
            self,
            dim_in: int,
            dim_hidden: int,
            dim_out: int,
            num_layers: int,
            omega0: float = 30.0,
            omega0_initial: float = 30.0,
            weight_scale: float = 6.0,
            num_pos_enc: int = 0,
            compute_dtype=mstype.float16) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.pos_enc = CoordPositionalEncoding(num_pos_enc)

        if dim_in <= 0:
            raise ValueError(f"'dim_in' should be a positive integer, but got {dim_in}.")
        if num_pos_enc < 0:
            raise ValueError(f"'num_pos_enc' should be a non-negative integer, but got {num_pos_enc}.")
        if dim_hidden <= 0:
            raise ValueError(f"'dim_hidden' should be a positive integer, but got {dim_hidden}.")
        ext_dim_in = dim_in * (2 * num_pos_enc + 1)

        if omega0 <= 0:
            raise ValueError(f"'omega0' should be a positive number, but got {omega0}.")

        acts = []
        layers = []
        for idx in range(num_layers - 1):
            if idx == 0:  # first layer
                layers.append(nn.Dense(ext_dim_in, dim_hidden, has_bias=True).to_float(compute_dtype))
                w_std = 1 / ext_dim_in
                layers[-1].weight.set_data(initializer(Uniform(w_std), layers[-1].weight.shape, compute_dtype))
                layers[-1].bias.set_data(initializer(Uniform(w_std), layers[-1].bias.shape, compute_dtype))
                acts.append(Sine(omega0_initial))
            else:
                layers.append(nn.Dense(dim_hidden, dim_hidden, has_bias=True).to_float(compute_dtype))
                w_std = math.sqrt(weight_scale / dim_hidden) / omega0
                layers[-1].weight.set_data(initializer(Uniform(w_std), layers[-1].weight.shape, compute_dtype))
                layers[-1].bias.set_data(initializer(Uniform(w_std), layers[-1].bias.shape, compute_dtype))
                acts.append(Sine(omega0))
        self.layers = nn.CellList(layers)
        self.acts = nn.CellList(acts)

        self.last_layer = nn.Dense(dim_hidden, dim_out, has_bias=True).to_float(compute_dtype)
        w_std = math.sqrt(weight_scale / dim_hidden) / omega0
        self.last_layer.weight.set_data(initializer(Uniform(w_std), self.last_layer.weight.shape, compute_dtype))
        self.last_layer.bias.set_data(initializer(Uniform(w_std), self.last_layer.bias.shape, compute_dtype))

    def construct(self,
                  x: Tensor,
                  scale_modulations: Optional[Tensor] = None,
                  shift_modulations: Optional[Tensor] = None) -> Tensor:
        r"""construct"""
        x = self.pos_enc(x)  # [bsz, n_pts, dim_in] -> [bsz, n_pts, ext_dim_in]

        for layer_idx in range(self.num_layers - 1):
            if scale_modulations is None:
                scale = 1.
            else:
                scale = 1. + scale_modulations[layer_idx].expand_dims(1)  # [batch_size, 1, dim_hidden]

            if shift_modulations is None:
                shift = 0.
            else:
                shift = shift_modulations[layer_idx].expand_dims(1)  # [batch_size, 1, dim_hidden]

            residual = x
            x = self.layers[layer_idx](x)  # [batch_size, num_points, dim_hidden]
            x = scale * x + shift
            x = self.acts[layer_idx](x)  # [batch_size, num_points, dim_hidden]

            if layer_idx > 0:
                x = x + residual  # residual connection

        out = self.last_layer(x)  # [batch_size, num_points, dim_out]

        return out


class SirenWithHypernet(nn.Cell):
    r"""
    SIREN with hypernets.

    Args:
        inr_dim_in (int): Dimension of coordinate input.
        inr_dim_out (int): Dimension of SIREN output.
        inr_dim_hidden (int): Dimension of SIREN hidden layers.
        inr_num_layers (int): Number of SIREN layers.
        hyper_dim_in (int): Dimension of each hypernet input.
        hyper_dim_hidden (int): Dimension of hidden layers of the hypernet.
        hyper_num_layers (int): Number of layers of the hypernet.
        share_hypernet (bool, optional): Whether the modulations of all SIREN hidden layers are
            generated by the same modulation encoder. Default: ``True``.
        enable_scale (bool, optional): Whether to introduce scale modulations of the SIREN network.
            Default: ``False``, only generate shift modulations.
        num_pos_enc (int, optional): Number of positional embedding frequencies for the coordinate input.
            Default: ``0``.
        compute_dtype (ms.dtype, optional): Floating point data type of the network. Default: ``ms.dtype.float16``.

    Inputs:
        - **coordinate** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.
        - **hyper_in** (Tensor) - Tensor of shape :math:`(inr\_num\_layers - 1, batch\_size, hyper\_dim\_in)`.

    Outputs:
        Tensor of shape :math:`(batch\_size, num\_points, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>> from src.cell.pdeformer.inr.siren import SirenWithHypernet
        >>> inr_dim_in = 2
        >>> inr_dim_out = 1
        >>> inr_dim_hidden = 64
        >>> inr_num_layers = 4
        >>> hyper_dim_in = 16
        >>> hyper_dim_hidden = 32
        >>> hyper_num_layers = 2
        >>> siren_with_hypernet = SirenWithHypernet(inr_dim_in, inr_dim_out, inr_dim_hidden, inr_num_layers,
        >>>                                         hyper_dim_in, hyper_dim_hidden, hyper_num_layers,
        >>>                                         enable_scale=True, compute_dtype=ms.float32)
        >>> bsz = 16
        >>> n_pts = 100
        >>> coordinate = ms.Tensor(np.random.rand(bsz, n_pts, 2), ms.float32)
        >>> hyper_in = ms.Tensor(np.random.rand(inr_num_layers - 1, bsz, hyper_dim_in), ms.float32)
        >>> out = siren_with_hypernet(coordinate, hyper_in)
        >>> print(out.shape)
        (16, 100, 1)
    """

    def __init__(
            self,
            inr_dim_in: int,
            inr_dim_out: int,
            inr_dim_hidden: int,
            inr_num_layers: int,
            hyper_dim_in: int,
            hyper_dim_hidden: int,
            hyper_num_layers: int,
            share_hypernet: bool = True,
            enable_scale: bool = False,
            num_pos_enc: int = 0,
            compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.inr_num_layers = inr_num_layers
        self.enable_scale = enable_scale
        self.share_modulation_encoder = share_hypernet

        self.inr = Siren(
            dim_in=inr_dim_in,
            dim_hidden=inr_dim_hidden,
            dim_out=inr_dim_out,
            num_layers=inr_num_layers,
            num_pos_enc=num_pos_enc,
            compute_dtype=compute_dtype,
        )

        hyper_dim_out = inr_dim_hidden

        def new_hypernet_mlp():
            return MLP(hyper_dim_in, hyper_dim_out, hyper_dim_hidden,
                       hyper_num_layers, compute_dtype)

        if self.share_modulation_encoder:
            self.shift_modulation_encoder = new_hypernet_mlp()
            if self.enable_scale:
                self.scale_modulation_encoder = new_hypernet_mlp()
        else:
            num_hypernet = inr_num_layers - 1  # the number of hidden layers
            self.shift_modulation_encoders = nn.CellList([
                new_hypernet_mlp() for _ in range(num_hypernet)])
            if self.enable_scale:
                self.scale_modulation_encoders = nn.CellList([
                    new_hypernet_mlp() for _ in range(num_hypernet)])

    def construct(self, coordinate: Tensor, hyper_in: Tensor) -> Tensor:
        r"""construct"""
        if self.enable_scale:
            scale_modulations = []
            for idx in range(self.inr_num_layers - 1):
                encoder_in = hyper_in[idx]  # [n_graph, embed_dim]
                if self.share_modulation_encoder:
                    encoder_out = self.scale_modulation_encoder(encoder_in)
                else:
                    encoder_out = self.scale_modulation_encoders[idx](encoder_in)
                scale_modulations.append(encoder_out)

            scale_modulations = ops.stack(scale_modulations, axis=0)  # [inr_num_layers - 1, n_graph, embed_dim]
        else:
            scale_modulations = None

        shift_modulations = []
        for idx in range(self.inr_num_layers - 1):
            encoder_in = hyper_in[idx]  # [n_graph, embed_dim]
            if self.share_modulation_encoder:
                encoder_out = self.shift_modulation_encoder(encoder_in)
            else:
                encoder_out = self.shift_modulation_encoders[idx](encoder_in)
            shift_modulations.append(encoder_out)

        shift_modulations = ops.stack(shift_modulations, axis=0)  # [inr_num_layers - 1, n_graph, embed_dim]

        out = self.inr(coordinate, scale_modulations, shift_modulations)  # [n_graph, num_points, dim_out]
        return out
