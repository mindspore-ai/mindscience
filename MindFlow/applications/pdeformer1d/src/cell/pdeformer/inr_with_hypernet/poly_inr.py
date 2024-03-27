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
r"""PolyINR model."""
from typing import Optional

from mindspore import dtype as mstype
from mindspore import Tensor, nn, ops

from ...basic_block import MLP, UniformInitDense


class Clamp(nn.Cell):
    """Crop values within a fixed range."""

    def __init__(self, threshold=256.) -> None:
        super().__init__()
        if threshold <= 0.:
            raise ValueError(f"'threshold' ({threshold}) should be positive.")
        self.threshold = threshold

    def construct(self, x: Tensor) -> Tensor:
        """construct"""
        return ops.clamp(x, min=-self.threshold, max=self.threshold)


class PolyINR(nn.Cell):
    r"""
    PolyINR is a implicit neural representation (INR) architecture.
    For details, please refer to paper: https://arxiv.org/abs/2303.11424

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of the hidden features.
        num_layers (int): Number of layers.
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.
        - **scale_modulations** (Tensor, optional) - Tensor of shape :math:`(num\_layers-1, batch\_size, dim\_hidden)`.
        - **shift_modulations** (Tensor, optional) - Tensor of shape :math:`(num\_layers-1, batch\_size, dim\_hidden)`.

    Outputs:
        - **out** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_hidden)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import PolyINR
        >>> x = Tensor(np.random.randn(2, 10, 3), mstype.float32)
        >>> poly_inr = PolyINR(3, 64, 128, 2, compute_dtype=mstype.float32)
        >>> out = poly_inr(x)
        >>> print(out.shape)
        (2, 10, 64)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.coord_pad = nn.ConstantPad1d((0, 1), 1.0)
        self.affines = nn.CellList([UniformInitDense(dim_in, dim_hidden).to_float(compute_dtype)
                                    for _ in range(num_layers - 1)])
        self.act = nn.SequentialCell(nn.LeakyReLU(0.2), Clamp(256.))
        self.dense_layers = nn.CellList([UniformInitDense(dim_hidden, dim_hidden).to_float(compute_dtype)
                                         for _ in range(num_layers - 1)])
        self.last_layer = UniformInitDense(dim_hidden, dim_out).to_float(compute_dtype)

    def construct(self,
                  x: Tensor,  # [bsz, num_points, dim_in]
                  affine_modulations: Optional[Tensor] = None,  # [num_layers-1, bsz, dim_in+1, dim_hidden]
                  scale_modulations: Optional[Tensor] = None,  # [num_layers-1, bsz, dim_hidden]
                  shift_modulations: Optional[Tensor] = None,  # [num_layers-1, bsz, dim_hidden]
                  ) -> Tensor:
        r"""construct"""
        hidden_state = 1.

        x_pad = self.coord_pad(x)  # [bsz, n_pts, dim_in] -> [bsz, n_pts, dim_in+1]

        for layer_idx in range(self.num_layers - 1):
            if scale_modulations is None:
                scale = 1.0
            else:
                scale = 1. + scale_modulations[layer_idx].expand_dims(1)  # [bsz, 1, dim_hidden]

            if shift_modulations is None:
                shift = 0.0
            else:
                shift = shift_modulations[layer_idx].expand_dims(1)  # [bsz, 1, dim_hidden]

            tmp = self.affines[layer_idx](x)  # [bsz, n_pts, dim_hidden]
            if affine_modulations is not None:
                tmp2 = ops.matmul(x_pad, affine_modulations[layer_idx])  # [bsz, n_pts, dim_hidden]
                tmp = tmp + tmp2  # [bsz, n_pts, dim_hidden]
            hidden_state = hidden_state * tmp  # [bsz, n_pts, dim_hidden]
            hidden_state = self.dense_layers[layer_idx](hidden_state)  # [bsz, n_pts, dim_hidden]
            hidden_state = scale * hidden_state + shift
            hidden_state = self.act(hidden_state)  # [bsz, n_pts, dim_hidden]

        out = self.last_layer(hidden_state)  # [bsz, n_pts, dim_out]
        return out


class PolyINRWithHypernet(nn.Cell):
    r"""
    Poly-INR model with hypernets.
    The original version proposed in paper https://arxiv.org/abs/2303.11424 contains
    only affine modulations, and this implementation includes shift and
    scale modulations as well.

    Args:
        inr_dim_in (int): Dimension of coordinate input.
        inr_dim_out (int): Dimension of PolyINR output.
        inr_dim_hidden (int): Dimension of PolyINR hidden layers.
        inr_num_layers (int): Number of PolyINR layers.
        hyper_dim_in (int): Dimension of each hypernet input.
        hyper_dim_hidden (int): Dimension of hidden layers of the hypernet.
        hyper_num_layers (int): Number of layers of the hypernet.
        share_hypernet (bool, optional): Whether the modulations of all PolyINR
            hidden layers are generated by the same modulation encoder.
            Default: ``False``.
        enable_affine (bool, optional): Whether to introduce affine modulations
            of the PolyINR network. Default: ``True``.
        enable_shift (bool, optional): Whether to introduce shift modulations
            of the PolyINR network. Default: ``False``.
        enable_scale (bool, optional): Whether to introduce scale modulations
            of the PolyINR network. Default: ``False``.
        compute_dtype (ms.dtype, optional): Floating point data type of the
            network. Default: ``ms.dtype.float16``.

    Inputs:
        - **coordinate** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.
        - **hyper_in** (Tensor) - Tensor of shape :math:`(inr\_num\_layers - 1, batch\_size, hyper\_dim\_in)`.

    Outputs:
        Tensor of shape :math:`(batch\_size, num\_points, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import PolyINRWithHypernet
        >>> inr_dim_in = 2
        >>> inr_dim_out = 1
        >>> inr_dim_hidden = 128
        >>> inr_num_layers = 6
        >>> hyper_dim_in = 16
        >>> hyper_dim_hidden = 32
        >>> hyper_num_layers = 2
        >>> poly_inr = PolyINRWithHypernet(inr_dim_in, inr_dim_out, inr_dim_hidden, inr_num_layers,
        >>>                                hyper_dim_in, hyper_dim_hidden, hyper_num_layers,
        >>>                                True, True, True, True, mstype.float32)
        >>> bsz = 32
        >>> n_pts = 128
        >>> coord = Tensor(np.random.randn(bsz, n_pts, inr_dim_in), mstype.float32)
        >>> hyper_in = Tensor(np.random.randn(inr_num_layers - 1, bsz, hyper_dim_in), mstype.float32)
        >>> out = poly_inr(coord, hyper_in)
        >>> print(out.shape)
        (32, 128, 1)
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
            share_hypernet: bool = False,
            enable_affine: bool = False,
            enable_shift: bool = True,
            enable_scale: bool = True,
            compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.inr_num_layers = inr_num_layers
        self.enable_affine = enable_affine
        self.enable_shift = enable_shift
        self.enable_scale = enable_scale
        if not (enable_affine or enable_shift or enable_scale):
            raise ValueError(
                "For 'PolyINRWithHypernet', at least one of ['enable_affine', "
                "'enable_shift', 'enable_scale'] should be True.")
        self.share_hypernet = share_hypernet
        self.affine_modulations_shape = (
            inr_num_layers - 1, -1, inr_dim_in + 1, inr_dim_hidden)

        self.inr = PolyINR(
            dim_in=inr_dim_in,
            dim_hidden=inr_dim_hidden,
            dim_out=inr_dim_out,
            num_layers=inr_num_layers,
            compute_dtype=compute_dtype
        )

        def new_hypernet_mlp(mode):
            if mode == 'affine':
                hyper_dim_out = (inr_dim_in + 1) * inr_dim_hidden
            elif mode in ['shift', 'scale']:
                hyper_dim_out = inr_dim_hidden
            return MLP(hyper_dim_in, hyper_dim_out, hyper_dim_hidden,
                       hyper_num_layers, compute_dtype)

        if self.share_hypernet:
            if enable_affine:
                self.affine_hypernet = new_hypernet_mlp('affine')
            if enable_shift:
                self.shift_hypernet = new_hypernet_mlp('shift')
            if enable_scale:
                self.scale_hypernet = new_hypernet_mlp('scale')
        else:
            num_hypernet = inr_num_layers - 1  # the number of hidden layers
            if self.enable_affine:
                self.affine_hypernets = nn.CellList([
                    new_hypernet_mlp('affine') for _ in range(num_hypernet)])
            if self.enable_shift:
                self.shift_hypernets = nn.CellList([
                    new_hypernet_mlp('shift') for _ in range(num_hypernet)])
            if self.enable_scale:
                self.scale_hypernets = nn.CellList([
                    new_hypernet_mlp('scale') for _ in range(num_hypernet)])

    def construct(self, coordinate: Tensor, hyper_in: Tensor) -> Tensor:
        r"""construct"""
        if self.enable_affine:
            affine_modulations = []
            for idx in range(self.inr_num_layers - 1):
                encoder_in = hyper_in[idx]  # [n_graph, embed_dim]
                if self.share_hypernet:
                    encoder_out = self.affine_hypernet(encoder_in)
                else:
                    encoder_out = self.affine_hypernets[idx](encoder_in)
                affine_modulations.append(encoder_out)

            # tensor shape [inr_num_layers - 1, n_graph, (dim_in + 1) * inr_dim_hidden]
            # -> [inr_num_layers - 1, n_graph, dim_in + 1, inr_dim_hidden]
            affine_modulations = ops.stack(affine_modulations, axis=0).view(self.affine_modulations_shape)
        else:
            affine_modulations = None

        if self.enable_shift:
            shift_modulations = []
            for idx in range(self.inr_num_layers - 1):
                encoder_in = hyper_in[idx]  # [n_graph, embed_dim]
                if self.share_hypernet:
                    encoder_out = self.shift_hypernet(encoder_in)
                else:
                    encoder_out = self.shift_hypernets[idx](encoder_in)
                shift_modulations.append(encoder_out)

            shift_modulations = ops.stack(shift_modulations, axis=0)  # [inr_num_layers - 1, n_graph, inr_dim_hidden]
        else:
            shift_modulations = None

        if self.enable_scale:
            scale_modulations = []
            for idx in range(self.inr_num_layers - 1):
                encoder_in = hyper_in[idx]  # [n_graph, embed_dim]
                if self.share_hypernet:
                    encoder_out = self.scale_hypernet(encoder_in)
                else:
                    encoder_out = self.scale_hypernets[idx](encoder_in)
                scale_modulations.append(encoder_out)

            scale_modulations = ops.stack(scale_modulations, axis=0)  # [inr_num_layers - 1, n_graph, inr_dim_hidden]
        else:
            scale_modulations = None

        # Shape is [n_graph, num_points, dim_out].
        out = self.inr(coordinate, affine_modulations, scale_modulations, shift_modulations)

        return out
