# ============================================================================
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
r"""Multiplicative Filter Networks."""
import math
from typing import Optional
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import ops, nn, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer, Uniform, HeUniform, One, Zero

from ...basic_block import UniformInitDense


class LayerNorm(nn.Cell):
    r"""
    Manually implemented Layer Normalization, in order to support second-order derivative.

    Args:
        dim (int): The dimension to be normalized.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, \ldots, D)` where :math:`D` is the dimension to be normalized.

    Outputs:
        Tensor of shape :math:`(N, \ldots, D)` where :math:`D` is the dimension to be normalized.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        gamma = initializer(One(), (dim,), dtype=mstype.float32)
        beta = initializer(Zero(), (dim,), dtype=mstype.float32)

        self.gamma = Parameter(gamma, name="gamma")
        self.beta = Parameter(beta, name="beta")

        self.mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.sub1 = P.Sub()
        self.sub2 = P.Sub()
        self.add = P.Add()
        self.eps = eps
        self.mul = P.Mul()
        self.add2 = P.Add()
        self.real_div = P.RealDiv()

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output


class ResBlock(nn.Cell):
    r"""Residual block."""

    def __init__(self, dim_in: int, dim_out: int, compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.linear = UniformInitDense(dim_in, dim_out).to_float(compute_dtype)
        self.activation = nn.LeakyReLU(0.1)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        identity = x
        out = identity + self.activation(self.linear(x))
        return out


class Hypernet(nn.Cell):
    r"""Hypernet which can maps a latent vector to a set of modulations."""

    def __init__(self,
                 latent_dim: int,
                 num_modulations: int,
                 dim_hidden: int,
                 num_layers: int,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        layers = [LayerNorm(latent_dim).to_float(mstype.float32)]
        if num_layers == 1:
            layers += [UniformInitDense(latent_dim, num_modulations).to_float(compute_dtype)]
        else:
            layers += [UniformInitDense(latent_dim, dim_hidden).to_float(compute_dtype)]
            for _ in range(num_layers - 1):
                layers += [ResBlock(dim_hidden, dim_hidden, compute_dtype=compute_dtype)]
            layers += [UniformInitDense(dim_hidden, num_modulations).to_float(compute_dtype)]
        self.net = nn.SequentialCell(*layers)

    def construct(self, latent: Tensor) -> Tensor:
        r"""construct"""
        return self.net(latent)


class MFNLayer(nn.Cell):
    r"""
    A single MFN Layer.

    Args:
        - in1_features (int): The number of input features.
        - out_features (int): The number of output features.
        - compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.
    """

    def __init__(self,
                 in1_features: int,
                 out_features: int,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.in1_features = in1_features
        self.out_features = out_features
        if self.in1_features <= 0:
            raise ValueError(f"'in1_features' must be a positive integer, but got {in1_features}.")
        if self.out_features <= 0:
            raise ValueError(f"'out_features' must be a positive integer, but got {out_features}.")

        weight = initializer(HeUniform(negative_slope=math.sqrt(5)),
                             (out_features, in1_features), dtype=mstype.float32)
        bound = 1 / math.sqrt(self.in1_features)
        bias = initializer(Uniform(scale=bound),
                           (out_features,), dtype=mstype.float32)

        self.linear = nn.Dense(
            in1_features, out_features).to_float(compute_dtype)

        self.linear.weight.set_data(weight)
        self.linear.bias.set_data(bias)

    def construct(self,
                  input1: Tensor,
                  scale_modulations: Optional[Tensor] = None,  # [num_layers+1, bsz, out_features]
                  shift_modulations: Optional[Tensor] = None,  # [num_layers+1, bsz, out_features]
                  layer_idx: int = 0,
                  ) -> Tensor:
        r"""construct"""
        if scale_modulations is not None:
            scale_code = ops.unsqueeze(scale_modulations[layer_idx], dim=1)  # [bsz, 1, out_features]
        else:
            scale_code = 1.0

        if shift_modulations is not None:
            shift_code = ops.unsqueeze(shift_modulations[layer_idx], dim=1)  # [bsz, 1, out_features]
        else:
            shift_code = 0.0

        linear_trans = self.linear(input1)  # [bsz, n_pts, out_features]
        linear_trans = scale_code * linear_trans + shift_code

        return linear_trans


class DINOFourierLayer(nn.Cell):
    r"""Sine filter as used in FourierNet."""

    def __init__(self, in_features: int, out_features: int, weight_scale: int) -> None:
        super().__init__()
        weight = initializer(HeUniform(negative_slope=math.sqrt(
            5)), (in_features, out_features), dtype=mstype.float32)
        self.weight = Parameter(weight, name="weight")
        self.weight_scale = weight_scale

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        linear_trans = ops.matmul(x, self.weight * self.weight_scale)
        return ops.cat((ops.sin(linear_trans), ops.cos(linear_trans)), axis=-1)


class OriginalFourierLayer(nn.Cell):
    r"""Sine filter as used in FourierNet."""

    def __init__(self, in_features: int, out_features: int, weight_scale: float) -> None:
        super().__init__()
        self.linear = nn.Dense(in_features, out_features)
        if in_features <= 0:
            raise ValueError(f"'in_features' must be a positive integer, but got {in_features}.")
        scale = weight_scale / math.sqrt(in_features)
        self.linear.weight.set_data(initializer(
            Uniform(scale), self.linear.weight.shape, self.linear.weight.dtype))
        self.linear.bias.set_data(initializer(
            Uniform(math.pi), self.linear.bias.shape, self.linear.bias.dtype))

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        return ops.sin(self.linear(x))


class GaborLayer(nn.Cell):
    r"""Gabor-like filter as used in GaborNet."""

    def __init__(self, in_features: int, out_features: int, weight_scale: float, alpha=1.0, beta=1.0) -> None:
        super().__init__()
        self.linear = nn.Dense(in_features, out_features)
        self.mu_value = Parameter(2 * np.random.rand(
            in_features, out_features).astype(np.float32) - 1)
        if in_features <= 0:
            raise ValueError(f"'in_features' must be a positive integer, but got {in_features}.")
        if out_features <= 0:
            raise ValueError(f"'out_features' must be a positive integer, but got {out_features}.")
        if beta <= 0:
            raise ValueError(f"'beta' must be a positive number, but got {beta}.")
        self.gamma = Parameter(np.random.gamma(
            alpha, 1 / beta, size=(1, out_features)).astype(np.float32))
        scale = weight_scale / math.sqrt(in_features)
        self.linear.weight.set_data(ops.sqrt(self.gamma.transpose(1, 0)) * initializer(
            Uniform(scale), self.linear.weight.shape, self.linear.weight.dtype))
        self.linear.bias.set_data(initializer(
            Uniform(math.pi), self.linear.bias.shape, self.linear.bias.dtype))

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        dist = ops.sum((x.expand_dims(-1) - self.mu_value)**2, dim=-2)
        return ops.sin(self.linear(x)) * ops.exp(-0.5 * self.gamma * dist)


class MFNNet(nn.Cell):
    r"""
    Multiplicative Filter Network (MFN) with Fourier/Gabor features.

    Args:
        inr_dim_in (int): Dimension of coordinate input.
        inr_dim_out (int): Dimension of MFN output.
        inr_dim_hidden (int): Dimension of MFN hidden layers.
        inr_num_layers (int): Number of MFN layers.
        filter_type (str, optional): Type of filters used in MFN.
            Choices: [``original_fourier``, ``dino_fourier``, ``gabor``].
            Default: ``dino_fourier``.
        input_scale (float, optional): Input scale used in the Fourier/Gabor
            features.  Default: ``256.0``.
        gabor_alpha (float, optional): For the Gabor features, the gamma
            parameter is initialized using the Gamma distribution
            Gamma(alpha, beta). Default: ``6.0``.
        gabor_beta (float, optional): Similar to gabor_alpha. Default: ``1.0``.
        compute_dtype (ms.dtype, optional): Floating point data type of the
            network. Default: ``ms.dtype.float16``.

    Inputs:
        - coordinate (Tensor): Shape (batch_size, num_points, inr_dim_in).
        - scale_modulations (Tensor, optional) - Tensor of shape :math:`(num\_layers+1, batch\_size, dim\_hidden)`.
        - shift_modulations (Tensor, optional) - Tensor of shape :math:`(num\_layers+1, batch\_size, dim\_hidden)`.

    Outputs:
        Output features of shape (batch_size, num_points, inr_dim_out).

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import MFNNet
        >>> x = Tensor(np.random.randn(2, 10, 3), mstype.float32)
        >>> mfn = MFNNet(3, 64, 128, 2, compute_dtype=mstype.float32)
        >>> out = mfn(x)
        >>> print(out.shape)
        (2, 10, 64)
    """

    def __init__(
            self,
            inr_dim_in: int,
            inr_dim_out: int,
            inr_dim_hidden: int,
            inr_num_layers: int,
            filter_type: str = 'dino_fourier',
            input_scale: float = 256.0,
            gabor_alpha: float = 6.0,
            gabor_beta: float = 1.0,
            compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.inr_num_layers = inr_num_layers
        if self.inr_num_layers < 2:
            raise ValueError(f"'inr_num_layers' should be at least 2, but got {inr_num_layers}.")
        self.mfn_layers = nn.CellList(
            [MFNLayer(
                inr_dim_in, inr_dim_hidden,
                compute_dtype=compute_dtype,
            )] + [MFNLayer(
                inr_dim_hidden, inr_dim_hidden,
                compute_dtype=compute_dtype,
            ) for _ in range(inr_num_layers - 2)]
        )
        self.output_linear = nn.Dense(
            inr_dim_hidden, inr_dim_out).to_float(compute_dtype)
        self.layernorms = nn.CellList([LayerNorm(inr_dim_hidden).to_float(
            mstype.float32) for _ in range(inr_num_layers - 2)])

        if filter_type.lower() == "dino_fourier":
            if inr_dim_hidden % 2 != 0:
                raise ValueError(
                    f"inr_dim_hidden ({inr_dim_hidden}) should be divisible by 2"
                    " for MFN FourierNet (dino_fourier).")
            self.filters = nn.CellList([DINOFourierLayer(
                inr_dim_in,
                inr_dim_hidden // 2,
                input_scale / np.sqrt(inr_num_layers - 1),
            ).to_float(compute_dtype) for _ in range(inr_num_layers - 1)])
        elif filter_type.lower() == "original_fourier":
            self.filters = nn.CellList([OriginalFourierLayer(
                inr_dim_in,
                inr_dim_hidden,
                input_scale / np.sqrt(inr_num_layers - 1),
            ).to_float(compute_dtype) for _ in range(inr_num_layers - 1)])
        elif filter_type.lower() == "gabor":
            self.filters = nn.CellList([GaborLayer(
                inr_dim_in,
                inr_dim_hidden,
                input_scale / np.sqrt(inr_num_layers - 1),
                gabor_alpha / (inr_num_layers - 1),
                gabor_beta,
            ).to_float(compute_dtype) for _ in range(inr_num_layers - 1)])
        else:
            raise ValueError("MFN 'filter_type' should be in  ['original_fourier', "
                             f"'dino_fourier', 'gabor'], but got '{filter_type}'.")

    def construct(self,
                  coordinate: Tensor,
                  scale_modulations: Optional[Tensor] = None,  # [inr_num_layers-1, bsz, out_features]
                  shift_modulations: Optional[Tensor] = None,  # [inr_num_layers-1, bsz, out_features]
                  ) -> Tensor:
        r"""construct"""
        out = 0 * coordinate
        for i in range(self.inr_num_layers - 1):
            if i > 0:
                out = self.layernorms[i - 1](out)
            out = self.mfn_layers[i](out, scale_modulations, shift_modulations, layer_idx=i)
            out = out * self.filters[i](coordinate)
        out = self.output_linear(out)

        return out


class MFNNetWithHypernet(nn.Cell):
    r"""
    MFNNet model with hypernets.

    Args:
        inr_dim_in (int): Dimension of coordinate input.
        inr_dim_out (int): Dimension of MFN output.
        inr_dim_hidden (int): Dimension of MFN hidden layers.
        inr_num_layers (int): Number of MFN layers.
        hyper_dim_in (int): Dimension of each hypernet input.
        hyper_dim_hidden (int): Dimension of hidden layers of the hypernet.
        hyper_num_layers (int): Number of layers of the hypernet.
        filter_type (str, optional): Type of features used in MFN.
            Choices: [``original_fourier``, ``dino_fourier``, ``gabor``].
            Default: ``dino_fourier``.
        input_scale (float, optional): Input scale used in the Fourier/Gabor
            features.  Default: ``256.0``.
        gabor_alpha (float, optional): For the Gabor features, the gamma
            parameter is initialized using the Gamma distribution
            Gamma(alpha, beta). Default: ``6.0``.
        gabor_beta (float, optional): Similar to gabor_alpha. Default: ``1.0``.
        share_hypernet (bool, optional): Whether the modulations of all MFN
            hidden layers are generated by the same modulation encoder.
            Default: ``False``.
        enable_shift (bool, optional): Whether to introduce shift modulations
            of the MFN network. Default: ``False``.
        enable_scale (bool, optional): Whether to introduce scale modulations
            of the MFN network. Default: ``False``.
        compute_dtype (ms.dtype, optional): Floating point data type of the
            network. Default: ``ms.dtype.float16``.

    Inputs:
        - **coordinate** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.
        - **hyper_in** (Tensor) - Tensor of shape :math:`(inr\_num\_layers + 1, batch\_size, hyper\_dim\_in)`.

    Outputs:
        Tensor of shape :math:`(batch\_size, num\_points, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import MFNNetWithHypernet
        >>> inr_dim_in = 2
        >>> inr_dim_out = 1
        >>> inr_dim_hidden = 128
        >>> inr_num_layers = 6
        >>> hyper_dim_in = 16
        >>> hyper_dim_hidden = 32
        >>> hyper_num_layers = 2
        >>> mfn = MFNNetWithHypernet(inr_dim_in, inr_dim_out, inr_dim_hidden, inr_num_layers,
                                hyper_dim_in, hyper_dim_hidden, hyper_num_layers,
                                'dino_fourier', 256.0, 6.0, 1.0, False, True, True,
                                mstype.float32)
        >>> bsz = 32
        >>> n_pts = 128
        >>> coord = Tensor(np.random.randn(bsz, n_pts, inr_dim_in), mstype.float32)
        >>> hyper_in = Tensor(np.random.randn(inr_num_layers - 1, bsz, hyper_dim_in), mstype.float32)
        >>> out = mfn(coord, hyper_in)
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
            filter_type: str = 'dino_fourier',
            input_scale: float = 256.0,
            gabor_alpha: float = 6.0,
            gabor_beta: float = 1.0,
            share_hypernet: bool = False,
            enable_shift: bool = True,
            enable_scale: bool = True,
            compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.inr_num_layers = inr_num_layers
        self.enable_shift = enable_shift
        self.enable_scale = enable_scale
        if not (enable_shift or enable_scale):
            raise ValueError(
                "For MFN, at least one of ['enable_shift', 'enable_scale'] should be set to True.")
        self.share_hypernet = share_hypernet
        self.affine_modulations_shape = (
            inr_num_layers - 1, -1, inr_dim_in + 1, inr_dim_hidden)

        self.inr = MFNNet(
            inr_dim_in=inr_dim_in,
            inr_dim_hidden=inr_dim_hidden,
            inr_dim_out=inr_dim_out,
            inr_num_layers=inr_num_layers,
            filter_type=filter_type,
            input_scale=input_scale,
            gabor_alpha=gabor_alpha,
            gabor_beta=gabor_beta,
            compute_dtype=compute_dtype
        )

        def new_hypernet():
            return Hypernet(hyper_dim_in, inr_dim_hidden, hyper_dim_hidden,
                            hyper_num_layers, compute_dtype)

        if self.share_hypernet:
            if enable_shift:
                self.shift_hypernet = new_hypernet()
            if enable_scale:
                self.scale_hypernet = new_hypernet()
        else:
            num_hypernet = inr_num_layers - 1  # the number of hidden layers
            if self.enable_shift:
                self.shift_hypernets = nn.CellList([
                    new_hypernet() for _ in range(num_hypernet)])
            if self.enable_scale:
                self.scale_hypernets = nn.CellList([
                    new_hypernet() for _ in range(num_hypernet)])

    def construct(self, coordinate: Tensor, hyper_in: Tensor) -> Tensor:
        r"""construct"""

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
        out = self.inr(coordinate, scale_modulations, shift_modulations)

        return out
