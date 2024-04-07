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
r"""FUnction Encoder."""
from mindspore import nn, Tensor, ops
from mindspore import dtype as mstype

from ..basic_block import MLP
from .inr_with_hypernet import Siren, MFNNet, PolyINR


class DeepSetFuncEncoder(nn.Cell):
    r"""
    Encoder for functions defined on one-dimensional domain.

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of the hidden features.
        num_layers (int): Number of layers.
        point_fn (str): Point function type. Options are "mlp" and "poly_inr". Default: "poly_inr".

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(batch\_size, num\_points, dim\_out)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import DeepSetFuncEncoder
        >>> x = Tensor(np.random.randn(2, 10, 3), mstype.float32)
        >>> encoder = DeepSetFuncEncoder(3, 64, 128, 2, point_fn="poly_inr")
        >>> out = encoder(x)
        >>> print(out.shape)
        (2, 10, 64)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 6,
                 point_fn: str = "poly_inr",
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        if point_fn == "mlp":
            self.point_fn = MLP(dim_in, dim_hidden, dim_hidden,
                                num_layers//2, compute_dtype=compute_dtype)
        elif point_fn == "poly_inr":
            self.point_fn = PolyINR(dim_in, dim_hidden, dim_hidden,
                                    num_layers//2, compute_dtype=compute_dtype)
        elif point_fn == "mfn":
            self.point_fn = MFNNet(dim_in, dim_hidden, dim_hidden,
                                   num_layers//2, compute_dtype=compute_dtype)
        elif point_fn == "siren":
            self.point_fn = Siren(dim_in, dim_hidden, dim_hidden,
                                  num_layers//2, compute_dtype=compute_dtype)
        else:
            raise NotImplementedError(f"Point function '{point_fn}' not implemented!")
        self.post_fn = MLP(dim_hidden, dim_out, dim_hidden, num_layers=num_layers//2, compute_dtype=compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        out = self.point_fn(x)  # [..., num_points, dim_hidden]
        out = ops.mean(out, axis=-2)  # [..., dim_hidden]
        out = self.post_fn(out)  # [..., dim_out]
        return out


class WeightedDeepSetFuncEncoder(nn.Cell):
    r"""Encoder for functions defined on one-dimensional domain.

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of the hidden features.
        num_layers (int): Number of layers. Default: 6.
        point_fn (str): Point function type. Options are "mlp", "poly_inr",
            "poly_inr_shared", and "siren". Default: "poly_inr".
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(..., num\_points, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(..., dim\_out)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import WeightedDeepSetFuncEncoder
        >>> x = Tensor(np.random.randn(2, 10, 3), mstype.float32)
        >>> encoder = WeightedDeepSetFuncEncoder(3, 64, 128, 5, point_fn="poly_inr")
        >>> out = encoder(x)
        >>> print(out.shape)
        (2, 64)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 6,
                 point_fn: str = "poly_inr",
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        num_layers = num_layers // 2
        if point_fn == "mlp":
            self.point_fn = MLP(dim_in, dim_hidden, dim_hidden, num_layers, compute_dtype=compute_dtype)
            self.weight_fn = MLP(dim_in, dim_hidden, dim_hidden, num_layers, compute_dtype=compute_dtype)
        elif point_fn == "poly_inr":
            self.point_fn = PolyINR(dim_in, dim_hidden, dim_hidden, num_layers, compute_dtype=compute_dtype)
            self.weight_fn = PolyINR(dim_in, dim_hidden, dim_hidden, num_layers, compute_dtype=compute_dtype)
        elif point_fn == "mfn":
            self.point_fn = MFNNet(dim_in, dim_hidden, dim_hidden,
                                   num_layers, compute_dtype=compute_dtype)
            self.weight_fn = MFNNet(dim_in, dim_hidden, dim_hidden,
                                    num_layers, compute_dtype=compute_dtype)
        elif point_fn == "siren":
            self.point_fn = Siren(dim_in, dim_hidden, dim_hidden,
                                  num_layers, compute_dtype=compute_dtype)
            self.weight_fn = Siren(dim_in, dim_hidden, dim_hidden,
                                   num_layers, compute_dtype=compute_dtype)
        else:
            raise NotImplementedError(f"Point function '{point_fn}' not implemented!")
        self.post_fn = MLP(dim_hidden, dim_out, dim_hidden, num_layers, compute_dtype=compute_dtype)
        self.cast = ops.Cast()
        self.compute_dtype = compute_dtype

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        feature = self.point_fn(x)  # [..., num_points, dim_hidden]
        wt = self.weight_fn(x)  # [..., num_points, dim_hidden]
        wt = self.cast(wt, mstype.float32)
        probs = ops.softmax(wt, axis=-2)
        probs = self.cast(probs, self.compute_dtype)
        feature_probs = feature * probs
        out = ops.sum(feature_probs, dim=-2)  # [..., num_points, dim_hidden] -> [..., dim_hidden]
        out = self.post_fn(out)  # [..., dim_out]
        return out


class PatchedFuncEncoder(nn.Cell):
    r"""Encoder for functions defined on one-dimensional domain.

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of the hidden features.
        num_layers (int): Number of layers. Default: 3.
        patch_len (int): Length of the patch. Default: 16.
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(batch\_size * num\_points / patch\_len, dim\_out)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import PatchedFuncEncoder
        >>> dim_in, dim_out, dim_hidden, num_layers, patch_len = 3, 256, 256, 5, 4
        >>> num_points = 128
        >>> x = Tensor(np.random.randn(2, num_points, dim_in), mstype.float32)
        >>> encoder = PatchedFuncEncoder(dim_in, dim_out, dim_hidden, num_layers,
        >>>                              patch_len, compute_dtype=mstype.float32)
        >>> out = encoder(x)
        >>> print(out.shape)
        (64, 256)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 patch_len: int = 16,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.input_dim = patch_len * dim_in
        self.mlp = MLP(self.input_dim, dim_out, dim_hidden, num_layers, compute_dtype=compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        out = x.reshape(-1, self.input_dim)  # [bsz, num_points, dim_in] -> [bsz*num_patch, patch_len*dim_in]
        out = self.mlp(out)  # [bsz*num_patch, dim_out]
        return out
