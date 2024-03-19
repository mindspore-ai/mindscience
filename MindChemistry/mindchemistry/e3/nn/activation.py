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
"""activation"""
import numpy as np

from mindspore import Tensor, nn, ops, float32
from ..o3.irreps import Irreps

identity = ops.Identity()
NTOL = 1e-5


def _moment(f, n, dtype=float32):
    x = Tensor(np.random.randn(1000000), dtype=dtype)
    y = f(x).pow(n).mean().pow(-0.5)

    return y


def _parity_function(f, dtype=float32):
    x = Tensor(np.linspace(.0, 10., 256), dtype=dtype)
    y1, y2 = f(x).asnumpy(), f(-x).asnumpy()
    if np.max(np.abs(y1 - y2)) < NTOL:
        return 1
    if np.max(np.abs(y1 + y2)) < NTOL:
        return -1
    return 0


class _Normalize(nn.Cell):
    """_Normalize"""

    def __init__(self, f, dtype=float32):
        super().__init__()
        self.f = f
        self.factor = _moment(f, 2, dtype)
        if ops.abs(self.factor - 1.) < 1e-4:
            self._is_id = True
        else:
            self._is_id = False

    def construct(self, x):
        if self._is_id:
            return self.f(x)
        return self.f(x).mul(self.factor)


class Activation(nn.Cell):
    r"""
    Activation function for scalar-tensors. The parities of irreps may be changed according to the parity of each
    activation functions.
    Odd scalars require the corresponding activation functions to be odd or even.

    Args:
        irreps_in (Union[str, Irrep, Irreps]): the input irreps.
        acts (List[Func]): a list of activation functions for each part of `irreps_in`.
            The length of the `acts` will be clipped or filled by identity functions to match the length of `irreps_in`.

    Raises:
        ValueError: If `irreps_in` contain non-scalar irrep.
        ValueError: If a irrep in `irreps_in` is odd, but the corresponding activation function is neither even nor odd.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> Activation('3x0o+2x0e+1x0o', [ops.abs, ops.tanh])
        Activation [xx-] (3x0o+2x0e+1x0o -> 3x0e+2x0e+1x0o)
    """

    def __init__(self, irreps_in, acts, dtype=float32):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        while len(acts) < len(irreps_in):
            acts.append(None)
        irreps_out = []
        acts_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in.data, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(f"Activation cannot apply an activation function to a non-scalar input.")

                acts_out.append(_Normalize(act, dtype=dtype))
                p_out = _parity_function(acts_out[-1]) if p_in == -1 else p_in

                if p_out == 0:
                    raise ValueError(
                        "Parity is not match. The input scalar is odd but the activation is neither even nor odd."
                    )

                irreps_out.append((mul, (0, p_out)))

            else:
                acts_out.append(identity)
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = Irreps(irreps_out)
        self.acts = acts_out[:len(irreps_in)]

    def construct(self, v):
        """Implement the activation function for the input tensor."""
        vs = self.irreps_in.decompose(v)
        batch_shape = v.shape[:-1]
        out_list = []
        i = 0
        for act in self.acts:
            out_list.append(act(vs[i]).reshape(batch_shape + (self.irreps_in.data[i].dim,)))
            i += 1

        if len(out_list) > 1:
            out = ops.concat(out_list, axis=-1)
        elif len(out_list) == 1:
            out = out_list[0]
        else:
            out = ops.zeros_like(v)
        return out

    def __repr__(self):
        acts = "".join(["x" if a is not identity else "-" for a in self.acts])
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"
