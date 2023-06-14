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
from mindspore import Tensor, nn, Parameter, float32, ops
from mindspore.common.initializer import initializer

from ..nn.activation import _Normalize
from ..utils.initializer import renormal_initializer

identity = ops.Identity()


class _Layer(nn.Cell):
    r"""Single simple dense layer with parameter w."""

    def __init__(self, h_in, h_out, act, init_method='normal', dtype=float32):
        super().__init__()

        init_method = renormal_initializer(init_method)

        self.weight = Parameter(initializer(
            init_method, (h_in, h_out), dtype), name='Layer')
        self.act = act if act is not None else identity
        self.h_in = h_in
        self.h_out = h_out
        self.weight_numel = self.weight.numel()
        self.sqrt_h_in = ops.sqrt(Tensor(self.h_in, self.weight.dtype))

    def construct(self, x):
        w = self.weight / self.sqrt_h_in
        x = ops.matmul(x, w)
        x = self.act(x)
        return x

    def __repr__(self):
        return f"Layer ({self.h_in}->{self.h_out})"


class FullyConnectedNet(nn.SequentialCell):
    r"""
    Fully-connected Neural Network with normalized activation on scalars.

    Args:
        h_list (List[int]): a list of input, internal and output dimensions for dense layers.
        act (Func): activation function which will be automatically normalized. Default: None
        out_act (bool): whether apply the activation function on the output. Default: False

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Raises:
        TypeError: If the elements `h_list` are not `int`.

    Examples:
        >>> fc = FullyConnectedNet([4,10,20,12,6], ops.tanh)
        FullyConnectedNet [4, 10, 20, 12, 6]
        >>> v = ms.Tensor([.1,.2,.3,.4])
        >>> grad = ops.grad(fc, weights=fc.trainable_params())
        >>> fc(v).shape
        (6,)
        >>> [x.shape for x in grad(v)[1]]
        [(4, 10), (10, 20), (20, 12), (12, 6)]

    """

    def __init__(self, h_list, act=None, out_act=False, init_method='normal', dtype=float32):
        self.h_list = list(h_list)
        if act is not None:
            act = _Normalize(act, dtype=dtype)

        self.layer_list = []

        for i, (h1, h2) in enumerate(zip(self.h_list, self.h_list[1:])):
            if not isinstance(h1, int) or not isinstance(h2, int):
                raise TypeError

            if i == len(self.h_list) - 2 and (not out_act):
                a = identity
            else:
                a = act
            layer = _Layer(h1, h2, a, init_method, dtype=dtype)
            self.layer_list.append(layer)

        super().__init__(self.layer_list)
        self.weight_numel = sum([lay.weight_numel for lay in self.layer_list])

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.h_list} | {self.weight_numel} weights)"
