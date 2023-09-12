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
# ==============================================================================
"""optimizer"""
import time
from functools import reduce
from itertools import accumulate

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import mutable
from mindspore.scipy.optimize import minimize

from sciai.architecture.basic_block import NoArgNet
from sciai.utils import to_tensor, print_log, to_tuple


class LbfgsOptimizer:
    """
    L-BFGS second-order optimizer, which is currently only supported in PYNATIVE_MODE.

    Args:
        closure (Callable): The function which gives the loss.
        weights (list[Parameter]): The parameter to be optimized.

    Inputs:
        - **options** (Mapping[str, Any]) - Ref to mindspore.scipy.minimize.

    Outputs:
        OptimizeResults, Object holding optimization results.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn, ops
        >>> from sciai.architecture.basic_block import NoArgNet
        >>> from sciai.common import LbfgsOptimizer
        >>> ms.set_seed(1234)
        >>> class Net1In1Out(nn.Cell):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.dense1 = nn.Dense(2, 1)
        >>>     def construct(self, x):
        >>>         return self.dense1(x).sum()
        >>> net = Net1In1Out()
        >>> x = ops.ones((3, 2), ms.float32)
        >>> cell = NoArgNet(net, x)
        >>> optim_lbfgs = LbfgsOptimizer(cell, list(cell.trainable_params()))
        >>> res = optim_lbfgs.construct(options=dict(maxiter=None, gtol=1e-6))
        >>> print(res.x)
        [0.00279552 0.00540159  0.        ]
    """

    def __init__(self, closure, weights):
        super().__init__()
        self.fn = closure
        self.weights = mutable(weights)
        self.fn_grad = ops.grad(self.fn, grad_position=None, weights=weights)
        self.flat_fn = _FlattenFunctional(self.fn, weights)
        self.flat_fn_grad = _FlattenFunctional(self.fn_grad, weights)

    def construct(self, options):
        x0 = _flatten(self.weights)
        result = minimize(self.flat_fn, x0, method="lbfgs", jac=self.flat_fn_grad, options=options)
        return result


class _FlattenFunctional(nn.Cell):
    """
    Flatten functional class, which flatten the input and parameters.

    Args:
        f (nn.Cell): Function to flatten the trainable parameters.
        params (list[Parameter]): Trainable parameters of the network.

    Inputs:
        - **x** (Parameter) - Trainable parameter of the network weight.

    Outputs:
        L-BFGS result object.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``
    """

    def __init__(self, f, params):
        super().__init__()
        self.params = params
        self.f = f
        self.zeros = ops.Zeros()
        self.assign = ops.Assign()
        self.param_shapes = [p.shape for p in params]
        self.offset = [0,] + list(accumulate([reduce(lambda a, b: a * b, p.shape) for p in params]))

    def construct(self, x):
        for i, p in enumerate(self.params):
            self.assign(p, x[self.offset[i]: self.offset[i + 1]].reshape(self.param_shapes[i]))
        out = self.f()
        if isinstance(out, tuple):
            fout = self.zeros((self.offset[-1]), ms.float32)
            for i, o in enumerate(out):
                fout[self.offset[i]: self.offset[i + 1]] = o.ravel()
            out = fout
        return out


def _flatten(params):
    """
    Flatten parameters

    Args:
        params (tuple(Parameter)): Parameters to flatten.

    Returns:
        Tensor, Flattened parameter.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``
    """
    offset = [0,] + list(accumulate([reduce(lambda a, b: a * b, p.shape) for p in params]))
    x = ops.zeros((offset[-1]), ms.float32)
    for i, o in enumerate(params):
        x[offset[i]: offset[i + 1]] = o.ravel()
    return x


def lbfgs_train(loss_net, input_data, lbfgs_iter):
    """
    L-BFGS training function, which can only run on PYNATIVE mode currently.

    Args:
        loss_net (Cell): Network which returns loss as objective function.
        input_data (Union[Tensor, tuple[Tensor]]): Input data of the loss_net.
        lbfgs_iter (int): Number of iterations of the l-bfgs training process.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn, ops
        >>> from sciai.common import lbfgs_train
        >>> ms.set_seed(1234)
        >>> class Net1In1Out(nn.Cell):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.dense1 = nn.Dense(2, 1)
        >>>     def construct(self, x):
        >>>         return self.dense1(x).abs().sum()
        >>> net = Net1In1Out()
        >>> x = ops.ones((3, 2), ms.float32)
        >>> lbfgs_train(net, (x,), 1000)
        >>> loss = net(x)
        >>> print(loss)
        5.9944578e-06
    """
    no_arg_net = NoArgNet(loss_net, *to_tuple(to_tensor(input_data, ms.float32)))
    no_arg_net.to_float(ms.float32)
    original_mode = ms.get_context("mode")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    optim_lbfgs = LbfgsOptimizer(no_arg_net, list(loss_net.trainable_params()))
    start_time = time.time()
    _ = optim_lbfgs.construct(options=dict(maxiter=lbfgs_iter, gtol=1e-6))
    ms.set_context(mode=original_mode)
    this_time = time.time()
    print_log(f'l-bfgs total time:{this_time - start_time}')
