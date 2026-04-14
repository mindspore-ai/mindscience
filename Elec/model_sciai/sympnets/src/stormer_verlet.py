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
"""stormer verlet"""
import mindspore as ms
from mindspore import ops, Parameter
import numpy as np

from .utils import grad


class StormerVerlet:
    """Stormer-Verlet scheme."""

    def __init__(self, h, dh, iterations=10, order=4, n=1):
        """
        initialize
        Args:
            h: h(x) or None
            dh: dp, dq = dh(p, q) or None
            iterations: number of iterations
            order: problem order
            n: n
        """
        self.h = h
        self.dh = dh
        self.iterations = iterations
        self.order = order
        self.n = n
        self.cat = ops.Concat(axis=-1)
        self.solver_dict = {2: self.__sv2, 4: self.__sv4, 6: self.__sv6}

    def solve(self, x, h):
        if self.order not in self.solver_dict:
            raise NotImplementedError()
        solver = self.solver_dict.get(self.order)
        for _ in range(self.n):
            x = solver(x, h / self.n)
        return x

    def flow(self, x, h, steps):
        dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
        size = len(x.shape) if isinstance(x, np.ndarray) else len(x.size())
        x_ = [x]
        for _ in range(steps):
            x_.append(self.solve(x_[-1], h))
        shape = [steps + 1, dim] if size == 1 else [-1, steps + 1, dim]
        return np.hstack(x_).reshape(shape) if isinstance(x, np.ndarray) else self.cat(x_).view(shape)

    def __sv2(self, x, h):
        """Order 2.
        x: np.ndarray or mindspore.Tensor of shape [dim] or [num, dim].
        h: int
        """
        dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
        d = int(dim / 2)
        p0 = x[..., :d]
        q0 = x[..., d:]
        p1 = p0
        if callable(self.dh):
            for _ in range(self.iterations):
                p1 = p0 - h / 2 * self.dh(p1, q0)[1]
            q1 = q0 + h / 2 * self.dh(p1, q0)[0]
            q2 = q1
            for _ in range(self.iterations):
                q2 = q1 + h / 2 * self.dh(p1, q2)[0]
            p2 = p1 - h / 2 * self.dh(p1, q2)[1]
            if isinstance(x, np.ndarray):
                return np.hstack([p2, q2])
            return self.cat([p2, q2])
        if isinstance(x, ms.Tensor):
            for _ in range(self.iterations):
                x = Parameter(self.cat([p1, q0]))
                dh = grad(self.h, x)
                p1 = p0 - h / 2 * dh[..., d:]
            q1 = q0 + h / 2 * dh[..., :d]
            q2 = q1
            for _ in range(self.iterations):
                x = self.cat([p1, q2]).requires_grad_(True)
                dh = grad(self.h, x)
                q2 = q1 + h / 2 * dh[..., :d]
            p2 = p1 - h / 2 * dh[..., d:]
            return self.cat([p2, q2])
        raise ValueError()

    def __sv4(self, x, h):
        """Order 4"""
        r1 = 1 / (2 - 2 ** (1 / 3))
        r2 = - 2 ** (1 / 3) / (2 - 2 ** (1 / 3))
        sv2_ = self.__sv2(x, r1 * h)
        sv2_ = self.__sv2(sv2_, r2 * h)
        return self.__sv2(sv2_, r1 * h)

    def __sv6(self, x, h):
        """Order 6"""
        r1 = 1 / (2 - 2 ** (1 / 5))
        r2 = - 2 ** (1 / 5) / (2 - 2 ** (1 / 5))
        sv4_ = self.__sv4(x, r1 * h)
        sv4_ = self.__sv4(sv4_, r2 * h)
        return self.__sv4(sv4_, r1 * h)
