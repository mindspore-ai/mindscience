# Copyright 2024 Huawei Technologies Co., Ltd
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
"""utils file"""

import math
from mindspore import ops, nn, Parameter


class LossRecord:
    """LossRecord"""

    def __init__(self):
        self.last_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """reset"""
        self.last_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """update"""
        self.last_val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


class RBFExpansion(nn.Cell):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
            self,
            vmin=0,
            vmax=8,
            bins=40,
            lengthscale=None,
    ):
        """Register torch parameters for RBF expansion."""
        super(RBFExpansion, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.centers = Parameter(ops.linspace(self.vmin, self.vmax, self.bins), name="centers", requires_grad=False)
        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = ops.mean(ops.diff(self.centers))
            self.gamma = -1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = -1 / (lengthscale ** 2)

    def construct(self, distance):
        """Apply RBF expansion to interatomic distance tensor."""
        tmp1 = ops.unsqueeze(distance, dim=1)
        tmp2 = tmp1 - self.centers
        tmp3 = tmp2 ** 2
        tmp4 = self.gamma * tmp3
        res = ops.exp(tmp4)
        return res


class OneCycleLr():
    """one cycle learning rate scheduler"""

    def __init__(self, max_lr, steps_per_epoch, epochs, optimizer, pct_start=0.3, anneal_strategy="cos",
                 div_factor=25.0, final_div_factor=10000.0):
        """init"""
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.optimizer = optimizer
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.current_step = 0

        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor
        self.steps = self.steps_per_epoch * self.epochs
        self.step_size_up = float(self.pct_start * self.steps) - 1
        self.step_size_down = float(2 * self.pct_start * self.steps) - 2
        self.step_size_end = float(self.steps) - 1

        self.step()

    def _annealing_cos(self, start, end, pct):
        """annealing cosin"""
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def step(self):
        """step"""
        if self.current_step <= self.step_size_up:
            lr = self._annealing_cos(self.initial_lr, self.max_lr, self.current_step / self.step_size_up)
        else:
            lr = self._annealing_cos(self.max_lr, self.min_lr,
                                     (self.current_step - self.step_size_up) / (self.step_size_end - self.step_size_up))
        self.current_step = self.current_step + 1
        ### for AdamWeightDecay
        self.optimizer.learning_rate.set_data(lr)
