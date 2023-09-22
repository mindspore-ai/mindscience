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
"""scheduler for lr"""

from bisect import bisect_right

class WarmupMultiStepLR():
    """WarmupMultiStepLR"""
    def __init__(self,
                 base_lr,
                 milestones,
                 gamma=0.1,
                 warmup_factor=1.0 / 3,
                 warmup_iters=5,
                 warmup_method="linear",):
        if milestones != sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(milestones)
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted got {}".format(warmup_method)
            )
        self.base_lr = base_lr
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

    def get_lr(self, dataset_size, start_epoch, epochs):
        """get_lr"""
        lr_list = []
        for step in range(epochs * dataset_size):
            warmup_factor = 1
            if step < self.warmup_iters:
                if self.warmup_method == "constant":
                    warmup_factor = self.warmup_factor
                elif self.warmup_method == "linear":
                    alpha = float(step) / self.warmup_iters
                    warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            lr_list.append(
                warmup_factor *
                self.base_lr *
                self.gamma ** bisect_right(self.milestones, step))

        lr_list = lr_list[start_epoch * dataset_size:]
        return lr_list
