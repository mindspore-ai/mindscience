# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
init
"""
import os

from mindspore import ops, Tensor, nn, load_param_into_net, load_checkpoint


class SciModule(nn.Cell):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def load(self, ckpt_file):
        assert ckpt_file.endswith('.ckpt') and os.path.exists(ckpt_file)
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(self, param_dict)

    def set_grad(self):
        super.set_grad()

    @property
    def num_params(self):
        return sum([i.numel() for i in self.trainable_params()])