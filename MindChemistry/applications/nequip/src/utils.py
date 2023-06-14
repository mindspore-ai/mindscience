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
"""
visualization functions
"""
import os
import psutil

from mindspore import Tensor, dtype


def training_bar(epoch, size, current, loss=None):
    stride = 50
    while size < stride:
        stride //= 2
    if current % (size // stride) == 0:
        _complete = current * stride // size
        if loss is not None:
            loss = loss.asnumpy() if isinstance(loss, Tensor) else loss
            print(f'\r(loss = {loss:>4.4f}) ', end='')
        else:
            print('\r', end='')
        _memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        _ratio = (current + 1) / size * 100
        print(f'Training epoch {epoch + 1}: [\033[92m' + '■' * (_complete + 1) + '\033[0m' + ' ' * (
                stride - _complete - 1) + f'] {_ratio:.2f}%    Memory used: {_memory:>6.2f} MB   ', end='')
