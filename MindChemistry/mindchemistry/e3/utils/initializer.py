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

from mindspore.common.initializer import Initializer, _register, _init_random_uniform, _assignment, TruncatedNormal, \
    Normal, HeNormal, HeUniform, XavierUniform


@_register()
class Uniform(Initializer):
    r"""
    Generates an array with values sampled from Uniform distribution :math:`{U}(-\text{scale}, \text{scale})` in order
    to initialize a tensor.

    Args:
        scale (float): The bound of the Uniform distribution. Default: 1.0.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Uniform
        >>> tensor1 = initializer(Uniform(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('uniform', [1, 2, 3], mindspore.float32)
    """

    def __init__(self, scale=1.):
        super(Uniform, self).__init__(scale=scale)
        self.scale = scale

    def _initialize(self, arr):
        tmp = _init_random_uniform(0., self.scale, arr.shape)
        _assignment(arr, tmp)


def renormal_initializer(init_method):
    name_list = ['zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform']
    if not init_method in name_list and not isinstance(init_method, Initializer):
        raise ValueError(
            f'initial method \"{init_method}\" is not supported.')

    if init_method == 'truncatedNormal':
        init_method = TruncatedNormal(sigma=1.)
    elif init_method == 'normal':
        init_method = Normal(sigma=1.)
    elif init_method == 'uniform':
        init_method = Uniform()
    elif init_method == 'he_normal':
        init_method = HeNormal()
    elif init_method == 'he_uniform':
        init_method = HeUniform()

    return init_method
