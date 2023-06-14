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
from mindspore import Tensor, float32

from mindchemistry.e3 import BatchNorm


def test_batchnorm():
    irreps = "1x0e+2x1o"
    bn = BatchNorm(irreps, affine=False)
    v = Tensor([[0., 1., 0., 0., 0., 1., 0.]], dtype=float32)
    out = bn(v)


if __name__ == '__main__':
    test_batchnorm()
