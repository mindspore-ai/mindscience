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
# ==============================================================================
''' provide complex dft based on the real dft API in mindflow.dft '''
import numpy as np
import mindspore as ms
from mindspore import nn, ops, numpy as mnp
from mindflow.cell.neural_operators.dft import dft1, dft2, dft3


class MyDFTn(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        assert len(shape) in (1, 2, 3), 'only ndim 1, 2, 3 supported'

        n = shape[-1]
        ndim = len(shape)
        modes = tuple([_ // 2 for _ in shape[-ndim:-1]] + [n // 2 + 1]) if ndim > 1 else n // 2 + 1

        self.shape = tuple(shape)
        self.dft_cell = {
            1: dft1,
            2: dft2,
            3: dft3,
            }[ndim](shape, modes)

        # use mask to assemble slices of Tensors, avoiding dynamic shape
        # bug note: for unknown reasons, GRAPH_MODE cannot work with mask Tensors allocated using ops.ones()
        mask_x0 = np.ones(n//2 + 1)
        mask_xm = np.ones(n//2 + 1)
        mask_y0 = np.ones(shape)
        mask_z0 = np.ones(shape)
        mask_x0[0] = 0
        mask_xm[-1] = 0
        if ndim > 1:
            mask_y0[..., 0, :] = 0
        if ndim > 2:
            mask_z0[..., 0, :, :] = 0

        self.mask_x0 = ms.Tensor(mask_x0, dtype=ms.float32, const_arg=True)
        self.mask_xm = ms.Tensor(mask_xm, dtype=ms.float32, const_arg=True)
        self.mask_y0 = ms.Tensor(mask_y0, dtype=ms.float32, const_arg=True)
        self.mask_z0 = ms.Tensor(mask_z0, dtype=ms.float32, const_arg=True)

    def construct(self, ar, ai):
        shape = tuple(self.shape)
        n = shape[-1]
        ndim = len(shape)
        scale = float(np.prod(shape) ** .5)

        assert ai is None or ar.shape == ai.shape
        assert ar.shape[-ndim:] == shape

        brr, bri = self.dft_cell((ar, ar * 0))

        # n-D Fourier transform with last axis being real-transformed, output dimension (..., m, n//2+1)
        if ai is None:
            return brr * scale, bri * scale

        # n-D complex Fourier transform, output dimension (..., m, n)
        # call dft for real & imag parts separately and then assemble
        bir, bii = self.dft_cell((ai, ai * 0))

        br_half1 = ops.pad((brr - bii) * self.mask_xm, [0, n//2 - 1])
        bi_half1 = ops.pad((bri + bir) * self.mask_xm, [0, n//2 - 1])
        # bug note: mnp.roll() & mnp.flip are ok, but ops.roll() only supports GPU, ops.flip() has bug in MS2.4.0
        br_half2 = mnp.roll(mnp.flip(ops.pad((brr + bii) * self.mask_x0, [n//2 - 1, 0]), axis=-1), n//2, axis=-1)
        bi_half2 = mnp.roll(mnp.flip(ops.pad((bir - bri) * self.mask_x0, [n//2 - 1, 0]), axis=-1), n//2, axis=-1)
        if ndim > 1:
            br_half2 = br_half2 * (1 - self.mask_y0) + mnp.roll(mnp.flip(br_half2 * self.mask_y0, axis=-2), 1, axis=-2)
            bi_half2 = bi_half2 * (1 - self.mask_y0) + mnp.roll(mnp.flip(bi_half2 * self.mask_y0, axis=-2), 1, axis=-2)
        if ndim > 2:
            br_half2 = br_half2 * (1 - self.mask_z0) + mnp.roll(mnp.flip(br_half2 * self.mask_z0, axis=-3), 1, axis=-3)
            bi_half2 = bi_half2 * (1 - self.mask_z0) + mnp.roll(mnp.flip(bi_half2 * self.mask_z0, axis=-3), 1, axis=-3)

        br = br_half1 + br_half2
        bi = bi_half1 + bi_half2

        return br * scale, bi * scale

class MyiDFTn(MyDFTn):
    def __init__(self, shape):
        super().__init__(shape)

    def construct(self, ar, ai):
        ndim = len(self.shape)
        scale = float(np.prod(ar.shape[-ndim:]))
        br, bi = super().construct(ar, -ai)
        return br / scale, -bi / scale
