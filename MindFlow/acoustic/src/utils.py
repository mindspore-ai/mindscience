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
"""util functions"""
import os
import numpy as np


def choose_free_npu(index='HBM Usage Rate', n=None):
    '''
    Call the 'npu-smi' command on Linux to look for the most available NPU
    Args:
        index: str, the index for NPU availability, default is HBM Usage Rate
        n: int, number of NPUs to return
    Returns:
        device_id: int, the device_id for the most available and healthy NPU, -1 means no healthy NPU exists
    '''
    usages = [(999, -1)]

    for i in range(8):
        # check whether the i-th NPU exists and works healthily
        info = os.popen(f'npu-smi info -t health -i {i}')

        exist_flag = False
        healthy_flag = True

        for s in info:
            if 'Health' in s:
                exist_flag = True
                if s.split(':')[-1].strip() != 'OK':
                    healthy_flag = False
                    break

        if not exist_flag: continue
        if not healthy_flag: continue

        # check the usage of the i-th NPU
        info = os.popen(f'npu-smi info -t usages -i {i}')

        for s in info:
            if index in s:
                usages.append((float(s.split(':')[-1]), i))  # record the HBM usage rate of the current NPU
                break

    if not n:
        return min(usages)[1]

    return [i for usage, i in sorted(usages)[:n]]

def sloc2mask(slocs, shape, dxs=None):
    '''
    Convert source locations to masks with numpy
    Args:
        slocs: 2d array (ns, ndim), ns is the number of source locations,
            and the last dimension indicates the coordinates in the order of (z, y, x).
        shape: 1d array (ndim)
        dxs: 1d array (ndim), the grid intervals in each dimension
    Returns:
        mask: (ndim+1)-d array, the first dimension is the batch dimension,
            and the last ndim dimensions are space dimensions
    '''
    if dxs is None:
        dxs = np.ones_like(shape)

    assert np.shape(slocs)[-1] == len(shape) == len(dxs)

    mask = np.zeros([*np.shape(slocs)[:-1], *shape])

    for i, sloc in enumerate(np.reshape(slocs, [-1, len(shape)])):
        sidx = np.rint(np.divide(sloc, dxs)).astype(int)
        mask.reshape(-1, *shape)[tuple([i, *sidx])] = 1

    return mask

def mask2sloc(masks, ndim, dxs=None):
    ''' convert masks to source locations with numpy '''
    sidxs = np.argwhere(masks)[:, -ndim:].reshape(*masks.shape[:-ndim], ndim)
    return sidxs if dxs is None else np.multiply(sidxs, dxs)
