# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
"""test sciops einsum"""

import pytest
import numpy as np
from mindspore import ops

from mindscience.sciops import Einsum

ROTL = 1e-5

def calculate(equation, shapes, use_opt=True):
    es = Einsum(equation, use_opt=use_opt)
    tensors = [ops.randn(tp) for tp in shapes]
    return es(*tensors)

def cmp_accuracy(equation, shapes, use_opt=True):
    es = Einsum(equation, use_opt=use_opt)
    tensors = [ops.randn(tp) for tp in shapes]
    ms_res = es(*tensors)
    np_res = np.einsum(equation, *tensors)
    diff = np.abs(np_res - ms_res)
    return np.mean(diff)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_base_equation():
    """
    Feature: Einsum
    Description: test Einsum with different equations
    Expectation: success
    """
    equation = "ij->ji"
    shapes = [(15, 38)]
    res = calculate(equation, shapes)
    assert res.shape == (38, 15)

    equation = "ijkn->knji"
    shapes = [(15, 38, 123, 251)]
    res = calculate(equation, shapes)
    assert res.shape == (123, 251, 38, 15)

    equation = "ij,j->i"
    shapes = [(15, 38), (38,)]
    res = calculate(equation, shapes)
    assert res.shape == (15,)

    equation = "abcd,d->abc"
    shapes = [(15, 38, 123, 251), (251,)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 38, 123)

    equation = "ij,jk->ik"
    shapes = [(512, 1024), (1024, 512)]
    res = calculate(equation, shapes)
    assert res.shape == (512, 512)

    equation = "ij,kj->ik"
    shapes = [(15, 38), (123, 38)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 123)

    equation = "abCd,dFg->abCFg"
    shapes = [(15, 38, 123, 251), (251, 123, 38)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 38, 123, 123, 38)

    equation = "i,i->"
    shapes = [(1024,), (1024,)]
    res = calculate(equation, shapes)
    assert res.shape == ()

    equation = "ij,ij->ij"
    shapes = [(15, 38), (15, 38)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 38)

    equation = "ijkn,ijkn->ijkn"
    shapes = [(15, 38, 123, 251), (15, 38, 123, 251)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 38, 123, 251)

    equation = "ii->"
    shapes = [(256, 256)]
    res = calculate(equation, shapes)
    assert res.shape == ()

    equation = "iji->j"
    shapes = [(15, 38, 15)]
    res = calculate(equation, shapes)
    assert res.shape == (38,)

    equation = "nij,njk->nik"
    shapes = [(15, 38, 123), (15, 123, 251)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 38, 251)

    equation = "bij,jk->bik"
    shapes = [(15, 38, 123), (123, 251)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 38, 251)

    equation = "ij->i"
    shapes = [(15, 38)]
    res = calculate(equation, shapes)
    assert res.shape == (15,)

    equation = "ijkl->ik"
    shapes = [(15, 38, 123, 251)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 123)

    equation = "ijk,jk->i"
    shapes = [(15, 38, 123), (38, 123)]
    res = calculate(equation, shapes)
    assert res.shape == (15,)

    equation = "i,j->ij"
    shapes = [(15,), (38,)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 38)

    equation = "ij,ab->ijab"
    shapes = [(256, 16), (32, 16)]
    res = calculate(equation, shapes)
    assert res.shape == (256, 16, 32, 16)

    equation = "ij,jk,kl->il"
    shapes = [(256, 16), (16, 16), (16, 256)]
    res = calculate(equation, shapes)
    assert res.shape == (256, 256)

    equation = "bn,anm,bm->ba"
    shapes = [(15, 38), (15, 38, 123), (15, 123)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 15)

    equation = "ij,ij->"
    shapes = [(15, 38), (15, 38)]
    res = calculate(equation, shapes)
    assert res.shape == ()

    equation = "ijkn,ijkn->"
    shapes = [(15, 38, 123, 251), (15, 38, 123, 251)]
    res = calculate(equation, shapes)
    assert res.shape == ()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_complex_equation():
    """
    Feature: Einsum
    Description: test Einsum with different equations
    Expectation: success
    """
    equation = 'ijk,zui,zuj,zuw->zwk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (156, 32, 20)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 20, 33)

    equation = 'ijk,zui,zuj,uw->zwk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (32, 9)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 9, 33)

    equation = 'ijk,zui,zuj,zu->zuk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (156, 32)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 32, 33)

    equation = 'ijk,zui,zuj,u->zuk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (32,)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 32, 33)

    equation = 'ijk,zui,zvj,zuvw->zwk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (156, 32, 32, 9)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 9, 33)

    equation = 'ijk,zui,zvj,uvw->zwk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (32, 32, 9)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 9, 33)

    equation = 'ijk,zui,zvj,uv->zuk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (32, 32)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 32, 33)

    equation = 'ijk,zui,zvj,uv->zuvk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (32, 32)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 32, 32, 33)

    equation = 'ijk,zui,zvj,zuv->zuk'
    shapes = [(9, 9, 33), (156, 32, 9), (156, 32, 9), (156, 32, 32)]
    res = calculate(equation, shapes)
    assert res.shape == (156, 32, 33)

    equation = 'zui,zuj,kij->zuk'
    shapes = [(660, 128, 16), (660, 128, 16), (156, 16, 16)]
    res = calculate(equation, shapes)
    assert res.shape == (660, 128, 156)

    equation = 'vun,zuni->zvi'
    shapes = [(128, 128, 4), (660, 128, 4, 1)]
    res = calculate(equation, shapes)
    assert res.shape == (660, 128, 1)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ellipsis_equation():
    """
    Feature: Einsum
    Description: test Einsum with different equations including ellipsis
    Expectation: success
    """
    equation = "...bij,...jk->...bik"
    shapes = [(256, 128, 16), (16, 32)]
    res = calculate(equation, shapes)
    assert res.shape == (256, 128, 32)

    equation = "...bij,...jk->bik..."
    shapes = [(16, 256, 128, 16), (16, 16, 32)]
    res = calculate(equation, shapes)
    assert res.shape == (256, 128, 32, 16)

    equation = "...bij,j...k->...bik"
    shapes = [(15, 8, 256, 128, 16), (16, 15, 8, 32)]
    res = calculate(equation, shapes)
    assert res.shape == (15, 8, 256, 128, 32)

    equation = 'zui...,zuj,...kij->zu...k'
    shapes = [(256, 128, 16), (256, 128, 16), (156, 16, 16)]
    res = calculate(equation, shapes)
    assert res.shape == (256, 128, 156)

    equation = 'zui...,zuj,...kij->zu...k'
    shapes = [(256, 128, 16, 8), (256, 128, 16), (8, 156, 16, 16)]
    res = calculate(equation, shapes)
    assert res.shape == (256, 128, 8, 156)

    equation = 'zui...,zuj,...kij->zu...k'
    shapes = [(256, 128, 16, 8, 4), (256, 128, 16), (8, 4, 156, 16, 16)]
    res = calculate(equation, shapes)
    assert res.shape == (256, 128, 8, 4, 156)

    equation = 'zu...i,zuj,...kij->zuk'
    shapes = [(256, 128, 8, 4, 16), (256, 128, 16), (8, 4, 156, 16, 16)]
    res = calculate(equation, shapes)
    assert res.shape == (256, 128, 156)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_accuracy():
    """
    Feature: Einsum
    Description: test Einsum accuracy with different equations
    Expectation: success
    """
    equation = 'zui,zuj,kij->zuk'
    shapes = [(256, 128, 16), (256, 128, 16), (156, 16, 16)]
    diff = cmp_accuracy(equation, shapes)
    assert diff < ROTL

    equation = "ijkl->ik"
    shapes = [(9, 20, 31, 133)]
    diff = cmp_accuracy(equation, shapes)
    assert diff < ROTL

    equation = "ij,jk->ik"
    shapes = [(9, 20), (20, 31)]
    diff = cmp_accuracy(equation, shapes)
    assert diff < ROTL

    equation = "iji->j"
    shapes = [(31, 20, 31)]
    diff = cmp_accuracy(equation, shapes)
    assert diff < ROTL

    equation = "nij,njk->nik"
    shapes = [(9, 20, 31), (9, 31, 133)]
    diff = cmp_accuracy(equation, shapes)
    assert diff < ROTL

    equation = "bij,jk->bik"
    shapes = [(256, 128, 16), (16, 32)]
    diff = cmp_accuracy(equation, shapes)
    assert diff < ROTL

    equation = "iiik->ik"
    shapes = [(32, 32, 32, 128)]
    diff = cmp_accuracy(equation, shapes)
    assert diff < ROTL


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_labelorder_equation():
    """
    Feature: Einsum
    Description: test Einsum labelorder with different equations
    Expectation: success
    """
    equation = "ijk,zi->zjk"
    es = Einsum(equation, use_opt=False)
    assert es.trace == ((1, 0),)

    equation = "jik,zi->zjk"
    es = Einsum(equation, use_opt=False)
    assert es.trace == ((1, 0),)
