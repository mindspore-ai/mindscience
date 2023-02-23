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
"""create dataset"""
import numpy as np

import mindspore as ms

from mindflow.geometry import generate_sampling_config, Disk, CSGXOR


class MyIterable:
    """
    A source dataset that generates data from Python by invoking Python data source each epoch.

    The column names and column types of generated dataset depend on Python data defined by users.

    Args:
        domain (Numpy.array): the input data of domain.
        bc_outer (Numpy.array): the input data of Dirichlet boundary condition.
        bc_inner (Numpy.array): the input data of Neumann boundary condition.
        bc_inner_normal (Numpy.array): the normal of the surface at a Neumann boundary point P is a vector perpendicular
            to the tangent plane of the point.

    Raise:
        StopIteration: If the iteration is done, which means the iteration is bigger than length of domain.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> domain = np.random.random(size=(3, 2))
        >>> bc_outer = np.random.random(size=(3, 2))
        >>> bc_inner = np.random.random(size=(3, 2))
        >>> bc_inner_normal = np.random.random(size=(3, 2))
        >>> dataset = ms.dataset..GeneratorDataset(source=MyIterable(domain, bc_outer, bc_outer, bc_inner_normal))
        >>> dataset.get_dataset_size()
        3
    """
    def __init__(self, domain, bc_outer, bc_inner, bc_inner_normal):
        self._index = 0
        self._domain = domain.astype(np.float32)
        self._bc_outer = bc_outer.astype(np.float32)
        self._bc_inner = bc_inner.astype(np.float32)
        self._bc_inner_normal = bc_inner_normal.astype(np.float32)

    def __next__(self):
        if self._index >= len(self._domain):
            raise StopIteration

        item = (self._domain[self._index], self._bc_outer[self._index], self._bc_inner[self._index],
                self._bc_inner_normal[self._index])
        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._domain)


def _get_region(config):
    indisk_cfg = config["in_disk"]
    in_disk = Disk(indisk_cfg["name"], (indisk_cfg["center_x"], indisk_cfg["center_y"]), indisk_cfg["radius"])
    outdisk_cfg = config["out_disk"]
    out_disk = Disk(outdisk_cfg["name"], (outdisk_cfg["center_x"], outdisk_cfg["center_y"]), outdisk_cfg["radius"])
    union = CSGXOR(out_disk, in_disk)
    return in_disk, out_disk, union


def create_training_dataset(config):
    '''create_training_dataset'''
    in_disk, out_disk, union = _get_region(config)

    union.set_sampling_config(generate_sampling_config(config["data"]))
    domain = union.sampling(geom_type="domain")

    out_disk.set_sampling_config(generate_sampling_config(config["data"]))
    bc_outer, _ = out_disk.sampling(geom_type="BC")

    in_disk.set_sampling_config(generate_sampling_config(config["data"]))
    bc_inner, bc_inner_normal = in_disk.sampling(geom_type="BC")

    dataset = ms.dataset.GeneratorDataset(source=MyIterable(domain, bc_outer, bc_inner, (-1.0) * bc_inner_normal),
                                          column_names=["data", "bc_outer", "bc_inner", "bc_inner_normal"])
    return dataset


def _numerical_solution(x, y):
    return (4.0 - x ** 2 - y ** 2) / 4


def create_test_dataset(config):
    """create test dataset"""
    _, _, union = _get_region(config)
    union.set_sampling_config(generate_sampling_config(config["data"]))
    test_data = union.sampling(geom_type="domain")
    test_label = _numerical_solution(test_data[:, 0], test_data[:, 1]).reshape(-1, 1)
    return test_data, test_label
