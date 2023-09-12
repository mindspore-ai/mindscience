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
# ==============================================================================
"""dataset"""
import mindspore as ms
import numpy as np

from sciai.utils.check_utils import to_tuple
from sciai.utils.ms_utils import to_tensor


class DatasetGenerator:
    """
    Common data generator.

    Args:
        *data (any): Data to be iterated over.

    Raises:
        TypeError: If the input type is incorrect.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from sciai.common import DatasetGenerator
        >>> data = np.array(range(128)).reshape(-1, 2)
        >>> dg = DatasetGenerator(data)
        >>> print(len(dg))
        64
    """

    def __init__(self, *data):
        self.__data = to_tuple(data)

    def __getitem__(self, index):
        return to_tensor(tuple(_[index] for _ in self.__data), ms.float32)

    def __len__(self):
        return len(self.__data[0])


class Sampler:
    """
    Common data sampler.

    Args:
        dim (int): Data dimension.
        coords (Union[array, list]): Lower bound coordinate and upper bound coordinate, e.g., [[0.0, 0.0], [0.0, 1.0]].
        func (Callable): Exact solution function.
        name (str): Sampler name. Default: None.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from sciai.common import Sampler
        >>> def u(x_):
        >>>     t = x_[:, 0:1]
        >>>     x = x_[:, 1:2]
        >>>     return np.exp(-t) * np.sin(500 * np.pi * x)
        >>> ics_coords = np.array([[0.0, 0.0], [0.0, 1.0]])
        >>> ics_sampler = Sampler(2, ics_coords, u, name='Initial Condition 1')
        >>> x_batch, y_batch = ics_sampler.sample(10)
        >>> print(x_batch.shape, y_batch.shape)
        (10, 2), (10, 1)
    """

    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, n):
        """
        Sample points in given field.

        Args:
            n (int): Number of sample points.

        Returns:
            tuple[Tensor], x and y of n sample points.
        """
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(n, self.dim)
        y = self.func(x)
        return x, y

    def normalization_constants(self, n):
        """
        Normalization mean and standard deviation.

        Args:
            n (int): Number of sample points for mean and standard deviation calculation.

        Returns:
            tuple[Tensor], Mean and standard deviation of sampled points.
        """
        x, _ = self.sample(n)
        return x.mean(0), x.std(0)

    def fetch_minibatch(self, n, mu_x, sigma_x):
        """
        Fetch a minibatch of the sampler.

        Args:
            n (int): Number of sample points per minibatch.
            mu_x (int): Mean of the sample points.
            sigma_x (int): Standard deviation of the sample points.

        Returns:
            tuple[Tensor], A minibatch of normalized sample points from the sampler.
        """
        x, y = self.sample(n)
        x = (x - mu_x) / sigma_x
        return x, y
