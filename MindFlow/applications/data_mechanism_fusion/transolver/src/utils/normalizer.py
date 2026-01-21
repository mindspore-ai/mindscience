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
"""normalizer"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor

class GaussianNormalizer:
    """
    Gaussian Normalizer.
    Normalize data to mean 0 and std 1 along the batch dimension.

    Args:
        x (numpy.ndarray): Input data to calculate mean and std. Shape should be (Batch, ...).
        eps (float): A small value to avoid division by zero. Default: 1e-5.
    """
    def __init__(self, x, eps=1e-5):
        self.mean = Tensor(np.mean(x, axis=0), mstype.float32)
        self.std = Tensor(np.std(x, axis=0), mstype.float32)
        self.eps = eps

    def encode(self, x):
        """
        Normalize input.
        Formula: (x - mean) / (std + eps)
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Normalized tensor.
        """
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        """
        Denormalize input.
        Formula: x * (std + eps) + mean
        
        Args:
            x (Tensor): Normalized tensor.
            
        Returns:
            Tensor: Denormalized (original scale) tensor.
        """
        return (x * (self.std + self.eps)) + self.mean
        