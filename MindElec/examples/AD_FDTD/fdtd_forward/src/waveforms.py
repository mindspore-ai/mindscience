# Copyright 2021 Huawei Technologies Co., Ltd
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
#pylint: disable=W0613
"""
Waveforms.
"""
from abc import ABC, abstractmethod
import numpy as np
from .utils import tensor


class BaseWaveform(ABC):
    """
    Basic class for waveforms.

    Args:
        t (numpy.ndarray, shape=(nt,)): Time series.

    Returns:
        f (Tensor, shape=(nt,)): Time-domain waveforms.
    """

    def __call__(self, t):
        """
        Compute waveforms at time t.

        Args:
            t (numpy.ndarray, shape=(nt,)): Time series.

        Returns:
            f (Tensor, shape=(nt,)): Time-domain waveforms.
        """
        return tensor(self.compute(t))

    @abstractmethod
    def compute(self, t):
        """
        Compute waveforms at time t.

        Note:
            Modify this function to define a new waveform.

        Args:
            t (numpy.ndarray, shape=(nt,)): Time series.

        Returns:
            f (numpy.ndarray, shape=(nt,)): Time-domain waveforms.
        """


class Gaussian(BaseWaveform):
    """
    Gaussian Waveform.

    Args:
        fmax (float): Maximum frequency.
    """
    def __init__(self, fmax):
        super(Gaussian, self).__init__()
        self.tau = 0.5 / fmax
        self.t0 = 4.5 * self.tau

    def compute(self, t):
        """
        Compute Gaussian waveforms at time t.

        Args:
            t (numpy.ndarray, shape=(nt,)): Time series.

        Returns:
            f (numpy.ndarray, shape=(nt,)): Time-domain waveforms.
        """
        f = np.exp(-((t - self.t0) / self.tau)**2)
        return f


class NormDGaussian(BaseWaveform):
    """
    Normalized Derivative Gaussian Waveform.

    Args:
        fmax (float): Maximum frequency.
    """
    def __init__(self, fmax):
        super(NormDGaussian, self).__init__()
        self.tau = 0.5 / fmax
        self.t0 = 4.5 * self.tau

    def compute(self, t):
        """
        Compute normalized derivative Gaussian waveforms at time t.

        Args:
            t (numpy.ndarray, shape=(nt,)): Time series.

        Returns:
            f (numpy.ndarray, shape=(nt,)): Time-domain waveforms.
        """
        f = -np.sqrt(2. * np.exp(1.)) / self.tau * (t - self.t0) * \
            np.exp(-((t - self.t0) / self.tau)**2)
        return f


class CosineGaussian(BaseWaveform):
    """
    Cosine-Modulated Gaussian Waveform.
    """

    def __init__(self, fc, df):
        super(CosineGaussian, self).__init__()
        self.tau = 2. * np.sqrt(2.3) / (np.pi * df)
        self.t0 = np.sqrt(20.) * self.tau
        self.fc = fc
        self.wc = fc * 2. * np.pi

    def compute(self, t):
        """
        Compute Cosine-Modulated Gaussian waveforms at time t.

        Args:
            t (numpy.ndarray, shape=(nt,)): Time series.

        Returns:
            f (numpy.ndarray, shape=(nt,)): Time-domain waveforms.
        """
        f = np.cos(self.wc * (t - self.t0)) * np.exp(-((t - self.t0) / self.tau)**2)
        return f
