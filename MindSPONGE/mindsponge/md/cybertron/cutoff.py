# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
#
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""cybertron cutoff"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from .units import units

__all__ = [
    "CosineCutoff",
    "MollifierCutoff",
    "HardCutoff",
    "SmoothCutoff",
    "GaussianCutoff",
    "get_cutoff",
]

_CUTOFF_ALIAS = dict()


def _cutoff_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _CUTOFF_ALIAS:
            _CUTOFF_ALIAS[name] = cls

        for alias in aliases:
            if alias not in _CUTOFF_ALIAS:
                _CUTOFF_ALIAS[alias] = cls

        return cls

    return alias_reg


class Cutoff(nn.Cell):
    """Cutoff"""
    def __init__(self,
                 r_max=units.length(1, 'nm'),
                 r_min=0,
                 hyperparam='default',
                 return_mask=False,
                 reverse=False
                 ):
        super().__init__()
        self.name = 'cutoff'
        self.hyperparam = hyperparam
        self.r_min = r_min
        self.cutoff = r_max
        self.return_mask = return_mask
        self.reverse = reverse


@_cutoff_register('cosine')
class CosineCutoff(Cutoff):
    r"""Class of Behler cosine cutoff.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self,
                 r_max=units.length(1, 'nm'),
                 r_min='default',
                 hyperparam='default',
                 return_mask=False,
                 reverse=False
                 ):
        super().__init__(
            r_max=r_max,
            r_min=r_min,
            hyperparam=None,
            return_mask=return_mask,
            reverse=reverse,
        )

        self.name = 'cosine cutoff'
        self.pi = Tensor(np.pi, ms.float32)
        self.cos = P.Cos()
        self.logical_and = P.LogicalAnd()

    def construct(self, distances, neighbor_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function

        cuts = 0.5 * (self.cos(distances * self.pi / self.cutoff) + 1.0)
        if self.reverse:
            cuts = 1.0 - cuts
            ones = F.ones_like(cuts)
            cuts = F.select(distances < cuts, cuts, ones)
            if neighbor_mask is None:
                mask = distances >= 0
            else:
                mask = neighbor_mask
        else:
            mask = distances < self.cutoff
            if neighbor_mask is not None:
                mask = self.logical_and(mask, neighbor_mask)

        # Remove contributions beyond the cutoff radius
        cutoffs = cuts * mask

        if self.return_mask:
            return cutoffs, mask
        return cutoffs


@_cutoff_register('mollifier')
class MollifierCutoff(Cutoff):
    r"""Class for mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): Cutoff radius.
        eps (float, optional): offset added to distances for numerical stability.

    """

    def __init__(self,
                 r_max=units.length(1, 'nm'),
                 r_min='default',
                 hyperparam='default',
                 return_mask=False,
                 reverse=False
                 ):
        super().__init__(
            r_min=r_min,
            r_max=r_max,
            hyperparam=hyperparam,
            return_mask=return_mask,
            reverse=reverse,
        )

        self.name = "Mollifier cutoff"

        if hyperparam == 'default':
            self.eps = units.length(1.0e-8, 'nm')
        else:
            self.eps = hyperparam

        self.exp = P.Exp()
        self.logical_and = P.LogicalAnd()

    def construct(self, distances, neighbor_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """

        exponent = 1.0 - 1.0 / (1.0 - F.square(distances / self.cutoff))
        cutoffs = self.exp(exponent)

        if self.reverse:
            cutoffs = 1. - cutoffs
            ones = F.ones_like(cutoffs)
            cutoffs = F.select(distances < self.cutoff, cutoffs, ones)
            if neighbor_mask is None:
                mask = (distances + self.eps) >= 0
            else:
                mask = neighbor_mask
        else:
            mask = (distances + self.eps) < self.cutoff
            if neighbor_mask is not None:
                mask = self.logical_and(mask, neighbor_mask)

        cutoffs = cutoffs * mask

        return cutoffs, mask


@_cutoff_register('hard')
class HardCutoff(Cutoff):
    r"""Class of hard cutoff.

    .. math::
       f(r) = \begin{cases}
        1 & r \leqslant r_\text{cutoff} \\
        0 & r > r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.

    """

    def __init__(self,
                 r_max=units.length(1, 'nm'),
                 r_min=0,
                 hyperparam='default',
                 return_mask=False,
                 reverse=False
                 ):
        super().__init__(
            r_min=r_min,
            r_max=r_max,
            hyperparam=None,
            return_mask=return_mask,
            reverse=reverse,
        )

        self.name = "Hard cutoff"
        self.logical_and = P.LogicalAnd()

    def construct(self, distances, neighbor_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """

        if self.reverse:
            mask = distances >= self.cutoff
        else:
            mask = distances < self.cutoff

        if neighbor_mask is not None:
            self.logical_and(mask, neighbor_mask)

        if self.return_mask:
            return F.cast(mask, distances.dtype), mask
        return F.cast(mask, distances.dtype)


@_cutoff_register('smooth')
class SmoothCutoff(Cutoff):
    r"""Class of smooth cutoff by Ebert, D. S. et al:
        [ref] Ebert, D. S.; Musgrave, F. K.; Peachey, D.; Perlin, K.; Worley, S.
        Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003

    ..  math::
        r_min < r < r_max:
        f(r) = 1.0 -  6 * ( r / r_cutoff ) ^ 5
                   + 15 * ( r / r_cutoff ) ^ 4
                   - 10 * ( r / r_cutoff ) ^ 3
        r >= r_max: f(r) = 0
        r <= r_min: f(r) = 1

        reverse:
        r_min < r < r_max:
        f(r) =     6 * ( r / r_cutoff ) ^ 5
                - 15 * ( r / r_cutoff ) ^ 4
                + 10 * ( r / r_cutoff ) ^ 3
        r >= r_max: f(r) = 1
        r <= r_min: f(r) = 0

    Args:
        d_max (float, optional): the maximum distance (cutoff radius).
        d_min (float, optional): the minimum distance

    """

    def __init__(self,
                 r_max=units.length(1, 'nm'),
                 r_min=0,
                 hyperparam='default',
                 return_mask=False,
                 reverse=False
                 ):
        super().__init__(
            r_min=r_min,
            r_max=r_max,
            hyperparam=None,
            return_mask=return_mask,
            reverse=reverse,
        )

        if self.r_min >= self.cutoff:
            raise ValueError(
                'dis_min must be smaller than cutoff at SmmothCutoff')

        self.dis_range = self.cutoff - self.r_min

        self.pow = P.Pow()
        self.logical_and = P.LogicalAnd()

    def construct(self, distance, neighbor_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor or float): values of interatomic distances.

        Returns:
            mindspore.Tensor or float: values of cutoff function.

        """
        dd = distance - self.r_min
        dd = dd / self.dis_range
        cuts = -  6. * self.pow(dd, 5) \
            + 15. * self.pow(dd, 4) \
            - 10. * self.pow(dd, 3)

        if self.reverse:
            cutoffs = -cuts
            mask_upper = distance < self.cutoff
            mask_lower = distance > self.r_min
        else:
            cutoffs = 1 + cuts
            mask_upper = distance > self.r_min
            mask_lower = distance < self.cutoff

        if neighbor_mask is not None:
            mask_lower = self.logical_and(mask_lower, neighbor_mask)

        zeros = F.zeros_like(distance)
        ones = F.ones_like(distance)

        cutoffs = F.select(mask_upper, cutoffs, ones)
        cutoffs = F.select(mask_lower, cutoffs, zeros)

        if self.return_mask:
            return cutoffs, mask_lower
        return cutoffs


@_cutoff_register('gaussian')
class GaussianCutoff(Cutoff):
    r"""Class of hard cutoff.

    .. math::
       f(r) = \begin{cases}
        1 & r \leqslant r_\text{cutoff} \\
        0 & r > r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.

    """

    def __init__(self,
                 r_max=units.length(1, 'nm'),
                 r_min=0,
                 hyperparam='default',
                 return_mask=False,
                 reverse=False
                 ):
        super().__init__(
            r_min=r_min,
            r_max=r_max,
            hyperparam=hyperparam,
            return_mask=return_mask,
            reverse=reverse,
        )

        if hyperparam == 'default':
            self.sigma = units.length(1, 'nm')
        else:
            self.sigma = hyperparam

        self.sigma2 = self.sigma * self.sigma

        self.exp = P.Exp()
        self.logical_and = P.LogicalAnd()

    def construct(self, distance, neighbor_mask=None):
        """construct"""
        dd = distance - self.cutoff
        dd2 = dd * dd

        gauss = self.exp(-0.5 * dd2 / self.sigma2)

        if self.reverse:
            cuts = gauss
            ones = F.ones_like(cuts)
            cuts = F.select(distance < self.cutoff, cuts, ones)

            if neighbor_mask is None:
                mask = distance >= 0
            else:
                mask = neighbor_mask
        else:
            cuts = 1. - gauss
            mask = distance < self.cutoff
            if neighbor_mask is not None:
                mask = self.logical_and(mask, neighbor_mask)

        cuts = cuts * mask

        if self.return_mask:
            return cuts, mask
        return cuts


def get_cutoff(obj, r_max=units.length(1, 'nm'), r_min=0, hyperparam='default', return_mask=False, reverse=False):
    """get cutoff"""
    if obj is None or isinstance(obj, Cutoff):
        return obj
    if isinstance(obj, str):
        if obj not in _CUTOFF_ALIAS.keys():
            raise ValueError(
                "The class corresponding to '{}' was not found.".format(obj))
        return _CUTOFF_ALIAS[obj.lower()](
            r_min=r_min,
            r_max=r_max,
            hyperparam=hyperparam,
            return_mask=return_mask,
            reverse=reverse,
        )
    raise TypeError("Unsupported Cutoff type '{}'.".format(type(obj)))
