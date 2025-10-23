# Copyright 2024 Huawei Technologies Co., Ltd
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
"""radial_basis"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.mint as mint


class PolynomialEnvelope(nn.Cell):
    r"""
    Polynomial envelope function that ensures a smooth cutoff.

    Args:
        exponent (int): Exponent of the envelope function.

    Inputs:
      - **d_scaled** (Tensor): Scaled distances.
          The shape of tensor is :math:`(total\_edges)`.

    Outputs:
      - **env_val** (Tensor): The shape of tensor is :math:`(total\_edges)`.

    Raises:
        AssertionError: If exponent is not positive.
    """

    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def construct(self, d_scaled):

        env_val = mint.add(mint.add(mint.add(1, d_scaled**self.p, alpha=self.a),
                                    d_scaled**(self.p+1), alpha=self.b),
                           d_scaled**(self.p+2), alpha=self.c)
        mask = mint.lt(d_scaled, 1.0)
        env_val = mint.mul(env_val, mask)
        return env_val


class RadialBasis(nn.Cell):
    r"""
    Polynomial envelope function that ensures a smooth cutoff.

    Args:
        exponent (int): Exponent of the envelope function.

    Inputs:
        - **d_scaled** (Tensor): Scaled distances.
          The shape of tensor is :math:`(total\_edges,)`.

    Outputs:
        - **env_val** (Tensor): The shape of tensor is :math:`(total\_edges,)`.

    Raises:
        AssertionError: If exponent is not positive.
    """

    def __init__(
            self,
            num_radial,
            cutoff,
            rbf_name="gaussian",
            envelope_name="polynomial",
            envelope_exponent=5,
    ):
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        env_name = envelope_name.lower()

        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(envelope_exponent)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf_name.lower()

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(
                start=0, stop=1, num_gaussians=num_radial
            )
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def construct(self, d):
        d_scaled = mint.mul(d, self.inv_cutoff)

        env = self.envelope(d_scaled)
        return mint.mul(env.view(-1, 1), self.rbf(d_scaled))


class GaussianSmearing(ms.nn.Cell):
    r"""
    Gaussian Smearing

    Args:
        start (float): Start of the Gaussian smearing. Default: 0.0.
        stop (float): Stop of the Gaussian smearing. Default: 5.0.
        num_gaussians (int): Number of Gaussians. Default: 50.

    Inputs:
        - **dist** (Tensor): The shape of tensor is :math:`(total\_edges, num\_gaussians)`.

    Outputs:
        - **out** (Tensor): The shape of tensor is :math:`(total\_edges, num\_gaussians)`.
    """

    def __init__(
            self,
            start=0.0,
            stop=5.0,
            num_gaussians=50,
    ):
        super().__init__()
        offset = mint.arange(start, num_gaussians) / (num_gaussians/stop)
        self.coeff = -0.5 / (offset[1] - offset[0])**2
        self.offset = ms.Parameter(
            offset, name="offset", requires_grad=False).view(1, -1)

    def construct(self, dist):
        dist = mint.sub(dist.view(-1, 1), self.offset)
        return mint.exp(mint.mul(self.coeff, mint.pow(dist, 2)))
