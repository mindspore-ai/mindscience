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
# ============================================================================
"""process for multiscale pinns"""
import os

import yaml
import numpy as np

from sciai.common.dataset import Sampler
from sciai.utils import flatten_add_dim, parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


class Solution:
    """Exact solution class"""

    def __init__(self, a, b) -> None:
        """
        Args:
            a: Factor a.
            b: Factor b.
        """
        super().__init__()
        self.a, self.b = a, b

    def u(self, x_):
        """
        Args:
            x_:  t, x
        Returns: function result
        """
        t, x = x_[:, 0:1], x_[:, 1:2]
        return np.exp(-self.a * t) * np.sin(self.b * np.pi * x)

    def u_t(self, x):
        """du/dt"""
        return - self.a * self.u(x)

    def u_xx(self, x):
        """d2u/dx2"""
        return - (self.b * np.pi) ** 2 * self.u(x)

    def f(self, x):
        """f"""
        k = self.a / (self.b * np.pi) ** 2
        return self.u_t(x) - k * self.u_xx(x)


def get_data(args):
    """get training data"""
    # Parameters of equation
    a, b = 1, 500
    k = a / (b * np.pi) ** 2
    solution = Solution(a=a, b=b)
    # Hyper-parameter for Fourier feature embeddings
    sigma = 10
    # Domain boundaries, [[t_low, x_low], [t_up, x_up]]
    ics_coords = np.array([[0.0, 0.0], [0.0, 1.0]])
    bc1_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    bc2_coords = np.array([[0.0, 1.0], [1.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    # Create initial conditions samplers
    ics_sampler = Sampler(2, ics_coords, solution.u, name='Initial Condition 1')
    # Create boundary conditions samplers
    bcs_sampler1 = Sampler(2, bc1_coords, solution.u, name='Dirichlet BC1')
    bcs_sampler2 = Sampler(2, bc2_coords, solution.u, name='Dirichlet BC2')
    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, solution.f, name='Forcing')
    samplers = {"ics_sampler": ics_sampler,
                "bcs_sampler1": bcs_sampler1,
                "bcs_sampler2": bcs_sampler2,
                "res_sampler": res_sampler}
    # Test data
    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], args.nnum)[:, None]
    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], args.nnum)[:, None]
    t, x = np.meshgrid(t, x)

    x_star = np.hstack(flatten_add_dim(t, x))
    u_star = solution.u(x_star)
    f_star = solution.f(x_star)
    return x_star, f_star, u_star, samplers, k, sigma, t, x
