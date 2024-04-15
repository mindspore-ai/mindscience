# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
"""
Collective variables by position
"""

from inspect import signature

from mindspore import Tensor
from mindspore import nn, ops

from ..colvar import Colvar
from ..atoms import AtomsBase, Vector
from ...function import Units


class Distance(Colvar):
    r"""Colvar for distance.

    Args:
        atoms (AtomsBase):  Atoms of shape `(..., 2, D)` to calculate distance of shape `(...)` or `(..., 1)`.
                            Cannot be used with `atoms0` or `atoms1`. Default: ``None``.
                            `D` means spatial dimension of the simulation system. Usually is 3.
        atoms0 (AtomsBase): Initial point of atoms with shape `(..., D)` of the distance with shape
                            `(...)` or `(..., 1)`. Must be used with `atoms1`, and cannot be used with `atoms`.
                            Default: ``None``.

        atoms1 (AtomsBase): Terminal point of atoms with shape `(..., D)` of the distance with shape
                            `(...)` or `(..., 1)`. Must be used with `atoms0`, and cannot be used with `atoms`.
                            Default: ``None``.

        vector (Vector):    Vector with shape `(..., D)` of the distance with shape `(...)` or `(..., 1)`

        use_pbc (bool):     Whether to calculate distance under periodic boundary condition.
                            Default: ``None``.

        batched (bool):     Whether the first dimension of the input index in atoms is the batch size.
                            Default: ``False``.

        keepdims (bool):    If True, the last axis will be left, and the output shape will be `(..., 1)`.
                            If False, the shape of distance will be `(...)`
                            if None, its value will be determined according to the rank of vector:
                            False if the rank is greater than 1, otherwise True.
                            Default: ``None``.

        axis (int):         Axis along which the coordinate of atoms are take, of which the dimension must be 2.
                            It only works when initialized with `atoms`, `atoms0`, or `atoms1`.
                            Default: -2.

        name (str):         Name of the Colvar. Default: 'distance'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import Sponge
        >>> from sponge.colvar import Distance
        >>> from sponge.callback import RunInfo
        >>> cv_bond = Distance([0, 1])
        >>> # system is the Molecule object defined by user.
        >>> # energy is the Energy object defined by user.
        >>> # opt is the Optimizer object defined by user.
        >>> md = Sponge(system, potential=energy, optimizer=opt, metrics={'bond': cv_bond})
        >>> run_info = RunInfo(1000)
        >>> md.run(2000, callbacks=[run_info])
        [MindSPONGE] Started simulation at 2024-02-19 15:43:11
        [MindSPONGE] Step: 1000, E_pot: -117.30916, bond: 1.4806036
        [MindSPONGE] Step: 2000, E_pot: -131.60872, bond: 1.4821533
        [MindSPONGE] Finished simulation at 2024-02-19 15:44:03
        [MindSPONGE] Simulation time: 51.27 seconds.
    """

    def __init__(self,
                 atoms: AtomsBase = None,
                 atoms0: AtomsBase = None,
                 atoms1: AtomsBase = None,
                 vector: Vector = None,
                 use_pbc: bool = None,
                 batched: bool = False,
                 keepdims: bool = None,
                 axis: int = -2,
                 name: str = 'distance',
                 ):

        super().__init__(
            periodic=False,
            use_pbc=use_pbc,
            name=name,
        )

        # (..., D)
        if vector is None:
            self.vector = Vector(atoms=atoms,
                                 atoms0=atoms0,
                                 atoms1=atoms1,
                                 batched=batched,
                                 use_pbc=use_pbc,
                                 axis=axis,
                                 keepdims=False,
                                 )
        else:
            self.vector = vector

        if keepdims is None:
            if self.vector.ndim > 1:
                keepdims = False
            else:
                keepdims = True

        shape = self.vector.shape[:-1]
        if keepdims:
            shape += (1,)
        self._set_shape(shape)

        self.keepdims = keepdims

        self.norm_last_dim = None
        # MindSpore < 2.0.0-rc1
        if 'ord' not in signature(ops.norm).parameters.keys():
            self.norm_last_dim = nn.Norm(-1, self.keepdims)

    def get_unit(self, units: Units = None) -> str:
        """return unit of the collective variables"""
        return units.length_unit_name

    def construct(self, coordinate: Tensor, pbc_box: bool = None):
        r"""calculate distance.

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
                                    `B` means batchsize, i.e. number of walkers in simulation.
                                    `A` means number of atoms in system.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
                                    Default: ``None``.

        Returns:
            distance (Tensor):      Tensor of shape `(B, ...)`. Data type is float.

        """

        # (B, ..., D)
        vector = self.vector(coordinate, pbc_box)

        # (B, ...) or (B, ..., 1)
        if self.norm_last_dim is None:
            return ops.norm(vector, None, -1, self.keepdims)

        return self.norm_last_dim(vector)
