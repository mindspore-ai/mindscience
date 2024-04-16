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
from mindspore import ops, nn
from mindspore.ops import functional as F

from ...function import any_none, any_not_none
from ...function import get_integer, check_broadcast
from ..colvar import Colvar
from ..atoms import AtomsBase, Vector, get_atoms


class Angle(Colvar):
    r"""
    Colvar for angle.

    Args:
        atoms (AtomsBase):      Atoms of shape `(..., 3, D)` to form a angle of shape `(...)` or `(..., 1)`.
                                Cannot be used with `atoms_a` or `atoms_b`.
                                Default: ``None``.  `D` means spatial dimension of the simulation system. Usually is 3.

        atoms_a (AtomsBase):    Atoms A with shape `(..., D)` to form a angle of shape `(...)` or `(..., 1)`.
                                Must be used with `atoms_b` and `atoms_c`. Cannot be used with `atoms`.
                                Default: ``None``.

        atoms_b (AtomsBase):    Atoms B with shape `(..., D)` to form a angle of shape `(...)` or `(..., 1)`.
                                Must be used with `atoms_a` and `atoms_c`. Cannot be used with `atoms`.
                                Default: ``None``.

        atoms_c (AtomsBase):    Atoms C with shape `(..., D)` to form a angle of shape `(...)` or `(..., 1)`.
                                Must be used with `atoms_a` and `atoms_b`. Cannot be used with `atoms`.
                                Default: ``None``.

        vector1 (Vector):       Vector 1 of shape `(..., D)` to form of a angle with shape `(...)` or `(..., 1)`.
                                Must be used with `vector2`. Cannot be used with Atoms.
                                Default: ``None``.

        vector2 (Vector):       Vector 2 of shape `(..., D)` to form of a angle with shape `(...)` or `(..., 1)`.
                                Must be used with `vector1`. Cannot be used with Atoms.
                                Default: ``None``.

        use_pbc (bool):         Whether to calculate distance under periodic boundary condition.
                                Default: ``None``.

        batched (bool):         Whether the first dimension of the input index in atoms is the batch size.
                                Default: ``False``.

        keepdims (bool):        Whether to keep the dimension of the last dimension of vector.
                                Default: ``False``.

        axis (int):             Axis to gather the points from coordinate of atoms. Default: -2

        name (str):             Name of the Colvar. Default: 'angle'

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import Sponge
        >>> from sponge.colvar import Angle
        >>> from sponge.callback import RunInfo
        >>> cv_angle = Angle([0, 1, 2])
        >>> # system is the Molecule object defined by user.
        >>> # energy is the Energy object defined by user.
        >>> # opt is the Optimizer object defined by user.
        >>> md = Sponge(system, potential=energy, optimizer=opt, metrics={'angle': cv_angle})
        >>> run_info = RunInfo(1000)
        >>> md.run(2000, callbacks=[run_info])
        [MindSPONGE] Started simulation at 2024-02-19 15:43:11
        [MindSPONGE] Step: 1000, E_pot: -117.30916, angle: 1.9461793
        [MindSPONGE] Step: 2000, E_pot: -131.60872, angle: 1.9336755
        [MindSPONGE] Finished simulation at 2024-02-19 15:44:03
        [MindSPONGE] Simulation time: 51.27 seconds.
    """

    def __init__(self,
                 atoms: AtomsBase = None,
                 atoms_a: AtomsBase = None,
                 atoms_b: AtomsBase = None,
                 atoms_c: AtomsBase = None,
                 vector1: Vector = None,
                 vector2: Vector = None,
                 use_pbc: bool = None,
                 batched: bool = False,
                 keepdims: bool = None,
                 axis: int = -2,
                 name: str = 'angle'
                 ):

        super().__init__(
            periodic=False,
            use_pbc=use_pbc,
            name=name,
            unit='rad',
        )

        if any_not_none([atoms, atoms_a, atoms_b, atoms_c]) and any_not_none([vector1, vector2]):
            raise ValueError('The atoms and vector cannot be used at same time!')

        axis = get_integer(axis)
        self.keepdims = keepdims

        self.atoms = None
        self.vector1 = None
        self.vector2 = None
        self.split3 = None
        if any_not_none([vector1, vector2]):
            if any_none([vector1, vector2]):
                raise ValueError('vector1 must be used with vector2!')
            self.vector1 = vector1
            self.vector2 = vector2
        else:
            if atoms is None:
                if not all([atoms_a, atoms_b, atoms_c]):
                    raise ValueError
                self.vector1 = Vector(atoms0=atoms_b,
                                      atoms1=atoms_a,
                                      batched=batched,
                                      use_pbc=use_pbc,
                                      axis=axis,
                                      )
                self.vector2 = Vector(atoms0=atoms_b,
                                      atoms1=atoms_c,
                                      batched=batched,
                                      use_pbc=use_pbc,
                                      axis=axis,
                                      )
            else:
                if any_not_none([atoms_a, atoms_b, atoms_c]):
                    raise ValueError('atoms cannot be used with atoms_a, atoms_b and atoms_c!')
                self.atoms = get_atoms(atoms, batched)
                shape = (1,) + self.atoms.shape
                if shape[axis] != 3:
                    raise ValueError(f'The axis {axis} of atoms must be 3 but got: {shape[axis]}')
                self.split3 = ops.Split(axis, 3)

        if self.atoms is None:
            if self.vector1.ndim > self.vector2.ndim:
                new_shape = (1,) * (self.vector1.ndim - self.vector2.ndim)
                self.vector2.reshape(new_shape)
            if self.vector1.ndim < self.vector2.ndim:
                new_shape = (1,) * (self.vector2.ndim - self.vector1.ndim)
                self.vector1.reshape(new_shape)

            # (..., D)
            shape = check_broadcast(self.vector1.shape, self.vector2.shape)

            if self.keepdims is None:
                if len(shape) > 1:
                    self.keepdims = False
                else:
                    self.keepdims = True

            # (...)
            shape = shape[:-1]
            if self.keepdims:
                # (..., 1)
                shape += (1,)
            self._set_shape(shape)

        else:
            if self.keepdims is None:
                if self.atoms.ndim > 2:
                    self.keepdims = False
                else:
                    self.keepdims = True
            # (1, ..., 3, D)
            shape = (1,) + self.atoms.shape
            # (1, ..., D)
            shape = shape[:axis] + shape[axis+1:]
            # (...)
            shape = shape[1:-1]
            if self.keepdims:
                # (..., 1)
                shape += (1,)
            self._set_shape(shape)

            self.squeeze = ops.Squeeze(axis)

        self.norm_last_dim = None
        # MindSpore < 2.0.0-rc1
        if 'ord' not in signature(ops.norm).parameters.keys():
            self.norm_last_dim = nn.Norm(-1, self.keepdims)

        self.reduce_sum = ops.ReduceSum(self.keepdims)

    def construct(self, coordinate: Tensor, pbc_box: bool = None):
        r"""calculate angle.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    `B` means batchsize, i.e. number of walkers in simulation.
                                    `A` means Number of atoms in system.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Default: ``None``.

        Returns:
            angle (Tensor):         Tensor of shape (B, ...) or (B, ..., 1). Data type is float.
        """

        if self.atoms is None:
            # (B, ..., D)
            vector1 = self.vector1(coordinate, pbc_box)
            vector2 = self.vector2(coordinate, pbc_box)
        else:
            # (B, ..., 3, D)
            atoms = self.atoms(coordinate, pbc_box)
            # (B, ..., 1, D)
            pos_a, pos_b, pos_c = self.split3(atoms)
            # (B, ..., D) <- (B, ..., 1, D)
            pos_a = self.squeeze(pos_a)
            pos_b = self.squeeze(pos_b)
            pos_c = self.squeeze(pos_c)

            # (B, ..., D)
            vector1 = self.get_vector(pos_b, pos_a, pbc_box)
            vector2 = self.get_vector(pos_b, pos_c, pbc_box)

        # (B, ...) or (B, ..., 1) <- (B, ..., D)
        if self.norm_last_dim is None:
            dis1 = ops.norm(vector1, None, -1, self.keepdims)
            dis2 = ops.norm(vector2, None, -1, self.keepdims)
        else:
            dis1 = self.norm_last_dim(vector1)
            dis2 = self.norm_last_dim(vector2)
        dot12 = self.reduce_sum(vector1*vector2, -1)

        # (B, ...) or (B, ..., 1)
        cos_theta = dot12 / dis1 / dis2

        return F.acos(cos_theta)
