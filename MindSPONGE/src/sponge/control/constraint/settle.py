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
SETTLE Constraint algorithm
"""

from typing import Union, List, Tuple

import numpy as np
from numpy import ndarray
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import ops, Tensor
from mindspore.ops import functional as F

from . import Constraint
from ...system import Molecule
from ...function import get_ms_array


class EinsumWrapper(ms.nn.Cell):
    r"""
    Implement particular Einsum operation.

    Args:
        equation (str):  an equation representing the operation.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, equation: str):
        super().__init__(auto_prefix=False)
        self.equation = equation

    def construct(self, xy: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculation for Einsum operation"""

        result = None
        if self.equation == 'ijk,ijk->ij':
            ijk1, ijk2 = xy
            ij = ops.ReduceSum()(ijk1*ijk2, -1)
            result = ij
        elif self.equation == 'ijk,ijl->ikl':
            ijk, ijl = xy
            ijkl1 = ijk[..., None].broadcast_to(ijk.shape + ijl.shape[-1:])
            ijkl2 = ijl[..., None, :].broadcast_to(ijl.shape[:2] + ijk.shape[-1:] + ijl.shape[-1:])
            result = (ijkl1 * ijkl2).sum(axis=1)
        else:
            raise NotImplementedError("This equation is not implemented")
        return result


class SETTLE(Constraint):
    """
    SETTLE constraint controller.

    Reference Shuichi Miyamoto and Peter A. Kollman.
    SETTLE An Analytical Version of the SHAKE and RATTLE Algorithm for Rigid Water Models.

    Args:
        system (:class:`sponge.system.Molecule`): Simulation system.
        index (Union[Tensor, ndarray, List[int]], optional): Array of settle index
          of shape :math:`(C, 3)` or :math:`(B, C, 3)`, and the data type is int.
          If ``None`` is given, the `settle_index` in `system` will be used. Default: ``None``.
        distance (Union[Tensor, ndarray, List[float]], optional): Array of settle distance
          of shape :math:`(C, 2)` or :math:`(B, C, 2)`, and the type is float.
          If ``None`` is given, the `settle_dis` in `system` will be used. Default: ``None``.

    Inputs:
        - **coordinate** (Tensor) - Coordinate. Tensor of shape :math:`(B, A, D)`.
          Data type is float.
          Here :math:`B` is the number of walkers in simulation,
          :math:`A` is the number of atoms and
          :math:`D` is the spatial dimension of the simulation system, which is usually 3.
        - **velocity** (Tensor) - Velocity. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **force** (Tensor) - Force. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **energy** (Tensor) - Energy. Tensor of shape :math:`(B, 1)`. Data type is float.
        - **kinetics** (Tensor) - Kinetics. Tensor of shape :math:`(B, D)`. Data type is float.
        - **virial** (Tensor) - Virial. Tensor of shape :math:`(B, D)`. Data type is float.
        - **pbc_box** (Tensor) - Pressure boundary condition box. Tensor of shape :math:`(B, D)`.
          Data type is float.
        - **step** (int) - Simulation step. Default: ``0``.

    Outputs:
        - coordinate, Tensor of shape :math:`(B, A, D)`. Coordinate. Data type is float.
        - velocity, Tensor of shape :math:`(B, A, D)`. Velocity. Data type is float.
        - force, Tensor of shape :math:`(B, A, D)`. Force. Data type is float.
        - energy, Tensor of shape :math:`(B, 1)`. Energy. Data type is float.
        - kinetics, Tensor of shape :math:`(B, D)`. Kinetics. Data type is float.
        - virial, Tensor of shape :math:`(B, D)`. Virial. Data type is float.
        - pbc_box, Tensor of shape :math:`(B, D)`. Periodic boundary condition box.
          Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import Molecule
        >>> from sponge.control import SETTLE
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> controller = SETTLE(system)
    """

    def __init__(self,
                 system: Molecule,
                 index: Union[Tensor, ndarray, List[int]] = None, # (B, C, 3)
                 distance: Union[Tensor, ndarray, List[float]] = None, # (B, C, 2)
                 ):
        super(SETTLE, self).__init__(system, bonds='all-bonds')
        print('[MindSPONGE] The settle constraint is used for the molecule system.')

        # pylint: disable=invalid-name
        bsize = system.num_walker
        def _check_shape(value: Tensor, name: str, dim: int) -> Tensor:
            if value.ndim < 2 or value.ndim > 3:
                raise ValueError(f'The rank(ndim) of {name} should be 2 or 3 but got {value.ndim}')
            if value.ndim == 2:
                value = F.expand_dims(value, 0)
            if value.shape[0] != 1 and value.shape[0] != bsize:
                raise ValueError(f'The first dimension of {name} must be equal to 1 or '
                                 f'the number of multiple walkers of the simulation system ({bsize})'
                                 f'but got: {value.shape[0]}')
            if value.shape[-1] != dim:
                raise ValueError(f'The last dimension of {name} must be {dim} but got {value.shape[-1]}')
            return value

        index = get_ms_array(index, ms.int32)
        if index is None:
            index = system.settle_index
        index = _check_shape(index, 'index', 3)

        distance = get_ms_array(distance, ms.float32)
        if distance is None:
            distance = system.settle_length
        distance = _check_shape(distance, 'distance', 2)

        num_settle = index.shape[1]

        if distance.shape[-2] != num_settle:
            raise ValueError(f'The number of constraint in `distance` ({distance.shape[-2]}) does not match '
                             f'the number of constraint in `index` ({num_settle}).')

        self.num_constraints = num_settle * 3

        # (B, C)
        dis_legs = distance[..., 0]
        dis_base = distance[..., 1]

        # (B, C * 3)
        self.settle_index = index.reshape((bsize, -1))
        self.bs_index = ops.broadcast_to(self.settle_index[..., None], (bsize, self.settle_index.shape[1], 3))
        self._mass = msnp.take_along_axis(system.atom_mass,
                                          self.settle_index,
                                          axis=-1).reshape((bsize, num_settle, 3, 1))

        # (B, C, 3)
        self.target_yvector = Tensor(np.array([[[0, 1, 0]] * num_settle] * bsize, np.float32))
        self.target_zvector = Tensor(np.array([[[0, 0, 1]] * num_settle] * bsize, np.float32))

        # (B, 1, 4)
        self.identity_quaternion = Tensor([[[1, 0, 0, 0]]] * bsize, ms.float32)
        self.inverse_quaternion = Tensor([[[1, -1, -1, -1]]] * bsize, ms.float32)

        hx = Tensor(0.5 * dis_base.astype(np.float32), ms.float32)
        hy = Tensor(np.sqrt(dis_legs ** 2 - (0.5 * dis_base) ** 2).astype(np.float32), ms.float32)

        # (B, C, 1)
        self.mass_vertex = self._mass[..., 0, :]
        self.mass_base = self._mass[..., 1, :]

        # (B, C)
        self.rb = hy / (1 + 2 * self.mass_base[..., 0] / self.mass_vertex[..., 0])
        self.ra = hy - self.rb
        self.rc = hx

        # (B, C, 3, 3)
        self.crd_1 = msnp.zeros((bsize, num_settle, 3, 3), ms.float32)
        self.crd_1[..., 0, 1] = self.ra
        self.crd_1[..., 1, 1] = -self.rb
        self.crd_1[..., 2, 1] = -self.rb
        self.crd_1[..., 1, 0] = -self.rc
        self.crd_1[..., 2, 0] = self.rc

        self.aindices = Tensor([0, 1], ms.int32)
        self.bindices = Tensor([0, 2], ms.int32)
        self.cindices = Tensor([0, 3], ms.int32)
        self.hindices = Tensor([1, 2], ms.int32)

        self.dot = EinsumWrapper('ijk,ijk->ij')
        self.einsum = EinsumWrapper('ijk,ijl->ikl')
        self.scatter_update = ops.tensor_scatter_elements


    def get_vector_transform(self, vec1: Tensor, vec2: Tensor):
        r"""
        Get the transform quaternion from a vector to another.

        Args:
            vec1 (Tensor): The initial vector.
            vec2 (Tensor): The target vector.

        Returns:
            Tensor, the transform quaternion.
        """
        # (B, G, D)
        cross_vector = msnp.cross(vec1, vec2, axisc=-1)
        # (B, G, 1)
        cross_norm = msnp.norm(cross_vector, axis=-1, keepdims=True)
        zero_index = msnp.where(cross_norm == 0, 1, 0)
        qs = msnp.zeros((vec1.shape[0], vec1.shape[0], 1))
        # (B, G, D)
        qv = msnp.zeros_like(vec1)
        # (B, G, 1)
        qs += msnp.norm(vec1, axis=-1, keepdims=True) * msnp.norm(vec2, axis=-1, keepdims=True)
        qs += self.dot((vec1, vec2))[..., None]
        # (B, G, D)
        qv += cross_vector
        # (B, G, 4)
        q = msnp.concatenate((qs, qv), axis=-1)
        q = q * (1 - zero_index) + self.identity_quaternion * zero_index
        q /= msnp.norm(q, axis=-1, keepdims=True)
        return q

    def get_inverse(self, quater: Tensor):
        r"""
        Get the inverse operation of a given quaternion.

        Args:
            quater (Tensor): The given quaternion.

        Returns:
            Tensor, :math:`quater^{-1}`.
        """
        factor = msnp.norm(quater, axis=-1, keepdims=True) ** 2
        return quater * self.inverse_quaternion / factor

    def quaternion_multiply(self, tensor_1: Tensor, tensor_2: Tensor):
        r"""
        Calculate the quaternion multiplication.

        Args:
            tensor_1 (Tensor): The first quaternion,
                if the size of last dimension is 3, it will be padded to 4 auto.
            tensor_2 (Tensor): The second quaternion,
                if the size of last dimension is 3, it will be padded to 4 auto.

        Returns:
            Tensor, the quaternion product of tensor_1 and tensor_2.
        """
        # (B, G, 4)
        if tensor_1.shape[-1] == 3:
            tensor_1 = msnp.pad(tensor_1, ((0, 0), (0, 0), (1, 0)), mode='constant', constant_value=0)
        if tensor_2.shape[-1] == 3:
            tensor_2 = msnp.pad(tensor_2, ((0, 0), (0, 0), (1, 0)), mode='constant', constant_value=0)
        # (B, G, 1)
        s_1 = tensor_1[..., [0]]
        s_2 = tensor_2[..., [0]]
        # (B, G, 3)
        v_1 = tensor_1[..., 1:]
        v_2 = tensor_2[..., 1:]
        # (B, G, 1)
        s = s_1 * s_2
        # (B, G, 1)
        d = self.dot((v_1, v_2))[..., None]
        s -= d
        # (B, G, 3)
        v = msnp.zeros_like(v_1)
        v += s_1 * v_2
        v += v_1 * s_2
        v += msnp.cross(v_1, v_2, axisc=-1)
        q = msnp.concatenate((s, v), axis=-1)
        return q

    def hamiltonian_product(self, q, v):
        r"""
        Perform Hamiltonian product.

        Args:
            q (Tensor): The transform quaternion.
            v (Tensor): The vector to be transformed.

        Returns:
            Tensor, The Hamiltonian product of q and v, :math:`q v q^{-1}`.
        """
        # (B, G, 4)
        iq = self.get_inverse(q)
        op1 = self.quaternion_multiply(v, iq)
        res = self.quaternion_multiply(q, op1)
        return res

    def group_hamiltonian_product(self, q, vec):
        r"""
        Perform hamiltonian product in a 4-dimensional vector.

        Args:
            q (Tensor): The transform quaternion.
            vec (Tensor): The vector to be transformed.

        Returns:
            Tensor, The Hamiltonian product of q and v, :math:`q v q^{-1}`.
        """
        # (B, G, 1, 4)
        group_q = q[..., None, :]
        # (B, G, a, 4)
        pad_q = msnp.pad(group_q, ((0, 0), (0, 0), (0, 2), (0, 0)), mode='wrap')
        # (B, G*a, 4)
        res_q = pad_q.reshape((q.shape[0], q.shape[1]*3, q.shape[2]))
        iq = self.get_inverse(res_q)
        op1 = self.quaternion_multiply(vec.reshape((q.shape[0], q.shape[1]*3, vec.shape[2])), iq)
        # (B, G, a, D)
        res = self.quaternion_multiply(res_q, op1)[..., 1:].reshape(vec.shape)
        return res


    def get_transform(self, crd_):
        r"""
        Get the transform between A0B0C0 and a0b0c0.

        Args:
            crd_ (Tensor): The given water molecular coordinates.

        Returns:
            This function will return the transform and inverse transform from crd_ SETTLE axis.
        """
        # (B, G, A, D)
        crd = crd_.reshape((crd_.shape[0], -1, 3, crd_.shape[-1]))
        # (B, G, D)
        mass_center = msnp.sum(crd * self._mass, axis=-2) / self._mass.sum(axis=-2)
        # (B, G, D)
        cross_vector_1 = crd[..., 1, :] - crd[..., 0, :]
        cross_vector_2 = crd[..., 2, :] - crd[..., 1, :]
        cross_vector = msnp.cross(cross_vector_1, cross_vector_2, axisc=-1)
        cross_vector /= msnp.norm(cross_vector, axis=-1, keepdims=True)
        # (B, G, 4)
        transform_1 = self.get_vector_transform(cross_vector, self.target_zvector)

        inverse_transform_1 = self.get_inverse(transform_1)
        # (B, G, D)
        oa_vector = crd[..., 0, :] - mass_center
        oa_vector /= msnp.norm(oa_vector, axis=-1, keepdims=True)
        transformed_oa_vector = self.hamiltonian_product(transform_1, oa_vector)[..., 1:]
        transformed_oa_vector /= msnp.norm(transformed_oa_vector, axis=-1, keepdims=True)
        # (B, G, 4)
        transform_2 = self.get_vector_transform(transformed_oa_vector, self.target_yvector)
        inverse_transform_2 = self.get_inverse(transform_2)
        # (B, G, 4)
        combine_transform = self.quaternion_multiply(transform_2, transform_1)
        combine_itransform = self.quaternion_multiply(inverse_transform_1, inverse_transform_2)
        return combine_transform, combine_itransform

    def apply_transform(self, q: Tensor, vec: Tensor):
        r"""
        Apply the quaternion transform q to a given vector vec.

        Args:
            q (Tensor): The transform quaternion.
            vec (Tensor): The vector to be transformed.

        Returns:
            Tensor, The transformed vector.
        """
        # (B, G, a, D)
        crd = vec.reshape((vec.shape[0], -1, 3, vec.shape[-1]))
        crd -= self.get_mass_center(crd)
        # (B, G, a, 1)
        norm_crd = msnp.norm(crd, axis=-1, keepdims=True)
        # (B, G, a, D)
        crd /= norm_crd
        transform_crd = self.group_hamiltonian_product(q, crd)
        res = transform_crd * norm_crd
        # (B, A, D)
        return res.reshape(vec.shape)

    def _rotation(self, crd_1_0_, crd_1_1_):
        """ Get the rotation quaternion between triangles. """
        crd_1_0 = crd_1_0_.reshape((crd_1_0_.shape[0], -1, 3, crd_1_0_.shape[-1]))
        crd_1_1 = crd_1_1_.reshape((crd_1_1_.shape[0], -1, 3, crd_1_1_.shape[-1]))
        # (B, G)
        sin_phi = crd_1_1[..., 0, 2] / self.ra
        sin_phi = sin_phi.clip(-0.999999, 0.999999)
        phi = msnp.arcsin(sin_phi)
        sin_psi = (crd_1_1[..., 1, 2] - crd_1_1[..., 2, 2]) / 2 / self.rc / msnp.sqrt(1 - sin_phi ** 2)
        sin_psi = sin_psi.clip(-0.999999, 0.999999)
        psi = msnp.arcsin(sin_psi)
        xb2 = -self.rc * msnp.sqrt(1 - sin_psi ** 2)
        yb2 = -self.rb * msnp.sqrt(1 - sin_phi ** 2) - self.rc * sin_phi * sin_psi
        yc2 = -self.rb * msnp.sqrt(1 - sin_phi ** 2) + self.rc * sin_phi * sin_psi
        alpha = xb2 * (crd_1_0[..., 1, 0] - crd_1_0[..., 2, 0]) + \
                yb2 * (crd_1_0[..., 1, 1] - crd_1_0[..., 0, 1]) + \
                yc2 * (crd_1_0[..., 2, 1] - crd_1_0[..., 0, 1])
        beta = xb2 * (crd_1_0[..., 2, 1] - crd_1_0[..., 1, 1]) + \
               yb2 * (crd_1_0[..., 1, 0] - crd_1_0[..., 0, 0]) + \
               yc2 * (crd_1_0[..., 2, 0] - crd_1_0[..., 0, 0])
        gamma = (crd_1_0[..., 1, 0] - crd_1_0[..., 0, 0]) * crd_1_1[..., 1, 1] - \
                crd_1_1[..., 1, 0] * (crd_1_0[..., 1, 1] - crd_1_0[..., 0, 1]) + \
                (crd_1_0[..., 2, 0] - crd_1_0[..., 0, 0]) * crd_1_1[..., 2, 1] - \
                crd_1_1[..., 2, 0] * (crd_1_0[..., 2, 1] - crd_1_0[..., 0, 1])
        sin_theta = (alpha * gamma - beta * msnp.sqrt(alpha ** 2 + beta ** 2 - gamma ** 2)) / (alpha ** 2 + beta ** 2)
        sin_theta = sin_theta.clip(-0.999999, 0.999999)
        theta = msnp.arcsin(sin_theta)

        # (B, G, 4)
        cos_phi = msnp.pad(msnp.cos(phi[..., None] / 2), ((0, 0), (0, 0), (0, 3)), mode='constant', constant_value=0)
        sin_phi = msnp.pad(msnp.sin(phi[..., None] / 2), ((0, 0), (0, 0), (0, 3)), mode='constant', constant_value=0)
        cos_psi = msnp.pad(msnp.cos(psi[..., None] / 2), ((0, 0), (0, 0), (0, 3)), mode='constant', constant_value=0)
        sin_psi = msnp.pad(msnp.sin(psi[..., None] / 2), ((0, 0), (0, 0), (0, 3)), mode='constant', constant_value=0)
        cos_theta = msnp.pad(msnp.cos(theta[..., None] / 2), ((0, 0), (0, 0), (0, 3)), mode='constant',
                             constant_value=0)
        sin_theta = msnp.pad(msnp.sin(theta[..., None] / 2), ((0, 0), (0, 0), (0, 3)), mode='constant',
                             constant_value=0)

        phi_minus_value = -sin_phi[..., self.aindices]
        psi_minus_value = -sin_psi[..., self.bindices]
        theta_minus_value = -sin_theta[..., self.cindices]
        phi_add_value = sin_phi[..., self.aindices[::-1]]
        psi_add_value = sin_psi[..., self.bindices[::-1]]
        theta_add_value = sin_theta[..., self.cindices[::-1]]

        sin_phi[..., self.aindices] += phi_minus_value
        sin_phi[..., self.aindices] += phi_add_value
        sin_psi[..., self.bindices] += psi_minus_value
        sin_psi[..., self.bindices] += psi_add_value
        sin_theta[..., self.cindices] += theta_minus_value
        sin_theta[..., self.cindices] += theta_add_value

        # (B, 4)
        quater_phi = cos_phi + sin_phi
        quater_psi = cos_psi + sin_psi
        quater_theta = cos_theta + sin_theta

        rotation = self.quaternion_multiply(quater_theta, self.quaternion_multiply(quater_phi, quater_psi))
        rotation /= msnp.norm(rotation, axis=-1, keepdims=True)
        return rotation

    def _swap_h(self, crd_1_1_, crd_1_3_):
        """ Swap the H atoms if the resulted hydrogen from SETTLE are mixed. """
        # (B, G, a, D)
        #pylint: disable=invalid-name
        crd_1_1 = crd_1_1_.reshape((crd_1_1_.shape[0], -1, 3, crd_1_1_.shape[-1]))
        crd_1_3 = crd_1_3_.reshape((crd_1_3_.shape[0], -1, 3, crd_1_3_.shape[-1]))
        # (B, G)
        Hs_0_0 = crd_1_1[..., 1, 2]
        Hs_0_1 = crd_1_1[..., 2, 2]
        mask_0 = Hs_0_0 >= Hs_0_1
        Hs_1_0 = crd_1_3[..., 1, 2]
        Hs_1_1 = crd_1_3[..., 2, 2]
        mask_1 = Hs_1_0 >= Hs_1_1
        # (B, G, 1, 1)
        H_swap_mask = msnp.where(mask_0 == mask_1, 0, 1)[..., None, None]
        # (B, G, 2, D)
        h_minus_value = -crd_1_3[..., self.hindices, :] * H_swap_mask
        h_add_value = crd_1_3[..., self.hindices[::-1], :] * H_swap_mask
        # Swap the coordinate of hydrogen
        crd_1_3[..., self.hindices, :] += h_minus_value
        crd_1_3[..., self.hindices, :] += h_add_value

        new_crd = crd_1_3.reshape(crd_1_3_.shape)
        return new_crd

    def get_mass_center(self, crd_):
        r"""
        Get the mass center of a given molecule.

        Args:
            crd_ (Tensor): The coordinates.

        Returns:
            Tensor, The mass center of the molecule.
        """
        # (B, G, A, D)
        crd = crd_.reshape((crd_.shape[0], -1, 3, crd_.shape[-1]))
        # (B, G, 1, D)
        cm = msnp.sum(crd * self._mass, axis=-2) / self._mass.sum(axis=-2)
        return cm[..., None, :]

    def get_vel_force_update(self, crd0_, vel0_):
        r"""
        Get the update of velocity and force.

        Args:
            crd0_ (Tensor): The coordinate after SETTLE in the origin axis.
            vel0_ (Tensor): The initial velocity.

        Returns:
            Tensor, The constraint velocity.
            Tensor, The constraint force.
        """
        #pylint: disable=invalid-name
        crd0 = crd0_.reshape((crd0_.shape[0], -1, 3, 3))
        vel0 = vel0_.reshape((vel0_.shape[0], -1, 3, 3))
        # (B, C, 1)
        AB = msnp.norm(crd0[..., 1, :] - crd0[..., 0, :], axis=-1, keepdims=True)
        AC = msnp.norm(crd0[..., 2, :] - crd0[..., 0, :], axis=-1, keepdims=True)
        BC = msnp.norm(crd0[..., 2, :] - crd0[..., 1, :], axis=-1, keepdims=True)

        # (B, C, D)
        e_AB = (crd0[..., 1, :] - crd0[..., 0, :]) / AB
        e_CA = (crd0[..., 0, :] - crd0[..., 2, :]) / AC
        e_BC = (crd0[..., 2, :] - crd0[..., 1, :]) / BC

        # (B, C)
        cosA = (AB ** 2 + AC ** 2 - BC ** 2) / 2 / AB / AC
        cosB = (AB ** 2 + BC ** 2 - AC ** 2) / 2 / AB / BC
        cosC = (BC ** 2 + AC ** 2 - AB ** 2) / 2 / BC / AC

        # (B, C)
        d = (2 * (self.mass_vertex + self.mass_base) ** 2 +
             2 * self.mass_vertex * self.mass_base * cosA * cosB * cosC -
             2 * (self.mass_base ** 2) * (cosA ** 2) -
             self.mass_vertex * (self.mass_vertex + self.mass_base) * (cosB ** 2 +cosC ** 2)
             ) * self.time_step * 0.5 / self.mass_base

        # (B, C, D)
        v_AB_ = vel0[..., 1, :] - vel0[..., 0, :]
        v_CA_ = vel0[..., 0, :] - vel0[..., 2, :]
        v_BC_ = vel0[..., 2, :] - vel0[..., 1, :]

        # (B, C, 1)
        v_AB = self.dot((e_AB, v_AB_))[..., None]
        v_CA = self.dot((e_CA, v_CA_))[..., None]
        v_BC = self.dot((e_BC, v_BC_))[..., None]

        # (B, C)
        tau_AB = (v_AB * (2 * (self.mass_vertex + self.mass_base) - self.mass_vertex * cosC ** 2) +
                  v_BC * (self.mass_base * cosC * cosA - (self.mass_vertex + self.mass_base) * cosB) +
                  v_CA * (self.mass_vertex * cosB * cosC - 2 * self.mass_base * cosA)
                  ) * self.mass_vertex / d

        tau_BC = (v_BC * ((self.mass_vertex + self.mass_base) ** 2 - (self.mass_base ** 2) * (cosA ** 2)) +
                  v_CA * self.mass_vertex * (self.mass_base * cosA * cosB -
                                             (self.mass_vertex + self.mass_base) * cosC) +
                  v_AB * self.mass_vertex * (self.mass_base * cosC * cosA -
                                             (self.mass_vertex + self.mass_base) * cosB)) / d

        tau_CA = (v_CA * (2 * (self.mass_vertex + self.mass_base) - self.mass_vertex * cosB ** 2) +
                  v_AB * (self.mass_vertex * cosB * cosC - 2 * self.mass_base * cosA) +
                  v_BC * (self.mass_base * cosA * cosB - (self.mass_vertex + self.mass_base) * cosC)
                  ) * self.mass_vertex / d

        # (B, C, D)
        df_A = tau_AB * e_AB - tau_CA * e_CA
        dv_A = self.time_step * df_A / 2 / self.mass_vertex
        df_B = tau_BC * e_BC - tau_AB * e_AB
        dv_B = self.time_step * df_B / 2 / self.mass_base
        df_C = tau_CA * e_CA - tau_BC * e_BC
        dv_C = self.time_step * df_C / 2 / self.mass_base

        # (B, A, D)
        return msnp.stack((dv_A, dv_B, dv_C), axis=-2).reshape(vel0_.shape), \
            msnp.stack((df_A, df_B, df_C), axis=-2).reshape(vel0_.shape)

    def construct(self,
                  coordinate: Tensor,
                  velocity: Tensor,
                  force: Tensor,
                  energy: Tensor,
                  kinetics: Tensor,
                  virial: Tensor = None,
                  pbc_box: Tensor = None,
                  step: int = 0,
                  ):
        r"""
        Control the pressure of the simulation system.

        Args:
            coordinate (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            velocity (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            force (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            energy (Tensor): Tensor of shape :math:`(B, 1)`. Data type is float.
            kinetics (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            virial (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            pbc_box (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            step (int): Simulation step. Default: ``0``.

        Returns:
            - **coordinate** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **velocity** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **force** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **energy** (Tensor) - Tensor of shape :math:`(B, 1)`. Data type is float.
            - **kinetics** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.
            - **virial** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.
            - **pbc_box** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.

        Note:
            :math:`B` is the number of walkers in simulation.
            :math:`A` is the number of atoms.
            :math:`D` is the spatial dimension of the simulation system. Usually is 3.
        """
        #pylint: disable=invalid-name
        # (B, A, D)
        crd_old_0_0 = msnp.take_along_axis(self._coordinate.copy(), self.settle_index[..., None], axis=-2)
        crd_new_0_1 = msnp.take_along_axis(coordinate.copy(), self.settle_index[..., None], axis=-2)
        mc = self.get_mass_center(crd_new_0_1)
        # (B, C, 4)
        transform, itransform = self.get_transform(crd_old_0_0)
        # (B, A, D)
        crd_old_1_0 = self.apply_transform(transform, crd_old_0_0)
        crd_new_1_1 = self.apply_transform(transform, crd_new_0_1)
        rotation = self._rotation(crd_old_1_0, crd_new_1_1)
        crd_new_1_3 = self.apply_transform(rotation, self.crd_1)
        crd_new_1_3 = self._swap_h(crd_new_1_1, crd_new_1_3)
        # (B, G, a, D)
        crd_new_0_3 = self.apply_transform(itransform, crd_new_1_3).reshape((crd_old_0_0.shape[0], -1, 3,
                                                                             crd_old_0_0.shape[-1]))
        # (B, G, a, D)
        crd_new_0_3 += mc
        # (B, A, D)
        crd_new_0_3 = crd_new_0_3.reshape(crd_old_0_0.shape)

        update_crd = msnp.take_along_axis(coordinate.copy(), self.settle_index[..., None], axis=-2)
        update_vel = msnp.take_along_axis(velocity.copy(), self.settle_index[..., None], axis=-2)
        update_frc = msnp.take_along_axis(force.copy(), self.settle_index[..., None], axis=-2)
        dv, df = self.get_vel_force_update(update_crd, update_vel)

        coordinate = self.scatter_update(coordinate, self.bs_index, crd_new_0_3, axis=-2)

        velocity = self.scatter_update(velocity, self.bs_index, update_vel+dv, axis=-2).clip(-20., 20.)
        force = self.scatter_update(force, self.bs_index, update_frc+df, axis=-2)

        if self._pbc_box is not None:
            # (B,G,C,D)
            group_crd = crd_new_0_1.reshape((crd_new_0_1.shape[0], -1, 3, crd_new_0_1.shape[-1]))
            group_frc = df.reshape((df.shape[0], -1, 3, df.shape[-1]))

            # (B,G,D)
            df_b = group_frc[:, :, 1, :]
            df_c = group_frc[:, :, 2, :]
            vec_ab = group_crd[:, :, 1, :] - group_crd[:, :, 0, :]
            vec_ac = group_crd[:, :, 2, :] - group_crd[:, :, 0, :]

            # (B,D) <- (B,A,D,D) <- (B,A,D) x (B,A,D)
            virial += -0.5 * (df_b * vec_ab + df_c * vec_ac).sum(-2)


        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
