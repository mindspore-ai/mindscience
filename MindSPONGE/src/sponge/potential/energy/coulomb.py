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
"""Electroinc interaction"""

from typing import Union, List
import numpy as np
from numpy import ndarray
from scipy.special import erfcinv

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F

from ...colvar import Distance
from .energy import NonbondEnergy, _energy_register
from ...function import functions as func
from ...function import gather_value, get_ms_array, get_arguments
from ...function.units import Units, GLOBAL_UNITS, Length
from ...system.molecule import Molecule


@_energy_register('coulomb_energy')
class CoulombEnergy(NonbondEnergy):
    r"""Coulomb interaction

    Math:

    .. math::
        E_{ele}(r_{ij}) = \sum_{ij} k_{coulomb} \frac{q_i q_j}{r_{ij}}

    Args:
        atom_charge (Union[Tensor, ndarray, List[float]]):
            Array of atomic charge. The shape of array is `(B, A)`, and the data type is float.

        cutoff (Union[float, Length, Tensor]):  Cut-off distance. Default: ``None``.

        pbc_box (Union[Tensor, ndarray, List[float]]):
            Array of PBC box with shape `(B, A, D)`, and the data type is float. Default: ``None``.

        exclude_index (Union[Tensor, ndarray, List[int]]):
            Tensor of the exclude index, required by PME. Default: ``None``.

        damp_dis (Union[float, Length, Tensor]):
            A distance :math:`l_{\alpha}` to calculate the damping factor :math:`\alpha = l_{\alpha}^-1`
            for damped shifted force (DSF) method. Default: Length(0.48, 'nm')

        pme_accuracy (float): Accuracy for particle mesh ewald (PME) method.

        use_pme (bool): Whether to use particle mesh ewald (PME) method to calculate the coulomb interaction
            of the system in PBC box. If `False` is given, the damped shifted force (DSF) method
            will be used for PBC. Default: ``True``.

        parameters (dict): Force field parameters. Default: ``None``.

        length_unit (str): Length unit. If None is given, it will be assigned with the global length unit.
            Default: 'nm'

        energy_unit (str): Energy unit. If None is given, it will be assigned with the global energy unit.
            Default: 'kj/mol'

        name (str): Name of the energy. Default: 'coulomb'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 system: Molecule = None,
                 atom_charge: Union[Tensor, ndarray, List[float]] = None,
                 cutoff: Union[float, Length, Tensor] = None,
                 pbc_box: Union[Tensor, ndarray, List[float]] = None,
                 exclude_index: Union[Tensor, ndarray, List[int]] = None,
                 damp_dis: Union[float, Length, Tensor] = Length(0.48, 'nm'),
                 pme_accuracy: float = 1e-4,
                 use_pme: bool = True,
                 parameters: dict = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 name: str = 'coulomb_energy',
                 **kwargs,
                 ):

        super().__init__(
            name=name,
            cutoff=cutoff,
            use_pbc=(pbc_box is not None),
            length_unit=length_unit,
            energy_unit=energy_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if parameters is not None:
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)

        if system is None:
            self.input_unit = self.units.length_unit
        else:
            self.input_unit = system.units.length_unit
            if atom_charge is None:
                atom_charge = system.atom_charge
            if pbc_box is None and system.pbc_box is not None:
                self._use_pbc = True
                pbc_box = system.pbc_box * self.units.convert_length_from(system.units)
                if cutoff is None:
                    cutoff = self.units.length(1, 'nm')

        if isinstance(cutoff, Length):
            cutoff = cutoff(self.units)
        self.cutoff = get_ms_array(cutoff, ms.float32)

        self.atom_charge = atom_charge
        self.coulomb_const = Tensor(self.units.coulomb, ms.float32)

        self.damp_dis = damp_dis

        self._dsf_warrning = '[WARNING] Using `cutoff` without periodic boundary condition (PBC) will call ' \
                             'the damped shifted force algorithm to estimate the Coulomb interaction, ' \
                             'which is only recommended for minimization. It is recommended to set ' \
                             'the `cutoff` to `None` to perform the MD simulation without PBC.'

        self.pme_coulomb = None
        self.dsf_coulomb = None
        self.use_pme = use_pme
        if self.cutoff is not None:
            if self._use_pbc and self.use_pme:
                self.pme_coulomb = ParticleMeshEwaldCoulomb(cutoff=self.cutoff,
                                                            pbc_box=pbc_box,
                                                            exclude_index=exclude_index,
                                                            accuracy=pme_accuracy,
                                                            length_unit=self.units.length_unit)
            else:
                if not self.use_pbc:
                    print(self._dsf_warrning)
                self.dsf_coulomb = DampedShiftedForceCoulomb(self.cutoff, damp_dis=self.damp_dis,
                                                             length_unit=self.units.length_unit)
    @staticmethod
    def check_system(system: Molecule) -> bool:
        """Check if the system needs to calculate this energy term"""
        return system.atom_charge is not None

    @staticmethod
    def coulomb_interaction(qi: Tensor, qj: Tensor, inv_dis: Tensor, mask: Tensor = None) -> Tensor:
        """calculate Coulomb interaction using Coulomb's law"""

        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi * qj

        # (B,A,N)
        energy = qiqj * inv_dis

        if mask is not None:
            # (B,A,N) * (B,A,N)
            energy *= mask

        # (B,A)
        energy = F.reduce_sum(energy, -1)
        # (B,1)
        energy = func.keepdims_sum(energy, 1) * 0.5

        return energy

    def set_cutoff(self, cutoff: Union[float, Length, Tensor], unit: str = None):
        """set cutoff distance"""
        super().set_cutoff(cutoff, unit)
        if self.cutoff is not None:
            if self._use_pbc and self.use_pme:
                self.pme_coulomb.set_cutoff(self.cutoff)
            else:
                if self.dsf_coulomb is None:
                    if not self.use_pbc:
                        print(self._dsf_warrning)
                    self.dsf_coulomb = DampedShiftedForceCoulomb(self.cutoff,
                                                                 damp_dis=self.damp_dis,
                                                                 length_unit=self.units.length_unit)
                else:
                    self.dsf_coulomb.set_cutoff(self.cutoff)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Vectors from central atom to neighbouring atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        neighbour_distance *= self.input_unit_scale
        inv_neigh_dis = msnp.reciprocal(neighbour_distance)
        if neighbour_mask is not None:
            inv_neigh_dis = msnp.where(neighbour_mask, inv_neigh_dis, 0)

        atom_charge = self.identity(self.atom_charge)
        # (B,A,1)
        qi = F.expand_dims(atom_charge, -1)
        # (B,A,N)
        qj = gather_value(atom_charge, neighbour_index)

        if self.cutoff is None:
            energy = self.coulomb_interaction(qi, qj, inv_neigh_dis, neighbour_mask)
        else:
            if pbc_box is not None and self.use_pme:
                coordinate *= self.input_unit_scale
                pbc_box *= self.input_unit_scale
                energy = self.pme_coulomb(coordinate,
                                          qi, qj, neighbour_distance,
                                          inv_neigh_dis, neighbour_mask,
                                          pbc_box)
            else:
                energy = self.dsf_coulomb(
                    qi, qj, neighbour_distance, inv_neigh_dis, neighbour_mask)

        return energy * self.coulomb_const


class DampedShiftedForceCoulomb(Cell):
    r"""Damped shifted force coulomb potential

    Reference:

        Mei, H.; Liu, Q.; Liu, L.; Lai, X.; Li, J.
        An improved charge transfer ionic-embedded atom method potential for aluminum/alumina
        interface system based on damped shifted force method [J].
        Computational Materials Science, 2016, 115: 60-71.

    Args:
        cutoff (Union[float, Length, Tensor]): Cutoff distance.

        damp_dis (Union[float, Length, Tensor]):
            A distance :math:`l_{\alpha}` to calculate the damping factor :math:`\alpha = l_{\alpha}^-1`.
            Default: Length(0.48, 'nm')

        length_unit (str): Length unit. If None is given, it will be assigned with the global length unit.
            Default: 'nm'
    """

    def __init__(self,
                 cutoff: Union[float, Length, Tensor] = Length(1, 'nm'),
                 damp_dis: Union[float, Length, Tensor] = Length(0.48, 'nm'),
                 length_unit: str = 'nm',
                 ):

        super().__init__()

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        if isinstance(cutoff, Length):
            cutoff = cutoff(self.units)
        self.cutoff = get_ms_array(cutoff, ms.float32)
        self.inv_cutoff = msnp.reciprocal(self.cutoff)

        if isinstance(damp_dis, Length):
            damp_dis = damp_dis(self.units)

        self.alpha = msnp.reciprocal(damp_dis, ms.float32)

        self.erfc = ops.Erfc()
        self.dsf_scale = None
        self.dsf_shift = None
        self.dsf_self = None

        self._build_dsf()

    def set_alpha(self, alpha: float):
        """set damping parameter `\alpha`"""
        self.alpha = get_ms_array(alpha, ms.float32)
        self._build_dsf()
        return self

    def set_cutoff(self, cutoff: Tensor):
        """set cutoff distance"""
        self.cutoff = get_ms_array(cutoff, ms.float32)
        self.inv_cutoff = msnp.reciprocal(self.cutoff)
        self._build_dsf()
        return self

    def construct(self,
                  qi: Tensor,
                  qj: Tensor,
                  dis: Tensor,
                  inv_dis: Tensor,
                  mask: Tensor = None,
                  ):
        r"""Calculate energy term.

        """

        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi*qj

        e_ij = qiqj * (self.erfc(self.alpha * dis) * inv_dis - self.dsf_shift +
                       self.dsf_scale * (dis * self.inv_cutoff - 1))

        if mask is None:
            mask = dis < self.cutoff
        else:
            mask = F.logical_and(mask, dis < self.cutoff)

        e_ij = F.select(mask, e_ij, F.zeros_like(e_ij))

        # (B,A)
        e_ij = F.reduce_sum(e_ij, -1)
        # (B,1)
        e_ij = func.keepdims_sum(e_ij, 1)

        # (B, 1) <- (B, A, 1)
        sum_qi2 = F.reduce_sum(F.square(qi), -2)
        # (B, 1)
        e_ii = self.dsf_self * sum_qi2

        energy = 0.5 * (e_ii + e_ij)

        return energy

    def _build_dsf(self):
        """build shifted constant"""
        ac = self.alpha * self.cutoff
        # erfc(\alpha R_c)
        erfc_ac = self.erfc(ac)
        # e^{-\alpha^2 R_c^2}
        exp_ac2 = msnp.exp(-F.square(ac))
        # \frac{\alpha}{\sqrt{\pi}}
        a_pi = self.alpha / msnp.sqrt(msnp.pi)

        # \frac{erfc(\alpha R_c)}{R_c}
        self.dsf_shift = erfc_ac * self.inv_cutoff
        # \frac{erfc(\alpha R_c)}{R_c} + \frac{2\alpha}{\sqrt{\pi}} e^{-\alpha^2 R_c^2}
        self.dsf_scale = self.dsf_shift + 2 * a_pi * exp_ac2
        # \frac{erfc(\alpha R_c)}{R_c} + \frac{\alpha}{\sqrt{\pi}} e^{-\alpha^2 R_c^2} +
        #   \frac{\alpha}{\sqrt{\pi}}
        self.dsf_self = self.dsf_shift + a_pi * (exp_ac2 + 1)


class RFFT3D(Cell):
    r"""rfft3d"""

    def __init__(self, fftx, ffty, fftz, fftc, inverse):
        Cell.__init__(self)
        self.cast = ops.Cast()
        if ms.get_context("device_target") == "Ascend":
            self.rfft3d = ops.FFTWithSize()
            self.irfft3d = ops.FFTWithSize()
        else:
            from ...customops import FFTOP
            fftop = FFTOP()
            self.rfft3d, self.irfft3d = fftop.register()
        self.inverse = inverse
        if self.inverse:
            self.norm = msnp.ones(fftc, dtype=ms.float32) * fftx * ffty * fftz
            self.norm = 1 / self.norm
            self.norm[1:-1] *= 2
        else:
            self.norm = msnp.ones(fftc, dtype=ms.float32) * fftx * ffty * fftz
            self.norm[1:-1] /= 2

    def construct(self, x):
        if self.inverse:
            return self.irfft3d(x)
        return self.rfft3d(x)

    def bprop(self, x, out, dout):
        #pylint: disable=unused-argument
        if self.inverse:
            ans = self.rfft3d(dout)
        else:
            ans = self.irfft3d(dout)
        return (ans,)


class ParticleMeshEwaldCoulomb(Cell):
    r"""Particle mesh ewald algorithm for electronic interaction

    Reference:

        Essmann, U.; Perera, L.; Berkowitz, M. L.; Darden, T.; Lee, H.; Pedersen, L. G.
        A Smooth Particle Mesh Ewald Method [J].
        The Journal of Chemical Physics, 1995, 103(19): 8577-8593.

    Args:
        pbc_box (Union[Tensor, ndarray, List[float]]):
            Array of PBC box with shape `(B, A, D)`, and the data type is float. Default: ``None``.

        cutoff (Union[float, Length, Tensor]): Cutoff distance. Default: Length(1, 'nm')

        exclude_index (Union[Tensor, ndarray, List[int]]):
            Array of indexes of atoms that should be excluded from neighbour list.
            The shape of the tensor is `(B, A, Ex)`. The data type is int. Default: ``None``.

        accuracy (float): Accuracy for PME. Default: 1e-4

        length_unit (str): Length unit. If None is given, it will be assigned with the global length unit.
            Default: 'nm'

    """

    def __init__(self,
                 pbc_box: Union[Tensor, ndarray, List[float]],
                 cutoff: Union[float, Length, Tensor] = Length(1, 'nm'),
                 exclude_index: Tensor = None,
                 accuracy: float = 1e-4,
                 length_unit: str = 'nm',
                 ):

        super().__init__()

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        if isinstance(cutoff, Length):
            cutoff = cutoff(self.units)
        self.cutoff = get_ms_array(cutoff, ms.float32)

        self.accuracy = accuracy
        self.alpha = erfcinv(self.accuracy) / self.cutoff

        self.nfft = None
        self.fftx = None
        self.ffty = None
        self.fftz = None
        self.fftc = None
        self.fftx = None
        self.ffty = None
        self.fftz = None
        self.b = None
        self.rfft3d = None
        self.irfft3d = None
        self.cast = ops.Cast()
        self.erfc = ops.Erfc()

        pbc_box = get_ms_array(pbc_box, ms.float32)
        nfft = pbc_box[0] * 10 // 4 * 4
        self.set_nfft(nfft)

        self.exclude_pairs = None
        self.get_exclude_distance = None
        self.set_exclude_index(exclude_index)

        ma = [1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0]
        ma = Tensor([[ma[i], ma[j], ma[k]]for i in range(4)
                     for j in range(4) for k in range(4)], ms.float32)
        self.ma = ma.reshape(1, 1, 64, 3)
        mb = [0, 0.5, -1, 0.5]
        mb = Tensor([[mb[i], mb[j], mb[k]]for i in range(4)
                     for j in range(4) for k in range(4)], ms.float32)
        self.mb = mb.reshape(1, 1, 64, 3)
        mc = [0, 0.5, 0, -0.5]
        mc = Tensor([[mc[i], mc[j], mc[k]]for i in range(4)
                     for j in range(4) for k in range(4)], ms.float32)
        self.mc = mc.reshape(1, 1, 64, 3)
        md = [0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]
        md = Tensor([[md[i], md[j], md[k]]for i in range(4)
                     for j in range(4) for k in range(4)], ms.float32)
        self.md = md.reshape(1, 1, 64, 3)
        self.base_grid = Tensor([[i, j, k] for i in range(4) for j in range(4) for k in range(4)],
                                ms.int32).reshape(1, 1, 64, 3)

        self.batch_constant = msnp.ones((self.exclude_index.shape[0], self.exclude_index.shape[1], 64, 1), ms.int32)
        self.batch_constant *= msnp.arange(0, self.exclude_index.shape[0]).reshape(-1, 1, 1, 1)

        self.reduce_prod = ops.ReduceProd()

    @staticmethod
    def _m(u, n):
        """get factor m"""
        if n == 2:
            if u > 2 or u < 0:
                return 0
            return 1 - abs(u - 1)
        self = ParticleMeshEwaldCoulomb._m
        return u / (n - 1) * self(u, n - 1) + (n - u) / (n - 1) * self(u - 1, n - 1)

    @staticmethod
    def _b(k, fftn, order=4):
        """get factor b"""
        tempc2 = complex(0, 0)
        tempc = complex(0, 2 * (order - 1) * msnp.pi * k / fftn)
        res = np.exp(tempc)
        for kk in range(order - 1):
            tempc = complex(0, 2 * msnp.pi * k / fftn * kk)
            tempc = np.exp(tempc)
            tempf = ParticleMeshEwaldCoulomb._m(kk + 1, order)
            tempc2 += tempf * tempc
        res = res / tempc2
        return abs(res) * abs(res)

    def set_cutoff(self, cutoff: Tensor):
        """set cutoff distance"""
        self.cutoff = get_ms_array(cutoff, ms.float32)
        self.alpha = erfcinv(self.accuracy) / self.cutoff
        return self

    def set_alpha(self, alpha: Tensor):
        """set the parameter beta"""
        self.alpha = get_ms_array(alpha, ms.float32)
        return self

    def set_exclude_index(self, exclude_index: Union[Tensor, ndarray, List[int]] = None):
        """set exclude index"""
        if exclude_index is None:
            self.exclude_index = None
            return self

        self.exclude_index = get_ms_array(exclude_index, ms.int32)
        if exclude_index.ndim != 3:
            raise ValueError('The rank of exclude index must be 3.')
        if exclude_index.shape[2] == 0:
            self.exclude_index = None
            return self

        t = []
        for batch in self.exclude_index:
            t.append([])
            for i, ex in enumerate(batch):
                for ex_atom in ex:
                    if i < ex_atom < self.exclude_index.shape[1]:
                        t[-1].append([i, ex_atom])
        self.exclude_pairs = msnp.array(t)
        self.get_exclude_distance = Distance(atoms=self.exclude_pairs, use_pbc=True, batched=True)

        return self

    def set_nfft(self, nfft: Tensor):
        """set nfft"""
        self.nfft = get_ms_array(nfft, ms.int32).reshape((-1, 1, 3))
        self.fftx = int(self.nfft[0][0][0])
        self.ffty = int(self.nfft[0][0][1])
        self.fftz = int(self.nfft[0][0][2])
        if self.fftx % 4 != 0 or self.ffty % 4 != 0 or self.fftz % 4 != 0:
            raise ValueError(
                "The FFT grid number for PME must be a multiple of 4")
        self.fftc = self.fftz // 2 + 1
        self.ffkx = msnp.arange(self.fftx)
        self.ffkx = msnp.where(self.ffkx > self.fftx / 2,
                               self.fftx - self.ffkx, self.ffkx).reshape(-1, 1, 1)
        self.ffky = msnp.arange(self.ffty)
        self.ffky = msnp.where(self.ffky > self.ffty / 2,
                               self.ffty - self.ffky, self.ffky).reshape(1, -1, 1)
        self.ffkz = msnp.arange(self.fftc).reshape(1, 1, -1)
        bx = msnp.array([self._b(i, self.fftx) for i in range(self.fftx)])
        by = msnp.array([self._b(i, self.ffty) for i in range(self.ffty)])
        bz = msnp.array([self._b(i, self.fftz) for i in range(self.fftc)])
        self.b = bx.reshape(-1, 1, 1) * by.reshape(1, -1, 1) * bz.reshape(1, 1, -1)

        self.multi_batch_fft = None
        if ms.get_context("device_target") == "Ascend":
            # Ascend platform & mindspore version >= 2.0.0
            self.multi_batch_fft = True
            self.rfft3d = ops.FFTWithSize(signal_ndim=3, real=True, inverse=False)
            self.irfft3d = ops.FFTWithSize(signal_ndim=3, real=True, inverse=True, norm="forward")
        else:
            # GPU platform
            self.multi_batch_fft = False
            self.rfft3d = RFFT3D(self.fftx, self.ffty,
                                 self.fftz, self.fftc, inverse=False)
            self.irfft3d = RFFT3D(self.fftx, self.ffty,
                                  self.fftz, self.fftc, inverse=True)

    def calculate_direct_energy(self,
                                qi: Tensor,
                                qj: Tensor,
                                dis: Tensor,
                                inv_dis: Tensor,
                                mask: Tensor = None):
        """Calculate the direct energy term."""
        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi*qj
        energy = qiqj * inv_dis * (self.erfc(self.alpha * dis))

        if mask is None:
            mask = dis < self.cutoff
        else:
            mask = F.logical_and(mask, dis < self.cutoff)

        energy = msnp.where(mask, energy, F.zeros_like(energy))

        # (B,A)
        energy = F.reduce_sum(energy, -1)
        # (B,1)
        energy = func.keepdims_sum(energy, 1) * 0.5

        return energy

    def calculate_self_energy(self, qi: Tensor, pbc_box: Tensor):
        """Calculate the direct energy term."""
        # (B,A,1) = (B,A,1) * (B,A,1)
        qiqi = qi * qi

        # (B,1)
        qiqi_sum = F.reduce_sum(qiqi, 1)
        qi_sum = F.reduce_sum(qi, 1)

        #pylint:disable=invalid-unary-operand-type
        energy = (0 - self.alpha) / msnp.sqrt(msnp.pi) * qiqi_sum
        energy -= qi_sum * 0.5 * msnp.pi / (self.alpha * self.alpha * self.reduce_prod(pbc_box, 1))
        return energy

    def calculate_exclude_energy(self, coordinate: Tensor, qi: Tensor, pbc_box: Tensor):
        """Calculate the excluded correction energy term."""
        if self.exclude_index is not None:
            # (B,b)
            dis = self.get_exclude_distance(coordinate, pbc_box)
            # (B,A) <- (B,A,1)
            qi = F.reshape(qi, (qi.shape[0], -1))
            # (B,b,2) <- (B,A)
            qi = gather_value(qi, self.exclude_pairs)
            # (B,b) <- (B,b,2)
            qiqj = self.reduce_prod(qi, -1)
            energy = -qiqj * F.erf(self.alpha * dis) / dis
            energy = func.keepdims_sum(energy, -1)
            return energy
        return msnp.zeros((qi.shape[0], 1), ms.float32)

    def calculate_reciprocal_energy(self, coordinate: Tensor, qi: Tensor, pbc_box: Tensor):
        """Calculate the reciprocal energy term."""

        # (B,A,3) <- (B,A,3) / (B,1,3) * (B,1,3)
        pbc_box = pbc_box.reshape((-1, 1, 3))
        frac = coordinate / F.stop_gradient(pbc_box) % 1.0 * self.nfft
        grid = self.cast(frac, ms.int32)
        frac = frac - F.floor(frac)

        # (B,A,64,3) <- (B,A,1,3) + (1,1,64,3)
        neibor_grids = F.expand_dims(grid, 2) - self.base_grid
        neibor_grids %= F.expand_dims(self.nfft, 2)

        # (B,A,64,3) <- (B,A,1,3) * (1,1,64,3)
        frac = F.expand_dims(frac, 2)
        neibor_q = frac * frac * frac * self.ma + frac * \
            frac * self.mb + frac * self.mc + self.md

        # (B,A,64) <- (B,A,1) * reduce (B,A,64,3)
        neibor_q = qi * self.reduce_prod(neibor_q, -1)

        # (B,A,64,4) <- concat (B,A,64,1) (B,A,64,3)
        neibor_grids = F.concat((self.batch_constant, neibor_grids), -1)

        # (B, fftx, ffty, fftz)
        q_matrix = msnp.zeros([1, self.fftx, self.ffty, self.fftz], ms.float32)
        q_matrix = F.tensor_scatter_add(
            q_matrix, neibor_grids.reshape(-1, 4), neibor_q.reshape(-1))

        # pylint:disable=invalid-unary-operand-type
        mprefactor = msnp.pi * msnp.pi / -self.alpha / self.alpha

        # (fftx, ffty, fftc) = (fftx, 1, 1) + (1, ffty, 1) + (1, 1, fftc)
        msq = self.ffkx * self.ffkx / pbc_box[0][0][0] / pbc_box[0][0][0] + \
            self.ffky * self.ffky / pbc_box[0][0][1] / pbc_box[0][0][1] + \
            self.ffkz * self.ffkz / pbc_box[0][0][2] / pbc_box[0][0][2]
        msq[0][0][0] = 1
        bc = 1.0 / msnp.pi / msq * \
            msnp.exp(mprefactor * msq) / self.reduce_prod(pbc_box, -1)[0]
        bc[0][0][0] = 0
        bc *= self.b

        # depends on mindspore versions
        if self.multi_batch_fft:
            # if mindspore_version >= 2.0.0:
            # the batch dimension in the following part is supported.
            fq = self.rfft3d(q_matrix)
            bcfq_real = bc * ops.stop_gradient(fq.real())
            bcfq_imag = bc * ops.stop_gradient(fq.imag())
            bcfq = ops.Complex()(bcfq_real, bcfq_imag)
            fbcfq = self.irfft3d(bcfq)
        else:
            # if mindspore_version < 2.0.0:
            # the batch dimension in the following part is ignored due to
            # the limitation of the operator FFT3D.
            fq = self.rfft3d(q_matrix.reshape(self.fftx, self.ffty, self.fftz))
            bcfq = bc * fq
            fbcfq = self.irfft3d(bcfq)
            F.expand_dims(fbcfq, 0)

        energy = q_matrix * fbcfq
        energy = 0.5 * F.reduce_sum(energy, (-1, -2, -3))
        energy = energy.reshape(-1, 1)

        return energy

    def construct(self,
                  coordinate: Tensor,
                  qi: Tensor,
                  qj: Tensor,
                  dis: Tensor,
                  inv_dis: Tensor,
                  mask: Tensor = None,
                  pbc_box: Tensor = None):
        """Calculate energy term."""

        direct_energy = self.calculate_direct_energy(
            qi, qj, dis, inv_dis, mask)
        self_energy = self.calculate_self_energy(qi, pbc_box)
        exclude_energy = self.calculate_exclude_energy(coordinate, qi, pbc_box)
        reciprocal_energy = self.calculate_reciprocal_energy(
            coordinate, qi, pbc_box)

        e = direct_energy + self_energy + exclude_energy + reciprocal_energy
        return e.astype(ms.float32)
