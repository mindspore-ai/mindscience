# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
from numpy import exp

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore import jit
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F

from ...colvar import AtomDistances
from .energy import NonbondEnergy
from ...function import functions as func
from ...function.functions import gather_values
from ...function.units import Units


@jit
def coulomb_interaction(qi: Tensor, qj: Tensor, inv_dis: Tensor, mask: Tensor = None):
    """calculate Coulomb interaction using Coulomb's law."""

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
    energy = func.keepdim_sum(energy, 1) * 0.5

    return energy


class CoulombEnergy(NonbondEnergy):
    r"""
    Coulomb interaction.

    .. Math::

        E_{ele}(r_{ij}) = \sum_{ij} k_{coulomb} \times q_i \times q_j / r_{ij}

    Args:
        atom_charge (Tensor):       Tensor of shape (B, A). Data type is float.
                                    Atom charge. Default: None.
        parameters (dict):          Force field parameters. Default: None.
        cutoff (float):             Cutoff distance. Default: None.
        use_pbc (bool, optional):   Whether to use periodic boundary condition. Default: None.
        use_pme (bool, optional):   Whether to use particle mesh ewald condition. Default: None.
        alpha (float):              Alpha for DSF and PME coulomb interaction.
                                    Default: 0.25.
        nfft (Tensor):              Parameter of FFT, required by PME. Default: None.
        exclude_index (Tensor):     Tensor of the exclude index, required by PME. Default: None.
        length_unit (str):          Length unit for position coordinates. Default: 'nm'.
        energy_unit (str):          Energy unit. Default: 'kj/mol'.
        units (Units):              Units of length and energy. Default: None.

    Returns:
        energy (Tensor), Tensor of shape (B, 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 atom_charge: Tensor = None,
                 parameters: dict = None,
                 cutoff: float = None,
                 use_pbc: bool = None,
                 use_pme: bool = False,
                 alpha: float = 0.25,
                 nfft: Tensor = None,
                 exclude_index: Tensor = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='coulomb_energy',
            output_dim=1,
            cutoff=cutoff,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        if parameters is not None:
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)

        self.atom_charge = self.identity(atom_charge)
        self.coulomb_const = Tensor(self.units.coulomb, ms.float32)

        if use_pme is None:
            use_pme = use_pbc
        self.use_pme = use_pme
        if self.use_pme and (not self.use_pbc):
            raise ValueError('PME cannot be used without periodic box conditions')

        self.pme_coulomb = None
        self.dsf_coulomb = None
        if self.use_pme:
            self.pme_coulomb = ParticleMeshEwaldCoulomb(self.cutoff, alpha, nfft, exclude_index, self.units)
        else:
            self.dsf_coulomb = DampedShiftedForceCoulomb(self.cutoff, alpha)

    def set_cutoff(self, cutoff: Tensor):
        """
        Set cutoff distance.

        Args:
            cutoff (Tensor):         Cutoff distance. Default: None.
        """
        if cutoff is None:
            if self.use_pbc:
                raise ValueError('cutoff cannot be none when using periodic boundary condition')
            self.cutoff = None
        else:
            self.cutoff = Tensor(cutoff, ms.float32)
            if self.dsf_coulomb is not None:
                self.dsf_coulomb.set_cutoff(cutoff)
            if self.pme_coulomb is not None:
                self.pme_coulomb.set_cutoff(cutoff)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None,
                  ):
        r"""
        Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.
        """

        inv_neigh_dis *= self.inverse_input_scale

        # (B,A,1)
        qi = F.expand_dims(self.atom_charge, -1)
        # (B,A,N)
        qj = gather_values(self.atom_charge, neighbour_index)

        if self.cutoff is None:
            energy = coulomb_interaction(qi, qj, inv_neigh_dis, neighbour_mask)
        else:
            neighbour_distance *= self.input_unit_scale
            if self.use_pme:
                energy = self.pme_coulomb(coordinate,
                                          qi, qj, neighbour_distance,
                                          inv_neigh_dis, neighbour_mask,
                                          pbc_box)
            else:
                energy = self.dsf_coulomb(
                    qi, qj, neighbour_distance, inv_neigh_dis, neighbour_mask)

        return energy * self.coulomb_const


class DampedShiftedForceCoulomb(Cell):
    r"""Damped shifted force coulomb potential.

    Args:

        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge.

        cutoff (float):         Cutoff distance. Default: None

        alpha (float):          Alpha. Default: 0.25

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Length unit for position coordinates. Default: None

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    """

    def __init__(self,
                 cutoff: float = None,
                 alpha: float = 0.25,
                 ):

        super().__init__()

        self.alpha = Parameter(Tensor(alpha, ms.float32), name='alpha', requires_grad=False)

        self.erfc = ops.Erfc()
        self.f_shift = None
        self.e_shift = None
        if cutoff is not None:
            self.set_cutoff(cutoff)

    def set_cutoff(self, cutoff: Tensor):
        """set cutoff distance"""
        self.cutoff = Tensor(cutoff, ms.float32)
        cutoff2 = F.square(self.cutoff)
        erfcc = self.erfc(self.alpha * self.cutoff)
        erfcd = msnp.exp(-F.square(self.alpha) * cutoff2)

        self.f_shift = -(erfcc / cutoff2 + 2 / msnp.sqrt(msnp.pi)
                         * self.alpha * erfcd / self.cutoff)
        self.e_shift = erfcc / self.cutoff - self.f_shift * self.cutoff

    def construct(self,
                  qi: Tensor,
                  qj: Tensor,
                  dis: Tensor,
                  inv_dis: Tensor,
                  mask: Tensor = None,
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """

        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi*qj
        energy = qiqj * inv_dis * (self.erfc(self.alpha * dis) -
                                   dis * self.e_shift - F.square(dis) * self.f_shift)

        if mask is None:
            mask = dis < self.cutoff
        else:
            mask = F.logical_and(mask, dis < self.cutoff)

        energy = msnp.where(mask, energy, 0.0)

        # (B,A)
        energy = F.reduce_sum(energy, -1)
        # (B,1)
        energy = func.keepdim_sum(energy, 1) * 0.5

        return energy

#pylint: disable=unused-argument
class RFFT3D(Cell):
    r"""rfft3d"""
    def __init__(self, fftx, ffty, fftz, fftc, inverse):
        Cell.__init__(self)
        self.cast = ms.ops.Cast()
        self.rfft3d = ms.ops.FFT3D()
        self.irfft3d = ms.ops.IFFT3D()
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
        if self.inverse:
            ans = self.rfft3d(dout)
        else:
            ans = self.irfft3d(dout)
        return (ans,)


class ParticleMeshEwaldCoulomb(Cell):
    r"""Particle mesh ewald algorithm for electronic interaction

    Args:

        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge.

        cutoff (float):         Cutoff distance. Default: None

        alpha (float):          the parameter of the Gaussian charge. Default: 0.275106

       nfft (Tensor):         Tensor of FFT parameter. Default: None

        exclude_index (Tensor):   Tensor of the exclude index. Default: None

        units (Units):              Units of length and energy. Default: None
    """

    def __init__(self,
                 cutoff: float = None,
                 alpha: float = 0.275106,
                 nfft: Tensor = None,
                 exclude_index: Tensor = None,
                 units: Units = None):

        super().__init__()

        self.units = units
        self.cutoff = cutoff
        self.alpha = Tensor(0.275106, ms.float32)
        self.erfc = ops.Erfc()
        self.input_unit_scale = 1
        self.exclude_index = None
        self.exclude_pairs = None
        self.get_exclude_distance = None
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
        self.set_nfft(nfft)
        self.double_gradient = Double_Gradient()
        #self.set_nfft([[4,4,4]])
        print(self.nfft, self.alpha)
        self.cast = ms.ops.Cast()

        ma = [1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0]
        ma = Tensor([[ma[i], ma[j], ma[k]]for i in range(4) for j in range(4) for k in range(4)], ms.float32)
        self.ma = ma.reshape(1, 1, 64, 3)
        mb = [0, 0.5, -1, 0.5]
        mb = Tensor([[mb[i], mb[j], mb[k]]for i in range(4) for j in range(4) for k in range(4)], ms.float32)
        self.mb = mb.reshape(1, 1, 64, 3)
        mc = [0, 0.5, 0, -0.5]
        mc = Tensor([[mc[i], mc[j], mc[k]]for i in range(4) for j in range(4) for k in range(4)], ms.float32)
        self.mc = mc.reshape(1, 1, 64, 3)
        md = [0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]
        md = Tensor([[md[i], md[j], md[k]]for i in range(4) for j in range(4) for k in range(4)], ms.float32)
        self.md = md.reshape(1, 1, 64, 3)
        self.base_grid = Tensor([[i, j, k] for i in range(4) for j in range(4) for k in range(4)],
                                ms.int32).reshape(1, 1, 64, 3)
        self.batch_constant = msnp.ones((exclude_index.shape[0], exclude_index.shape[1], 64, 1), ms.int32)
        self.batch_constant *= msnp.arange(0, exclude_index.shape[0]).reshape(-1, 1, 1, 1)
        self.set_exclude_index(exclude_index)
        if units:
            self.set_input_unit(units)
        if alpha:
            self.set_alpha(alpha)

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
        res = exp(tempc)
        for kk in range(order - 1):
            tempc = complex(0, 2 * msnp.pi * k / fftn * kk)
            tempc = exp(tempc)
            tempf = ParticleMeshEwaldCoulomb._m(kk + 1, order)
            tempc2 += tempf * tempc
        res = res / tempc2
        return abs(res) * abs(res)

    def set_input_unit(self, units: Units):
        """set the length unit for the input coordinates"""
        if units is None:
            self.input_unit_scale = 1
        elif isinstance(units, Units):
            self.input_unit_scale = Tensor(
                self.units.convert_length_from(units), ms.float32)
        else:
            raise TypeError('Unsupported type: '+str(type(units)))
        return self

    def set_cutoff(self, cutoff: Tensor):
        """set cutoff distance"""
        self.cutoff = Tensor(cutoff, ms.float32)

    def set_alpha(self, alpha: Tensor):
        """set the parameter beta"""
        self.alpha = Tensor(alpha, ms.float32)

    def set_exclude_index(self, exclude_index: Tensor):
        """set exclude index"""
        if exclude_index is None:
            self.exclude_index = None
        else:
            if exclude_index.ndim != 3:
                raise ValueError('The rank of exclude index must be 3.')
            if exclude_index.shape[2] == 0:
                self.exclude_index = None
            else:
                self.exclude_index = Tensor(exclude_index, ms.int32)
        if self.exclude_index is not None:
            t = []
            for batch in self.exclude_index:
                t.append([])
                for i, ex in enumerate(batch):
                    for ex_atom in ex:
                        if i < ex_atom < self.exclude_index.shape[1]:
                            t[-1].append([i, ex_atom])
            self.exclude_pairs = msnp.array(t)
            self.get_exclude_distance = AtomDistances(self.exclude_pairs, use_pbc=True, length_unit=self.units)

    def set_nfft(self, nfft: Tensor):
        """set nfft"""
        self.nfft = Tensor(nfft, ms.int32).reshape((-1, 1, 3))
        self.fftx = int(self.nfft[0][0][0])
        self.ffty = int(self.nfft[0][0][1])
        self.fftz = int(self.nfft[0][0][2])
        if self.fftx % 4 != 0 or self.ffty % 4 != 0 or self.fftz % 4 != 0:
            raise ValueError("The FFT grid number for PME must be a multiple of 4")
        self.fftc = self.fftz // 2 + 1
        self.ffkx = msnp.arange(self.fftx)
        self.ffkx = msnp.where(self.ffkx > self.fftx / 2, self.fftx - self.ffkx, self.ffkx).reshape(-1, 1, 1)
        self.ffky = msnp.arange(self.ffty)
        self.ffky = msnp.where(self.ffky > self.ffty / 2, self.ffty - self.ffky, self.ffky).reshape(1, -1, 1)
        self.ffkz = msnp.arange(self.fftc).reshape(1, 1, -1)

        bx = msnp.array([self._b(i, self.fftx) for i in range(self.fftx)])
        by = msnp.array([self._b(i, self.ffty) for i in range(self.ffty)])
        bz = msnp.array([self._b(i, self.fftz) for i in range(self.fftc)])

        self.b = bx.reshape(-1, 1, 1) * by.reshape(1, -1, 1) * bz.reshape(1, 1, -1)
        self.rfft3d = RFFT3D(self.fftx, self.ffty, self.fftz, self.fftc, inverse=False)
        self.irfft3d = RFFT3D(self.fftx, self.ffty, self.fftz, self.fftc, inverse=True)

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

        energy = msnp.where(mask, energy, 0.0)

        # (B,A)
        energy = F.reduce_sum(energy, -1)
        # (B,1)
        energy = func.keepdim_sum(energy, 1) * 0.5

        return energy

    def calculate_self_energy(self, qi: Tensor, pbc_box: Tensor):
        """Calculate the direct energy term."""
        # (B,A,1) = (B,A,1) * (B,A,1)
        qiqi = qi * qi

        # (B,1)
        qiqi_sum = F.reduce_sum(qiqi, 1)
        qi_sum = F.reduce_sum(qi, 1)

        energy = -self.alpha / msnp.sqrt(msnp.pi) * qiqi_sum
        energy -= qi_sum * 0.5 * msnp.pi / (self.alpha * self.alpha * F.reduce_prod(pbc_box, 1))
        return energy

    def calculate_exclude_energy(self, coordinate: Tensor, qi: Tensor, pbc_box: Tensor):
        """Calculate the excluded correction energy term."""
        if self.exclude_index is not None:
            # (B,b)
            dis = self.get_exclude_distance(coordinate, pbc_box) * self.input_unit_scale
            # (B,A) <- (B,A,1)：
            qi = F.reshape(qi, (qi.shape[0], -1))
            # (B,b,2) <- (B,A)：
            qi = gather_values(qi, self.exclude_pairs)
            # (B,b) <- (B,b,2)：
            qiqj = F.reduce_prod(qi, -1)
            energy = -qiqj * F.erf(self.alpha * dis) / dis
            energy = func.keepdim_sum(energy, -1)
            return energy
        return msnp.zeros((qi.shape[0], 1), ms.float32)

    def calculate_reciprocal_energy(self, coordinate: Tensor, qi: Tensor, pbc_box: Tensor):
        """Calculate the reciprocal energy term."""
        # the batch dimension in the following part is ignored due to the limitation of the operator FFT3D
        # (B,A,3) <- (B,A,3) / (B,1,3) * (B,1,3):
        pbc_box = pbc_box.reshape((-1, 1, 3))
        frac = coordinate / F.stop_gradient(pbc_box) % 1.0 * self.nfft
        grid = self.cast(frac, ms.int32)
        frac = frac - F.floor(frac)
        # (B,A,64,3) <- (B,A,1,3) + (1,1,64,3):
        neibor_grids = F.expand_dims(grid, 2) - self.base_grid
        neibor_grids %= F.expand_dims(self.nfft, 2)
        # (B,A,64,3) <- (B,A,1,3) * (1,1,64,3)
        frac = F.expand_dims(frac, 2)
        neibor_q = frac * frac * frac * self.ma + frac * frac * self.mb + frac * self.mc + self.md
        # (B,A,64) <- (B,A,1) * reduce (B,A,64,3)
        neibor_q = qi * F.reduce_prod(neibor_q, -1)
        # (B,A,64,4) <- concat (B,A,64,1) (B,A,64,3)：
        neibor_grids = F.concat((self.batch_constant, neibor_grids), -1)
        # (B, fftx, ffty, fftz)：
        q_matrix = msnp.zeros([1, self.fftx, self.ffty, self.fftz], ms.float32)
        q_matrix = F.tensor_scatter_add(q_matrix, neibor_grids.reshape(-1, 4), neibor_q.reshape(-1))

        mprefactor = msnp.pi * msnp.pi / -self.alpha / self.alpha
        # (fftx, ffty, fftc) = (fftx, 1, 1) + (1, ffty, 1) + (1, 1, fftc)
        msq = self.ffkx * self.ffkx / pbc_box[0][0][0] / pbc_box[0][0][0] + \
            self.ffky * self.ffky / pbc_box[0][0][1] / pbc_box[0][0][1] + \
            self.ffkz * self.ffkz / pbc_box[0][0][2] / pbc_box[0][0][2]
        msq[0][0][0] = 1
        bc = 1.0 / msnp.pi / msq * msnp.exp(mprefactor * msq) / F.reduce_prod(pbc_box, -1)[0]
        bc[0][0][0] = 0
        bc *= self.b
        fq = self.rfft3d(q_matrix.reshape(self.fftx, self.ffty, self.fftz))
        bcfq = bc * fq
        fbcfq = self.irfft3d(bcfq)
        fbcfq = F.expand_dims(fbcfq, 0)
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

        direct_energy = self.calculate_direct_energy(qi, qj, dis, inv_dis, mask)
        self_energy = self.calculate_self_energy(qi, pbc_box)
        exclude_energy = self.calculate_exclude_energy(coordinate, qi, pbc_box)
        reciprocal_energy = self.calculate_reciprocal_energy(coordinate, qi, pbc_box)
        return direct_energy + self_energy + exclude_energy + reciprocal_energy
