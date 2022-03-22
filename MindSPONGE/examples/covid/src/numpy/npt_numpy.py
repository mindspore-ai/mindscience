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
'''NPT'''

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, CSRTensor
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindsponge import Angle
from mindsponge import Bond
from mindsponge import Dihedral
from mindsponge import LangevinLiujian
from mindsponge import LennardJonesInformation
from mindsponge import MdInformation
from mindsponge import NonBond14
from mindsponge import NeighborList
from mindsponge import ParticleMeshEwald
from mindsponge import RestrainInformation
from mindsponge import SimpleConstarin
from mindsponge import VirtualInformation
from mindsponge import CoordinateMolecularMap
from mindsponge import BDBARO
from mindsponge.md.functions import angle_energy
from mindsponge.md.functions import angle_force_with_atom_energy
from mindsponge.md.functions import bond_energy
from mindsponge.md.functions import crd_to_uint_crd_quarter
from mindsponge.md.functions import dihedral_energy
from mindsponge.md.functions import dihedral_force_with_atom_energy
from mindsponge.md.functions import nb14_cf_energy
from mindsponge.md.functions import nb14_lj_energy
from mindsponge.md.functions import lj_energy
from mindsponge.md.functions import constrain
from mindsponge.md.functions import last_crd_to_dr
from mindsponge.md.functions import md_temperature
from mindsponge.md.functions import md_iteration_leap_frog_liujian
from mindsponge.md.functions import md_iteration_leap_frog_liujian_with_max_vel
from mindsponge.md.functions import pme_excluded_force
from mindsponge.md.functions import pme_reciprocal_force
from mindsponge.md.functions import refresh_boxmap_times
from mindsponge.md.functions import refresh_crd_vel
from mindsponge.md.functions import lj_force_with_virial_energy
from mindsponge.md.functions import pme_energy_update
from mindsponge.md.functions import dihedral_14_ljcf_force_with_atom_energy_with_virial
from mindsponge.md.functions import bond_force_with_atom_energy_with_virial
from mindsponge.md.functions import get_pme_bc, get_csr_excluded_index, broadcast_with_excluded_list,\
                                                                                   get_csr_residual_index


class Controller:
    '''controller'''

    def __init__(self, args_opt):
        self.input_file = args_opt.i
        self.initial_coordinates_file = args_opt.c
        self.amber_parm = args_opt.amber_parm
        self.restrt = args_opt.r
        self.mdcrd = args_opt.x
        self.mdout = args_opt.o
        self.mdbox = args_opt.box

        self.command_set = {}
        self.md_task = None
        self.commands_from_in_file()
        self.punctuation = ","

    def commands_from_in_file(self):
        '''command from in file'''
        file = open(self.input_file, 'r')
        context = file.readlines()
        file.close()
        self.md_task = context[0].strip()
        for val in context:
            val = val.strip()
            if val and val[0] != '#' and ("=" in val):
                val = val[:val.index(",")] if ',' in val else val
                assert len(val.strip().split("=")) == 2
                flag, value = val.strip().split("=")
                value = value.replace(" ", "")
                flag = flag.replace(" ", "")
                if flag not in self.command_set:
                    self.command_set[flag] = value
                else:
                    print("ERROR COMMAND FILE")


class NPT(nn.Cell):
    '''npt'''

    def __init__(self, args_opt):
        super(NPT, self).__init__()
        self.control = Controller(args_opt)
        self.md_info = MdInformation(self.control)
        self.mode = self.md_info.mode
        self.update_step = 0
        self.bond = Bond(self.control)
        self.bond_is_initialized = self.bond.is_initialized
        self.angle = Angle(self.control)
        self.angle_is_initialized = self.angle.is_initialized
        self.dihedral = Dihedral(self.control)
        self.dihedral_is_initialized = self.dihedral.is_initialized
        self.nb14 = NonBond14(
            self.control, self.dihedral, self.md_info.atom_numbers)
        self.nb14_is_initialized = self.nb14.is_initialized
        self.nb_info = NeighborList(
            self.control, self.md_info.atom_numbers, self.md_info.box_length)
        self.lj_info = LennardJonesInformation(
            self.control, self.md_info.nb.cutoff, self.md_info.sys.box_length)
        self.lj_info_is_initialized = self.lj_info.is_initialized

        self.liujian_info = LangevinLiujian(
            self.control, self.md_info.atom_numbers)
        self.liujian_info_is_initialized = self.liujian_info.is_initialized
        self.pme_method = ParticleMeshEwald(self.control, self.md_info)
        self.pme_is_initialized = self.pme_method.is_initialized
        self.restrain = RestrainInformation(
            self.control, self.md_info.atom_numbers, self.md_info.crd)
        self.restrain_is_initialized = self.restrain.is_initialized
        self.simple_constrain_is_initialized = 0

        self.simple_constrain = SimpleConstarin(
            self.control, self.md_info, self.bond, self.angle, self.liujian_info)
        self.simple_constrain_is_initialized = self.simple_constrain.is_initialized
        self.freedom = self.simple_constrain.system_freedom

        self.vatom = VirtualInformation(
            self.control, self.md_info, self.md_info.sys.freedom)
        self.vatom_is_initialized = 1

        self.random = P.UniformReal(seed=1)
        self.pow = P.Pow()
        self.five = Tensor(5.0, mstype.float32)
        self.third = Tensor(1 / 3, mstype.float32)
        self.mol_map = CoordinateMolecularMap(self.md_info.atom_numbers, self.md_info.sys.box_length, self.md_info.crd,
                                              self.md_info.nb.excluded_atom_numbers, self.md_info.nb.h_excluded_numbers,
                                              self.md_info.nb.h_excluded_list_start, self.md_info.nb.h_excluded_list)
        self.mol_map_is_initialized = 1
        self.init_params()
        self.init_tensor_1()
        self.init_tensor_2()
        self.op_define_1()
        self.op_define_2()
        self.depend = P.Depend()
        self.total_count = Parameter(
            Tensor(0, mstype.int32), requires_grad=False)
        self.accept_count = Parameter(
            Tensor(0, mstype.int32), requires_grad=False)
        self.is_molecule_map_output = self.md_info.output.is_molecule_map_output
        self.target_pressure = Tensor([self.md_info.sys.target_pressure], mstype.float32)
        self.nx = self.nb_info.nx
        self.ny = self.nb_info.ny
        self.nz = self.nb_info.nz
        self.nxyz = Tensor([self.nx, self.ny, self.nz], mstype.int32)
        self.pme_inverse_box_vector = Parameter(Tensor(
            self.pme_method.pme_inverse_box_vector, mstype.float32), requires_grad=False)
        self.pme_inverse_box_vector_init = Parameter(Tensor(
            self.pme_method.pme_inverse_box_vector, mstype.float32), requires_grad=False)
        self.mc_baro_is_initialized = 0
        self.bd_baro_is_initialized = 0
        self.constant_uint_max_float = 4294967296.0
        self.volume = Parameter(Tensor(self.pme_method.volume, mstype.float32), requires_grad=False)
        self.crd_scale_factor = Parameter(Tensor([1.0,], mstype.float32), requires_grad=False)

        self.bd_baro = BDBARO(self.control, self.md_info.sys.target_pressure,
                              self.md_info.sys.box_length, self.md_info.mode)
        self.bd_baro_is_initialized = self.bd_baro.is_initialized
        self.update_interval = Tensor([self.bd_baro.update_interval], mstype.float32)
        self.pressure = Parameter(Tensor([self.md_info.sys.d_pressure,], mstype.float32), requires_grad=False)
        self.compressibility = Tensor([self.bd_baro.compressibility], mstype.float32)
        self.bd_baro_dt = Tensor([self.bd_baro.dt], mstype.float32)
        self.bd_baro_taup = Tensor([self.bd_baro.taup], mstype.float32)
        self.system_reinitializing_count = Parameter(
            Tensor(0, mstype.int32), requires_grad=False)
        self.bd_baro_newv = Parameter(
            Tensor(self.bd_baro.new_v, mstype.float32), requires_grad=False)
        self.bd_baro_v0 = Parameter(
            Tensor(self.bd_baro.v0, mstype.float32), requires_grad=False)

    def init_params(self):
        '''init params'''
        self.bond_energy_sum = Tensor(0, mstype.int32)
        self.angle_energy_sum = Tensor(0, mstype.int32)
        self.dihedral_energy_sum = Tensor(0, mstype.int32)
        self.nb14_lj_energy_sum = Tensor(0, mstype.int32)
        self.nb14_cf_energy_sum = Tensor(0, mstype.int32)
        self.lj_energy_sum = Tensor(0, mstype.int32)
        self.ee_ene = Tensor(0, mstype.int32)
        self.total_energy = Tensor(0, mstype.int32)
        self.ntwx = self.md_info.ntwx
        self.atom_numbers = self.md_info.atom_numbers
        self.residue_numbers = self.md_info.residue_numbers
        self.bond_numbers = self.bond.bond_numbers
        self.angle_numbers = self.angle.angle_numbers
        self.dihedral_numbers = self.dihedral.dihedral_numbers
        self.nb14_numbers = self.nb14.nb14_numbers
        self.nxy = self.nb_info.nxy
        self.grid_numbers = self.nb_info.grid_numbers
        self.max_atom_in_grid_numbers = self.nb_info.max_atom_in_grid_numbers
        self.max_neighbor_numbers = self.nb_info.max_neighbor_numbers
        self.excluded_atom_numbers = self.md_info.nb.excluded_atom_numbers
        self.refresh_count = Parameter(
            Tensor(self.nb_info.refresh_count, mstype.int32), requires_grad=False)
        self.refresh_interval = self.nb_info.refresh_interval
        self.skin = self.nb_info.skin
        self.cutoff = self.nb_info.cutoff
        self.cutoff_square = self.nb_info.cutoff_square
        self.cutoff_with_skin = self.nb_info.cutoff_with_skin
        self.half_cutoff_with_skin = self.nb_info.half_cutoff_with_skin
        self.cutoff_with_skin_square = self.nb_info.cutoff_with_skin_square
        self.half_skin_square = self.nb_info.half_skin_square
        self.beta = self.pme_method.beta
        self.d_beta = Parameter(Tensor([self.pme_method.beta], mstype.float32), requires_grad=False)
        self.d_beta_init = Parameter(Tensor([self.pme_method.beta], mstype.float32), requires_grad=False)
        self.neutralizing_factor = Parameter(Tensor([self.pme_method.neutralizing_factor], mstype.float32),
                                             requires_grad=False)
        self.fftx = self.pme_method.fftx
        self.ffty = self.pme_method.ffty
        self.fftz = self.pme_method.fftz
        self.random_seed = self.liujian_info.random_seed
        self.dt = self.liujian_info.dt
        self.half_dt = self.liujian_info.half_dt
        self.exp_gamma = self.liujian_info.exp_gamma
        self.update = False
        self.file = None
        self.datfile = None
        self.max_velocity = self.liujian_info.max_velocity

        self.constant_kb = 0.00198716

    def init_tensor_1(self):
        '''init tensor'''
        self.uint_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.uint32), mstype.uint32),
                                  requires_grad=False)
        self.need_potential = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.need_pressure = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.atom_energy = Parameter(Tensor([0] * self.atom_numbers, mstype.float32), requires_grad=False)
        self.atom_virial = Parameter(Tensor([0] * self.atom_numbers, mstype.float32), requires_grad=False)
        self.frc = Parameter(Tensor(np.zeros([self.atom_numbers, 3]), mstype.float32), requires_grad=False)

        self.crd = Parameter(
            Tensor(np.array(self.md_info.coordinate).reshape(
                [self.atom_numbers, 3]), mstype.float32),
            requires_grad=False)
        self.crd_to_uint_crd_cof = Parameter(Tensor(
            self.md_info.pbc.crd_to_uint_crd_cof, mstype.float32), requires_grad=False)
        self.quarter_crd_to_uint_crd_cof = Parameter(Tensor(
            self.md_info.pbc.quarter_crd_to_uint_crd_cof, mstype.float32), requires_grad=False)
        self.uint_dr_to_dr_cof = Parameter(
            Tensor(self.md_info.pbc.uint_dr_to_dr_cof, mstype.float32), requires_grad=False)
        self.box_length = Parameter(
            Tensor(self.md_info.box_length, mstype.float32), requires_grad=False)
        self.box_length_1 = Tensor(self.md_info.box_length, mstype.float32)
        self.charge = Parameter(Tensor(np.asarray(self.md_info.h_charge), mstype.float32), requires_grad=False)
        self.old_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3]), mstype.float32), requires_grad=False)
        self.last_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3]), mstype.float32), requires_grad=False)
        self.mass = Tensor(self.md_info.h_mass, mstype.float32)
        self.mass_inverse = Tensor(self.md_info.h_mass_inverse, mstype.float32)
        self.res_mass = Tensor(self.md_info.res.h_mass, mstype.float32)
        self.res_mass_inverse = Tensor(
            self.md_info.res.h_mass_inverse, mstype.float32)
        self.velocity = Parameter(
            Tensor(self.md_info.velocity, mstype.float32), requires_grad=False)
        self.acc = Parameter(Tensor(np.zeros(
            [self.atom_numbers, 3], np.float32), mstype.float32), requires_grad=False)
        self.bond_atom_a = Tensor(np.asarray(
            self.bond.h_atom_a, np.int32), mstype.int32)
        self.bond_atom_b = Tensor(np.asarray(
            self.bond.h_atom_b, np.int32), mstype.int32)
        self.bond_k = Tensor(np.asarray(
            self.bond.h_k, np.float32), mstype.float32)
        self.bond_r0 = Tensor(np.asarray(
            self.bond.h_r0, np.float32), mstype.float32)
        self.angle_atom_a = Tensor(np.asarray(
            self.angle.h_atom_a, np.int32), mstype.int32)
        self.angle_atom_b = Tensor(np.asarray(
            self.angle.h_atom_b, np.int32), mstype.int32)
        self.angle_atom_c = Tensor(np.asarray(
            self.angle.h_atom_c, np.int32), mstype.int32)
        self.angle_k = Tensor(np.asarray(
            self.angle.h_angle_k, np.float32), mstype.float32)
        self.angle_theta0 = Tensor(np.asarray(
            self.angle.h_angle_theta0, np.float32), mstype.float32)
        self.dihedral_atom_a = Tensor(np.asarray(
            self.dihedral.h_atom_a, np.int32), mstype.int32)
        self.dihedral_atom_b = Tensor(np.asarray(
            self.dihedral.h_atom_b, np.int32), mstype.int32)
        self.dihedral_atom_c = Tensor(np.asarray(
            self.dihedral.h_atom_c, np.int32), mstype.int32)
        self.dihedral_atom_d = Tensor(np.asarray(
            self.dihedral.h_atom_d, np.int32), mstype.int32)
        self.pk = Tensor(np.asarray(self.dihedral.h_pk,
                                    np.float32), mstype.float32)
        self.gamc = Tensor(np.asarray(
            self.dihedral.h_gamc, np.float32), mstype.float32)
        self.gams = Tensor(np.asarray(
            self.dihedral.h_gams, np.float32), mstype.float32)
        self.pn = Tensor(np.asarray(self.dihedral.h_pn,
                                    np.float32), mstype.float32)
        self.ipn = Tensor(np.asarray(
            self.dihedral.h_ipn, np.int32), mstype.int32)
    def init_tensor_2(self):
        '''init tensor 2'''
        self.nb14_atom_a = Tensor(np.asarray(
            self.nb14.h_atom_a, np.int32), mstype.int32)
        self.nb14_atom_b = Tensor(np.asarray(
            self.nb14.h_atom_b, np.int32), mstype.int32)
        self.lj_scale_factor = Tensor(np.asarray(
            self.nb14.h_lj_scale_factor, np.float32), mstype.float32)
        self.cf_scale_factor = Tensor(np.asarray(
            self.nb14.h_cf_scale_factor, np.float32), mstype.float32)
        self.grid_n = Tensor(self.nb_info.grid_n, mstype.int32)
        self.grid_length = Parameter(
            Tensor(self.nb_info.grid_length, mstype.float32), requires_grad=False)
        self.grid_length_inverse = Parameter(
            Tensor(self.nb_info.grid_length_inverse, mstype.float32), requires_grad=False)
        self.bucket = Parameter(Tensor(
            np.asarray(self.nb_info.bucket, np.int32).reshape(
                [self.grid_numbers, self.max_atom_in_grid_numbers]),
            mstype.int32), requires_grad=False)  # Tobe updated
        self.bucket_init = Parameter(Tensor(
            np.asarray(self.nb_info.bucket, np.int32).reshape(
                [self.grid_numbers, self.max_atom_in_grid_numbers]),
            mstype.int32), requires_grad=False)  # Tobe updated
        self.atom_numbers_in_grid_bucket = Parameter(Tensor(self.nb_info.atom_numbers_in_grid_bucket, mstype.int32),
                                                     requires_grad=False)  # to be updated
        self.atom_numbers_in_grid_bucket_init = Parameter(
            Tensor(self.nb_info.atom_numbers_in_grid_bucket, mstype.int32),
            requires_grad=False)  # to be updated
        self.atom_in_grid_serial = Parameter(Tensor(np.zeros([self.nb_info.atom_numbers,], np.int32), mstype.int32),
                                             requires_grad=False)  # to be updated
        self.atom_in_grid_serial_init = Parameter(
            Tensor(np.zeros([self.nb_info.atom_numbers,], np.int32), mstype.int32),
            requires_grad=False)  # to be updated
        self.pointer = Parameter(
            Tensor(np.asarray(self.nb_info.pointer, np.int32).reshape(
                [self.grid_numbers, 125]), mstype.int32),
            requires_grad=False)
        self.pointer_init = Parameter(
            Tensor(np.asarray(self.nb_info.pointer, np.int32).reshape(
                [self.grid_numbers, 125]), mstype.int32),
            requires_grad=False)
        self.nl_atom_numbers = Parameter(Tensor(np.zeros([self.atom_numbers,], np.int32), mstype.int32),
                                         requires_grad=False)
        self.nl_atom_serial = Parameter(
            Tensor(np.zeros(
                [self.atom_numbers, self.max_neighbor_numbers], np.int32), mstype.int32),
            requires_grad=False)

        excluded_list_start = np.asarray(self.md_info.nb.h_excluded_list_start, np.int32)
        self.excluded_list_start = Tensor(excluded_list_start)
        excluded_list = np.asarray(self.md_info.nb.h_excluded_list, np.int32)
        self.excluded_list = Tensor(excluded_list)
        excluded_numbers = np.asarray(self.md_info.nb.h_excluded_numbers, np.int32)
        self.excluded_numbers = Tensor(excluded_numbers)

        self.need_refresh_flag = Tensor(np.asarray([0], np.int32), mstype.int32)
        self.atom_lj_type = Tensor(self.lj_info.atom_lj_type, mstype.int32)
        self.lj_a = Tensor(self.lj_info.h_lj_a, mstype.float32)
        self.lj_b = Tensor(self.lj_info.h_lj_b, mstype.float32)
        self.sqrt_mass = Tensor(self.liujian_info.h_sqrt_mass, mstype.float32)
        self.rand_state = Parameter(
            Tensor(self.liujian_info.rand_state, mstype.float32))
        self.zero_fp_tensor = Tensor(np.asarray([0,], np.float32))
        self.set_zero = Parameter(Tensor([0.0,], mstype.float32), requires_grad=False)
        self.set_zero_int = Parameter(Tensor(np.array([0,]), mstype.int32), requires_grad=False)

        self.zero_frc = Parameter(Tensor(np.zeros([self.atom_numbers, 3]), mstype.float32), requires_grad=False)
        self.zero_main_force = Parameter(Tensor(np.zeros([self.atom_numbers, 3]), mstype.float32), requires_grad=False)
        self.zero_ene = Parameter(Tensor([0,] * self.atom_numbers, mstype.float32), requires_grad=False)
        self.virial = Parameter(Tensor([0.0,], mstype.float32), requires_grad=False)
        self.potential = Parameter(Tensor([0.0,], mstype.float32), requires_grad=False)
        self.write_trajectory_interval = Tensor(self.md_info.output.write_trajectory_interval, mstype.int32)
        res_start = np.asarray(self.md_info.h_res_start, np.int32)
        res_end = np.asarray(self.md_info.h_res_end, np.int32)
        residual_indptr, residual_indices, residual_value = get_csr_residual_index(self.atom_numbers,
                                                                                   res_start, res_end)
        self.csr_residual_indptr = Tensor(residual_indptr)
        self.csr_residual_indices = Tensor(residual_indices)
        self.csr_residual_value = Tensor(residual_value)
        self.csr_residual_mask = CSRTensor(self.csr_residual_indptr, self.csr_residual_indices, self.csr_residual_value,
                                           (self.residue_numbers, self.atom_numbers))
        excluded_indptr, excluded_indices, excluded_value = \
            get_csr_excluded_index(excluded_list, excluded_list_start, excluded_numbers)
        self.atom_excluded_index = broadcast_with_excluded_list(excluded_indptr)
        self.csr_excluded_indptr = Tensor(excluded_indptr)
        self.csr_excluded_indices = Tensor(excluded_indices)
        self.csr_excluded_value = Tensor(excluded_value)
        self.csr_excluded_list = CSRTensor(self.csr_excluded_indptr, self.csr_excluded_indices, self.csr_excluded_value,
                                           (self.atom_numbers, self.csr_excluded_value.shape[0]))
        self.box_length_0 = self.md_info.box_length[0]
        self.box_length_1 = self.md_info.box_length[1]
        self.box_length_2 = self.md_info.box_length[2]
        box = (self.box_length_0, self.box_length_1, self.box_length_2)
        self.pme_bc = Tensor(get_pme_bc(self.fftx, self.ffty, self.fftz, box, self.beta), mstype.float32)


    def op_define_1(self):
        '''op define'''
        self.setup_random_state = P.MDIterationSetupRandState(
            self.atom_numbers, self.random_seed)

        self.neighbor_list_update = P.NeighborListRefresh(grid_numbers=self.grid_numbers,
                                                          atom_numbers=self.atom_numbers,
                                                          not_first_time=1, nxy=self.nxy,
                                                          excluded_atom_numbers=self.excluded_atom_numbers,
                                                          cutoff_square=self.cutoff_square,
                                                          half_skin_square=self.half_skin_square,
                                                          cutoff_with_skin=self.cutoff_with_skin,
                                                          half_cutoff_with_skin=self.half_cutoff_with_skin,
                                                          cutoff_with_skin_square=self.cutoff_with_skin_square,
                                                          refresh_interval=self.refresh_interval, cutoff=self.cutoff,
                                                          skin=self.skin,
                                                          max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                                          max_neighbor_numbers=self.max_neighbor_numbers)

        self.neighbor_list_update_forced_update = \
            P.NeighborListRefresh(grid_numbers=self.grid_numbers,
                                  atom_numbers=self.atom_numbers,
                                  not_first_time=1,
                                  nxy=self.nxy,
                                  excluded_atom_numbers=self.excluded_atom_numbers,
                                  cutoff_square=self.cutoff_square,
                                  half_skin_square=self.half_skin_square,
                                  cutoff_with_skin=self.cutoff_with_skin,
                                  half_cutoff_with_skin=self.half_cutoff_with_skin,
                                  cutoff_with_skin_square=self.cutoff_with_skin_square,
                                  refresh_interval=self.refresh_interval,
                                  cutoff=self.cutoff,
                                  skin=self.skin,
                                  max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                  max_neighbor_numbers=self.max_neighbor_numbers,
                                  forced_update=1)

    def op_define_2(self):
        '''op define 2'''
        self.neighbor_list_update_nb = \
            P.NeighborListRefresh(grid_numbers=self.grid_numbers,
                                  atom_numbers=self.atom_numbers,
                                  not_first_time=1, nxy=self.nxy,
                                  excluded_atom_numbers=self.excluded_atom_numbers,
                                  cutoff_square=self.cutoff_square,
                                  half_skin_square=self.half_skin_square,
                                  cutoff_with_skin=self.cutoff_with_skin,
                                  half_cutoff_with_skin=self.half_cutoff_with_skin,
                                  cutoff_with_skin_square=self.cutoff_with_skin_square,
                                  refresh_interval=self.refresh_interval,
                                  cutoff=self.cutoff,
                                  skin=self.skin,
                                  max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                  max_neighbor_numbers=self.max_neighbor_numbers,
                                  forced_update=1, forced_check=1)

        self.neighbor_list_update_pres = P.NeighborListRefresh(grid_numbers=self.grid_numbers,
                                                               atom_numbers=self.atom_numbers,
                                                               not_first_time=1, nxy=self.nxy,
                                                               excluded_atom_numbers=self.excluded_atom_numbers,
                                                               cutoff_square=self.cutoff_square,
                                                               half_skin_square=self.half_skin_square,
                                                               cutoff_with_skin=self.cutoff_with_skin,
                                                               half_cutoff_with_skin=self.half_cutoff_with_skin,
                                                               cutoff_with_skin_square=self.cutoff_with_skin_square,
                                                               refresh_interval=self.refresh_interval,
                                                               cutoff=self.cutoff,
                                                               skin=self.skin,
                                                               max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                                               max_neighbor_numbers=self.max_neighbor_numbers,
                                                               forced_update=0, forced_check=1)

        self.random_force = Tensor(
            np.zeros([self.atom_numbers, 3], np.float32), mstype.float32)

        # simple_constrain
        self.constrain_pair_numbers = self.simple_constrain.constrain_pair_numbers
        self.zero_pair_virial = Parameter(Tensor(np.zeros([self.constrain_pair_numbers,]), mstype.float32),
                                          requires_grad=False)
        self.last_pair_dr = Parameter(Tensor(np.zeros(
            [self.constrain_pair_numbers, 3], np.float32), mstype.float32), requires_grad=False)
        if self.simple_constrain_is_initialized:
            self.constrain_pair_numbers = self.simple_constrain.constrain_pair_numbers
            self.constrain_pair = np.array(
                self.simple_constrain.h_constrain_pair)
            self.atom_i_serials = Tensor(
                self.constrain_pair[:, 0], mstype.int32)
            self.atom_j_serials = Tensor(
                self.constrain_pair[:, 1], mstype.int32)
            self.constant_rs = Tensor(
                self.constrain_pair[:, 2], mstype.float32)
            self.constrain_ks = Tensor(
                self.constrain_pair[:, 3], mstype.float32)
            self.last_pair_dr = Parameter(Tensor(np.zeros(
                [self.constrain_pair_numbers, 3], np.float32), mstype.float32), requires_grad=False)
            self.constrain_frc = Parameter(Tensor(np.zeros(
                [self.atom_numbers, 3], np.float32), mstype.float32), requires_grad=False)
            self.iteration_numbers = self.simple_constrain.info.iteration_numbers
            self.half_exp_gamma_plus_half = self.simple_constrain.half_exp_gamma_plus_half
            self.dt_inverse = self.simple_constrain.dt_inverse
        if self.mol_map_is_initialized:
            self.box_map_times = Parameter(
                Tensor(self.mol_map.h_box_map_times, mstype.int32), requires_grad=False)

    def simulation_beforce_caculate_force(self):
        '''simulation before calculate force'''
        self.uint_crd = crd_to_uint_crd_quarter(
            self.quarter_crd_to_uint_crd_cof, self.crd)
        return self.uint_crd

    def simulation_caculate_force(self, uint_crd, scaler, nl_atom_numbers, nl_atom_serial):
        '''simulation calculate force'''
        uint_crd = self.simulation_beforce_caculate_force()
        self.need_pressure = 0
        self.virial = 0
        self.atom_virial = self.zero_ene
        if self.bd_baro_is_initialized:
            self.need_pressure += 1

        if self.lj_info_is_initialized:
            lj_force, atom_virial, _ = lj_force_with_virial_energy(self.atom_numbers, self.cutoff, self.beta, uint_crd,
                                                                   self.atom_lj_type, self.charge, scaler,
                                                                   nl_atom_numbers, nl_atom_serial, self.lj_a,
                                                                   self.lj_b)
            self.atom_virial += atom_virial
        else:
            lj_force = self.zero_main_force

        if self.pme_is_initialized:
            pme_excluded_force_res = pme_excluded_force(self.atom_numbers, self.beta, uint_crd, scaler, self.charge,
                                                        excluded_csr=self.csr_excluded_list,
                                                        excluded_row=self.atom_excluded_index)
            pme_reciprocal_force_res = pme_reciprocal_force(self.atom_numbers, self.fftx, self.ffty, self.fftz,
                                                            self.box_length_0, self.box_length_1, self.box_length_2,
                                                            self.pme_bc, uint_crd, self.charge)
            reciprocal_energy, self_energy, direct_energy, correction_energy = \
                                                    pme_energy_update(self.atom_numbers,
                                                                      self.beta,
                                                                      self.fftx, self.ffty, self.fftz,
                                                                      self.pme_bc, uint_crd,
                                                                      self.charge, self.nl_atom_numbers,
                                                                      self.nl_atom_serial,
                                                                      self.uint_dr_to_dr_cof,
                                                                      self.csr_excluded_value,
                                                                      self.atom_excluded_index,
                                                                      self.neutralizing_factor)
            self.virial = reciprocal_energy + self_energy + direct_energy + correction_energy
            pme_force = pme_excluded_force_res + pme_reciprocal_force_res
        else:
            pme_force = self.zero_main_force

        if self.nb14_is_initialized:
            nb14_force, _, atom_virial = \
                dihedral_14_ljcf_force_with_atom_energy_with_virial(self.atom_numbers,
                                                                    uint_crd, self.atom_lj_type,
                                                                    self.charge, scaler,
                                                                    self.nb14_atom_a,
                                                                    self.nb14_atom_b,
                                                                    self.lj_scale_factor,
                                                                    self.cf_scale_factor,
                                                                    self.lj_a, self.lj_b)
            self.atom_virial += atom_virial
        else:
            nb14_force = self.zero_main_force

        if self.bond_is_initialized:
            bond_force, _, atom_virial = bond_force_with_atom_energy_with_virial(self.atom_numbers,
                                                                                 uint_crd, scaler,
                                                                                 self.bond_atom_a,
                                                                                 self.bond_atom_b, self.bond_k,
                                                                                 self.bond_r0)
            self.atom_virial += atom_virial
        else:
            bond_force = self.zero_main_force

        if self.angle_is_initialized:
            angle_force, _ = angle_force_with_atom_energy(self.angle_numbers, uint_crd, scaler, self.angle_atom_a,
                                                          self.angle_atom_b, self.angle_atom_c,
                                                          self.angle_k, self.angle_theta0)
        else:
            angle_force = self.zero_main_force

        if self.dihedral_is_initialized:
            dihedral_force, _ = dihedral_force_with_atom_energy(self.dihedral_numbers, uint_crd, scaler,
                                                                self.dihedral_atom_a,
                                                                self.dihedral_atom_b,
                                                                self.dihedral_atom_c,
                                                                self.dihedral_atom_d, self.ipn,
                                                                self.pk, self.gamc, self.gams,
                                                                self.pn)
        else:
            dihedral_force = self.zero_main_force

        force = P.AddN()([lj_force, pme_force, nb14_force, bond_force, angle_force, dihedral_force])
        return force, self.atom_virial, self.virial, self.need_pressure

    def simulation_caculate_energy(self, uint_crd, uint_dr_to_dr_cof):
        '''simulation calculate energy'''
        lj_energy_res = lj_energy(self.atom_numbers, self.cutoff_square, uint_crd, self.atom_lj_type,
                                  uint_dr_to_dr_cof, self.nl_atom_numbers,
                                  self.nl_atom_serial, self.lj_a, self.lj_b)
        lj_energy_sum = lj_energy_res.sum(keepdims=True)
        reciprocal_energy, self_energy, direct_energy, correction_energy = \
            pme_energy_update(self.atom_numbers, self.beta, self.fftx, self.ffty, self.fftz,
                              self.pme_bc, uint_crd, self.charge, self.nl_atom_numbers,
                              self.nl_atom_serial, self.uint_dr_to_dr_cof,
                              self.csr_excluded_value, self.atom_excluded_index,
                              self.neutralizing_factor)
        ee_ene = reciprocal_energy + self_energy + direct_energy + correction_energy

        nb14_lj_energy_res = nb14_lj_energy(self.nb14_numbers, self.atom_numbers, uint_crd, self.atom_lj_type,
                                            self.charge, uint_dr_to_dr_cof, self.nb14_atom_a, self.nb14_atom_b,
                                            self.lj_scale_factor, self.lj_a, self.lj_b)
        nb14_cf_energy_res = nb14_cf_energy(self.nb14_numbers, self.atom_numbers, uint_crd, self.atom_lj_type,
                                            self.charge, uint_dr_to_dr_cof, self.nb14_atom_a, self.nb14_atom_b,
                                            self.cf_scale_factor)
        nb14_lj_energy_sum = P.ReduceSum(True)(nb14_lj_energy_res)
        nb14_cf_energy_sum = P.ReduceSum(True)(nb14_cf_energy_res)

        bond_energy_res = bond_energy(self.atom_numbers, self.bond_numbers, uint_crd,
                                      uint_dr_to_dr_cof, self.bond_atom_a, self.bond_atom_b,
                                      self.bond_k, self.bond_r0)
        bond_energy_sum = P.ReduceSum(True)(bond_energy_res)

        angle_energy_res = angle_energy(self.angle_numbers, uint_crd, uint_dr_to_dr_cof, self.angle_atom_a,
                                        self.angle_atom_b, self.angle_atom_c, self.angle_k, self.angle_theta0)
        angle_energy_sum = P.ReduceSum(True)(angle_energy_res)

        dihedral_energy_res = dihedral_energy(self.dihedral_numbers, uint_crd, uint_dr_to_dr_cof,
                                              self.dihedral_atom_a, self.dihedral_atom_b, self.dihedral_atom_c,
                                              self.dihedral_atom_d, self.ipn, self.pk, self.gamc, self.gams, self.pn)
        dihedral_energy_sum = P.ReduceSum(True)(dihedral_energy_res)

        total_energy = P.AddN()(
            [bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum,
             lj_energy_sum, ee_ene])
        return bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
               lj_energy_sum, ee_ene, total_energy

    def simulation_temperature(self):
        '''caculate temperature'''

        res_ek_energy = md_temperature(self.csr_residual_mask, self.velocity, self.mass, sparse=True)
        temperature = res_ek_energy.sum()
        return temperature

    def simulation_mditerationleapfrog_liujian(self, inverse_mass, sqrt_mass_inverse, crd, frc):
        '''simulation leap frog iteration liujian'''
        if self.max_velocity <= 0:
            return md_iteration_leap_frog_liujian(self.atom_numbers, self.half_dt, self.dt, self.exp_gamma,
                                                  inverse_mass, sqrt_mass_inverse, self.velocity, crd, frc,
                                                  self.acc)
        return md_iteration_leap_frog_liujian_with_max_vel(self.atom_numbers, self.half_dt, self.dt,
                                                           self.exp_gamma, inverse_mass, sqrt_mass_inverse,
                                                           self.velocity, crd, frc, self.acc, self.max_velocity)

    def simulation_mditerationleapfrog(self, force):
        '''simulation leap frog'''
        if self.max_velocity <= 0:
            res = self.md_iteration_leap_frog(
                self.velocity, self.crd, force, self.acc, self.mass_inverse)
        else:
            res = self.md_iteration_leap_frog_with_max_vel(
                self.velocity, self.crd, force, self.acc, self.mass_inverse)
        vel = F.depend(self.velocity, res)
        crd = F.depend(self.crd, res)
        return vel, crd, res

    def simulation_mdinformationgradientdescent(self, force):
        res = self.md_information_gradient_descent(self.crd, force)
        self.velocity = self.zero_frc
        vel = F.depend(self.velocity, res)
        crd = F.depend(self.crd, res)
        return vel, crd, res

    def main_print(self, *args):
        """compute the temperature"""
        steps, temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, lj_energy_sum, ee_ene = list(
            args)
        if steps == 1:
            print("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                  "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_")

        temperature = temperature.asnumpy()
        total_potential_energy = total_potential_energy.asnumpy()
        print("{:>7.0f} {:>7.3f} {:>11.3f}".format(steps, float(temperature), float(total_potential_energy)),
              end=" ")
        if self.bond.bond_numbers > 0:
            sigma_of_bond_ene = sigma_of_bond_ene.asnumpy()
            print("{:>10.3f}".format(float(sigma_of_bond_ene)), end=" ")
        if self.angle.angle_numbers > 0:
            sigma_of_angle_ene = sigma_of_angle_ene.asnumpy()
            print("{:>11.3f}".format(float(sigma_of_angle_ene)), end=" ")
        if self.dihedral.dihedral_numbers > 0:
            sigma_of_dihedral_ene = sigma_of_dihedral_ene.asnumpy()
            print("{:>14.3f}".format(float(sigma_of_dihedral_ene)), end=" ")
        if self.nb14.nb14_numbers > 0:
            nb14_lj_energy_sum = nb14_lj_energy_sum.asnumpy()
            nb14_cf_energy_sum = nb14_cf_energy_sum.asnumpy()
            print("{:>10.3f} {:>10.3f}".format(
                float(nb14_lj_energy_sum), float(nb14_cf_energy_sum)), end=" ")
        lj_energy_sum = lj_energy_sum.asnumpy()
        ee_ene = ee_ene.asnumpy()
        print("{:>7.3f}".format(float(lj_energy_sum)), end=" ")
        print("{:>12.3f}".format(float(ee_ene)))
        if self.file is not None:
            self.file.write("{:>7.0f} {:>7.3f} {:>11.3f} {:>10.3f} {:>11.3f} {:>14.3f} {:>10.3f} {:>10.3f} {:>7.3f}"
                            " {:>12.3f}\n".format(steps, float(temperature), float(total_potential_energy),
                                                  float(sigma_of_bond_ene), float(sigma_of_angle_ene),
                                                  float(sigma_of_dihedral_ene), float(nb14_lj_energy_sum),
                                                  float(nb14_cf_energy_sum), float(lj_energy_sum), float(ee_ene)))
        if self.datfile is not None:
            self.datfile.write(self.crd.asnumpy())

    def export_restart_file(self):
        '''export restart file'''
        filename = self.control.restrt
        file = open(filename, "w")
        file.write("mask\n")
        file.write(str(self.atom_numbers) + " " + "20210805 \n")
        vel = self.velocity.asnumpy()
        crd = self.crd.asnumpy()
        box_length = self.box_length.asnumpy()
        if self.atom_numbers % 2 == 0:
            for i in range(0, self.atom_numbers, 2):
                file.write("{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}\n".format(
                    float(crd[i][0]), float(crd[i][1]), float(crd[i][2]),
                    float(crd[i + 1][0]), float(crd[i + 1][1]), float(crd[i + 1][2])))
            for i in range(0, self.atom_numbers, 2):
                file.write("{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}\n".format(
                    float(vel[i][0]), float(vel[i][1]), float(vel[i][2]),
                    float(vel[i + 1][0]), float(vel[i + 1][1]), float(vel[i + 1][2])))
        else:
            for i in range(0, self.atom_numbers - 1, 2):
                file.write("{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}\n".format(
                    float(crd[i][0]), float(crd[i][1]), float(crd[i][2]),
                    float(crd[i + 1][0]), float(crd[i + 1][1]), float(crd[i + 1][2])))
            file.write("{:12.7f}{:12.7f}{:12.7f}\n".format(
                float(crd[-1][0]), float(crd[-1][1]), float(crd[-1][2])))
            for i in range(0, self.atom_numbers - 1, 2):
                file.write("{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}\n".format(
                    float(vel[i][0]), float(vel[i][1]), float(vel[i][2]),
                    float(vel[i + 1][0]), float(vel[i + 1][1]), float(vel[i + 1][2])))
            file.write("{:12.7f}{:12.7f}{:12.7f}\n".format(
                float(vel[-1][0]), float(vel[-1][1]), float(vel[-1][2])))

        file.write("{:12.7f} {:12.7f} {:12.7f} {:12.7f} {:12.7f} {:12.7f}\n".format(
            float(box_length[0]), float(box_length[1]), float(box_length[2]),
            90.0, 90.0, 90.0))
        file.close()

    def main_initial(self):
        """main initial"""
        if self.control.mdout:
            self.file = open(self.control.mdout, 'w')
            self.file.write("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                            "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_\n")
        if self.control.mdcrd:
            self.datfile = open(self.control.mdcrd, 'wb')

    def main_destroy(self):
        """main destroy"""
        if self.file is not None:
            self.file.close()
            print("Save .out file successfully!")
        if self.datfile is not None:
            self.datfile.close()
            print("Save .dat file successfully!")

    def main_volume_change(self, factor):
        '''main volume change'''
        self.box_length = factor * self.box_length
        self.crd_to_uint_crd_cof = self.constant_uint_max_float / self.box_length
        self.quarter_crd_to_uint_crd_cof = 0.25 * self.crd_to_uint_crd_cof
        self.uint_dr_to_dr_cof = 1.0 / self.crd_to_uint_crd_cof
        self.uint_crd = crd_to_uint_crd_quarter(self.quarter_crd_to_uint_crd_cof, self.crd)

        self.grid_length = self.box_length / self.nxyz
        self.grid_length_inverse = 1.0 / self.grid_length

        res = self.neighbor_list_update_pres(self.atom_numbers_in_grid_bucket, self.bucket,
                                             self.crd, self.box_length, self.grid_n,
                                             self.grid_length_inverse, self.atom_in_grid_serial,
                                             self.old_crd, self.crd_to_uint_crd_cof, self.uint_crd,
                                             self.pointer, self.nl_atom_numbers, self.nl_atom_serial,
                                             self.uint_dr_to_dr_cof, self.excluded_list_start, self.excluded_list,
                                             self.excluded_numbers, self.need_refresh_flag, self.refresh_count)  # Done

        self.volume = self.box_length[0] * self.box_length[1] * self.box_length[2]

        # PME_Update_Volume
        self.d_beta *= factor
        self.neutralizing_factor *= self.pow(factor, self.five)
        return res, self.volume, self.d_beta, self.neutralizing_factor


    def Constrain(self, constrain_frc, pair_virial_sum, update_step):
        "SIMPLE_CONSTARIN Constrain"
        test_uint_crd = self.uint_crd
        test_uint_crd, constrain_frc, pair_virial_sum = constrain(self.atom_numbers, self.constrain_pair_numbers,
                                                                  self.iteration_numbers, self.half_exp_gamma_plus_half,
                                                                  self.crd, self.quarter_crd_to_uint_crd_cof,
                                                                  self.mass_inverse, self.uint_dr_to_dr_cof,
                                                                  self.last_pair_dr, self.atom_i_serials,
                                                                  self.atom_j_serials, self.constant_rs,
                                                                  self.constrain_ks, update_step)

        pair_virial_sum = self.depend(pair_virial_sum, test_uint_crd)
        virial = P.ReduceSum(True)(pair_virial_sum)
        temp = (1.0 / self.dt / self.dt / 3.0 / self.volume) * virial
        self.pressure = self.pressure + temp * update_step

        crd, vel = refresh_crd_vel(self.dt_inverse, self.exp_gamma, self.half_exp_gamma_plus_half,
                                   self.crd, self.velocity, constrain_frc, self.mass_inverse)
        res = self.zero_fp_tensor
        return crd, vel, res, test_uint_crd, pair_virial_sum

    def main_iteration(self, update_step, force):
        '''main_Iteration'''
        if self.simple_constrain_is_initialized:
            self.last_pair_dr = last_crd_to_dr(self.crd, self.quarter_crd_to_uint_crd_cof, self.uint_dr_to_dr_cof,
                                               self.atom_i_serials, self.atom_j_serials)

        res1 = self.zero_fp_tensor
        if self.mode == 0:  # NVE
            self.velocity, self.crd, res2 = self.simulation_mditerationleapfrog(force)
        elif self.mode == -1:  # Minimization
            self.velocity, self.crd, res1 = self.simulation_mdinformationgradientdescent(force)
        else:
            if self.liujian_info_is_initialized:
                self.velocity, self.crd, _ = self. simulation_mditerationleapfrog_liujian(self.mass_inverse,
                                                                                          self.sqrt_mass,
                                                                                          self.crd, force)
        constrain_frc = self.zero_frc
        pair_virial_sum = self.zero_pair_virial

        self.crd, self.velocity, res2, test_uint_crd, pair_virial_sum = self.Constrain(constrain_frc, pair_virial_sum,
                                                                                       update_step)

        res3 = self.zero_fp_tensor
        res4 = self.zero_fp_tensor
        res5 = self.zero_fp_tensor
        return self.velocity, self.crd, res1, res2, res3, res4, res5, test_uint_crd, pair_virial_sum

    def get_pressure(self, vel, mass, atom_virial, d_virial, volume):
        ek = 0.5 * P.Mul()(P.ReduceSum(True)(vel * vel, 1), P.ExpandDims()(mass, -1))
        sum_of_atom_ek = P.ReduceSum()(ek)
        virial = P.ReduceSum()(atom_virial) + d_virial
        v_inverse = 1.0 / volume
        pressure = (sum_of_atom_ek * 2 + virial) / 3 * v_inverse
        return pressure

    def construct(self, step, print_step, update_step):
        '''construct'''
        if step == 1:
            res = self.neighbor_list_update_forced_update(self.atom_numbers_in_grid_bucket,
                                                          self.bucket,
                                                          self.crd,
                                                          self.box_length,
                                                          self.grid_n,
                                                          self.grid_length_inverse,
                                                          self.atom_in_grid_serial,
                                                          self.old_crd,
                                                          self.crd_to_uint_crd_cof,
                                                          self.uint_crd,
                                                          self.pointer,
                                                          self.nl_atom_numbers,
                                                          self.nl_atom_serial,
                                                          self.uint_dr_to_dr_cof,
                                                          self.excluded_list_start,
                                                          self.excluded_list,
                                                          self.excluded_numbers,
                                                          self.need_refresh_flag,
                                                          self.refresh_count)
            self.rand_state = self.setup_random_state()
        else:
            res = self.zero_fp_tensor
        force, self.atom_virial, self.virial, self.need_pressure = \
            self.simulation_caculate_force(self.uint_crd,
                                           self.uint_dr_to_dr_cof,
                                           self.nl_atom_numbers,
                                           self.nl_atom_serial)

        if update_step > 0:
            self.pressure = self.get_pressure(self.velocity, self.mass, self.atom_virial, self.virial, self.volume)
        self.velocity, self.crd, res1, res2, res3, res4, res5, test_uint_crd, _ = self.main_iteration(
            update_step, force)
        if update_step == 1:
            p_now = self.pressure

            self.crd_scale_factor = 1 - self.update_interval * self.compressibility * \
                                    self.bd_baro_dt / self.bd_baro_taup / 3 * (self.target_pressure - p_now)

            res3, self.volume, self.d_beta, self.neutralizing_factor = self.main_volume_change(self.crd_scale_factor)

        else:
            res3 = self.zero_fp_tensor
        self.uint_crd = crd_to_uint_crd_quarter(self.quarter_crd_to_uint_crd_cof, self.crd)

        res4 = self.neighbor_list_update(self.atom_numbers_in_grid_bucket,
                                         self.bucket,
                                         self.crd,
                                         self.box_length,
                                         self.grid_n,
                                         self.grid_length_inverse,
                                         self.atom_in_grid_serial,
                                         self.old_crd,
                                         self.crd_to_uint_crd_cof,
                                         self.uint_crd,
                                         self.pointer,
                                         self.nl_atom_numbers,
                                         self.nl_atom_serial,
                                         self.uint_dr_to_dr_cof,
                                         self.excluded_list_start,
                                         self.excluded_list,
                                         self.excluded_numbers,
                                         self.need_refresh_flag,
                                         self.refresh_count)

        res5 = refresh_boxmap_times(self.crd, self.old_crd, 1.0 / self.box_length, self.box_map_times)

        temperature = self.simulation_temperature()
        if print_step == 1:
            bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
            lj_energy_sum, ee_ene, total_energy = self.simulation_caculate_energy(self.uint_crd,
                                                                                  self.uint_dr_to_dr_cof)
        else:
            bond_energy_sum = self.zero_fp_tensor
            angle_energy_sum = self.zero_fp_tensor
            dihedral_energy_sum = self.zero_fp_tensor
            nb14_lj_energy_sum = self.zero_fp_tensor
            nb14_cf_energy_sum = self.zero_fp_tensor
            lj_energy_sum = self.zero_fp_tensor
            ee_ene = self.zero_fp_tensor
            total_energy = self.zero_fp_tensor
        return temperature, total_energy, bond_energy_sum, angle_energy_sum, dihedral_energy_sum, \
               nb14_lj_energy_sum, nb14_cf_energy_sum, lj_energy_sum, ee_ene, res, self.pressure, \
               res1, res2, res3, res4, res5, test_uint_crd
