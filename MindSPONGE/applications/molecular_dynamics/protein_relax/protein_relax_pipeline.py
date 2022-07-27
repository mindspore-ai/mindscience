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
'''main'''
import argparse
import os
import time
import warnings

import numpy as np
import mindspore as ms
from mindspore import context, nn, ops
from mindspore import numpy as mnp
from mindspore.common import Tensor
from mindsponge.common import residue_constants
from mindsponge.common.callback import RunInfo
from mindsponge.common.units import set_global_units
from mindsponge.common.utils import get_pdb_info
from mindsponge.data.hyperparam import ReconstructProtein as Protein
from mindsponge.data.parsers import read_pdb_via_xponge as read_pdb
from mindsponge.data.pdb_generator import gen_pdb
from mindsponge.metrics import get_structural_violations
from mindsponge.partition.neighbourlist import NeighbourList
from mindsponge.potential.energy import AngleEnergy, BondEnergy, DihedralEnergy, NB14Energy, NonBondEnergy
from mindsponge.potential.forcefield import ClassicalFF, Oscillator
from mindsponge.simulation import SimulationCell
from mindsponge.simulation.onestep import ClippedRunOneStepCell
from mindsponge.simulation.sponge import Sponge
from mindsponge.space.system import SystemCell


os.environ['GLOG_v'] = '3'
warnings.filterwarnings("ignore")
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Set the input pdb file path.")
parser.add_argument("-o", help="Set the output pdb file path.")
parser.add_argument("-addh", help="Set to 1 if need to add H atoms, default to be 1..", default=1)
args = parser.parse_args()
pdb_name = args.i
save_pdb_name = args.o
addh = args.addh

VIOLATION_TOLERANCE_ACTOR = 12.0
CLASH_OVERLAP_TOLERANCE = 1.5
C_ONE_HOT = nn.OneHot(depth=14)(Tensor(2, ms.int32))
N_ONE_HOT = nn.OneHot(depth=14)(Tensor(0, ms.int32))
DISTS_MASK_I = mnp.eye(14, 14)
CYS_SG_IDX = Tensor(5, ms.int32)
ATOMTYPE_RADIUS = Tensor(np.array(
    [1.55, 1.7, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.55, 1.55,
     1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.7, 1.55, 1.55, 1.55, 1.52, 1.52, 1.7, 1.55, 1.55,
     1.52, 1.7, 1.7, 1.7, 1.55, 1.52]), ms.float32)
LOWER_BOUND, UPPER_BOUND, RESTYPE_ATOM14_BOUND_STDDEV = \
    residue_constants.make_atom14_dists_bounds(overlap_tolerance=1.5, bond_length_tolerance_factor=12.0)
LOWER_BOUND = Tensor(LOWER_BOUND, ms.float32)
UPPER_BOUND = Tensor(UPPER_BOUND, ms.float32)
RESTYPE_ATOM14_BOUND_STDDEV = Tensor(RESTYPE_ATOM14_BOUND_STDDEV, ms.float32)


ms.set_seed(2333)
_, res_names, _, _, res_pointer, flatten_atoms, flatten_crds, init_res_names, init_res_ids, \
residue_index, aatype, atom14_positions, atom14_atom_exists, residx_atom14_to_atom37 = read_pdb(pdb_name, addh)

pdb_cell = Protein(res_names, res_pointer, flatten_atoms, init_res_names=init_res_names, init_res_ids=init_res_ids)

coordinates = flatten_crds
atomic_number = pdb_cell.atomic_numbers
h_mask = Tensor(np.where(atomic_number < 1.1, 1, 0)[None, :, None], ms.int32) # H:1, nonH:0
nonh_mask = 1-h_mask
atom_name = pdb_cell.atom_names
atom_type = pdb_cell.atom_types
resname = pdb_cell.res_names
resid = pdb_cell.res_id
init_resname = pdb_cell.init_res_names
init_resid = pdb_cell.init_res_ids
resid_num = int(init_resid[-1])
mass = pdb_cell.mass
charge = pdb_cell.charge
crd_mapping_ids = pdb_cell.crd_mapping_ids
bond_index = pdb_cell.bond_index
bond_params = pdb_cell.bond_params
rk_init = bond_params[:, 2]
req_init = bond_params[:, 3]
angle_index = pdb_cell.angle_index
angle_params = pdb_cell.angle_params
tk_init = angle_params[:, 3]
teq_init = angle_params[:, 4]
dihedral_index = pdb_cell.dihedral_params[:, [0, 1, 2, 3]]
idihedral_index = pdb_cell.idihedral_params[:, [0, 1, 2, 3]]
dihedral_index = ops.Cast()(mnp.vstack((dihedral_index, idihedral_index)), ms.int32)
dihedral_params = pdb_cell.dihedral_params
idihedral_params = pdb_cell.idihedral_params
dihedral_params = mnp.vstack((dihedral_params, idihedral_params))
pk_init = dihedral_params[:, 5]
pn_init = ops.Cast()(dihedral_params[:, 4], ms.int32)
phase_init = dihedral_params[:, 6]
vdw_params = pdb_cell.vdw_params
atomic_radius = vdw_params[:, 0]
well_depth = vdw_params[:, 1]
exclude_index = pdb_cell.excludes_index
nb14_index = pdb_cell.nb14_index
one_scee = np.array([5 / 6] * nb14_index.shape[-2])
one_scnb = np.array([.5] * nb14_index.shape[-2])
set_global_units('A', 'kcal/mol')
num_atoms = atom_type.shape[-1]
coordinates = Tensor(coordinates, ms.float32)
init_resid_new = Tensor(np.array(init_resid) - np.array(1), ms.int32)
net = nn.OneHot(depth=resid_num, axis=-1)
init_resid_onehot = net(init_resid_new)


def get_cosine_lr(lr_init, lr_max, total_steps, warmup_steps):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
            lr = float(lr_init) + lr_inc * (i + 1)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = lr_max * decayed
        lr = max(lr_init, lr) # Lower-bound lr by lr_init
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def relax_hydrogens(system, nblist, energy_terms, steps=100, grad_clip=1.0, clip_by_value=True,
                    warmup_fraction=0.2, lr_init=1e-7, lr_max=1e-4):
    """ relax hydrogens"""

    bond_energy, angle_energy, dihedral_energy, nonbond_energy, nb14_energy = energy_terms
    energy = ClassicalFF(
        bond_energy=bond_energy,
        angle_energy=angle_energy,
        dihedral_energy=dihedral_energy,
        nonbond_energy=nonbond_energy,
        nb14_energy=nb14_energy,
    )

    neighbour_list = nblist
    simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)
    total_steps = steps
    warmup_steps = int(warmup_fraction * steps)
    lr = get_cosine_lr(lr_init, lr_max, total_steps, warmup_steps)
    opt = nn.SGD(params=system.trainable_params(), learning_rate=Tensor(lr), momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False)

    onestep = ClippedRunOneStepCell(simulation_network, opt, loss_scale=1.0,
                                    include_mask=h_mask, grad_clip_value=grad_clip, clip_by_value=clip_by_value)
    md = Sponge(onestep)
    run_info = RunInfo(system,
                       get_vloss=False,
                       atom14_atom_exists=atom14_atom_exists,
                       residue_index=residue_index,
                       residx_atom14_to_atom37=residx_atom14_to_atom37,
                       aatype=aatype,
                       crd_mapping_masks=init_res_ids - 1,
                       crd_mapping_ids=crd_mapping_ids,
                       nonh_mask=nonh_mask,
                       print_interval=100,
                       )
    md.run(total_steps, callbacks=[run_info])

    if mnp.isnan(md.energy().sum()):
        return 0, None
    return system, md.energy().sum() / num_atoms


def relax_violation(system, nblist, energy_terms, steps=100, grad_clip=1.0, clip_by_value=True,
                    warmup_fraction=0.2, lr_init=1e-5, lr_max=1e-2):
    """relax violation loss, violation_mask should be used to compute harmonic_energy"""

    bond_energy, angle_energy, dihedral_energy, nonbond_energy, nb14_energy, harmonic_energy = energy_terms

    energy = ClassicalFF(
        bond_energy=bond_energy,
        angle_energy=angle_energy,
        dihedral_energy=dihedral_energy,
        nonbond_energy=nonbond_energy,
        nb14_energy=nb14_energy,
        harmonic_energy=harmonic_energy,
    )

    neighbour_list = nblist
    simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)

    total_steps = steps
    warmup_steps = int(warmup_fraction*steps)
    lr = get_cosine_lr(lr_init, lr_max, total_steps, warmup_steps)
    opt = nn.SGD(params=system.trainable_params(), learning_rate=Tensor(lr), momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False)

    onestep = ClippedRunOneStepCell(simulation_network, opt, loss_scale=1.0,
                                    include_mask=None, grad_clip_value=grad_clip, clip_by_value=clip_by_value)
    md = Sponge(onestep)
    run_info = RunInfo(system,
                       get_vloss=False,
                       atom14_atom_exists=atom14_atom_exists,
                       residue_index=residue_index,
                       residx_atom14_to_atom37=residx_atom14_to_atom37,
                       aatype=aatype,
                       crd_mapping_masks=init_res_ids - 1,
                       crd_mapping_ids=crd_mapping_ids,
                       nonh_mask=nonh_mask, # @ZhangJ. Check this, seems strange. Make CallBack less frequent.
                       print_interval=100,
                       )
    md.run(total_steps, callbacks=[run_info])

    if mnp.isnan(md.energy().sum()):
        return 0, None
    return system, md.energy().sum()/num_atoms


def main(loops, steps, clip_by_value, grad_clip, warmup_fraction, lr_init, lr_max):
    num_walkers = 1
    vio_loops = loops

    system = SystemCell(num_walkers=num_walkers, num_atoms=num_atoms, atomic_number=atomic_number, atom_name=atom_name,
                        atom_type=atom_type, resname=resname, resid=init_resid - 1, bond_index=bond_index, mass=mass,
                        coordinates=coordinates, charge=charge)
    neighbour_list = NeighbourList(system, cutoff=None, exclude_index=exclude_index)
    bond_energy = BondEnergy(bond_index, rk_init=rk_init, req_init=req_init, scale=1, pbc=False)
    angle_energy = AngleEnergy(angle_index, tk_init=tk_init, teq_init=teq_init, scale=1, pbc=False)
    dihedral_energy = DihedralEnergy(dihedral_index, pk_init=pk_init, pn_init=pn_init, phase_init=phase_init, scale=1,
                                     pbc=False)
    nonbond_energy = NonBondEnergy(num_atoms, charge=system.charge, atomic_radius=atomic_radius, well_depth=well_depth)
    nb14_energy = NB14Energy(nb14_index, nonbond_energy, one_scee=one_scee, one_scnb=one_scnb)
    beg_time = time.time()

    ### 1. Relax Hydrogens:
    # 1) Optimize Hydrogen Geometry
    energy_list = [bond_energy, angle_energy, dihedral_energy, None, nb14_energy]
    system, atomic_energy = relax_hydrogens(system, neighbour_list, energy_list, steps=steps[0],
                                            grad_clip=grad_clip[0], clip_by_value=clip_by_value[0],
                                            warmup_fraction=warmup_fraction[0], lr_init=lr_init[0], lr_max=lr_max[0])

    # 2) Optimize Hydrogen clashes
    energy_list = [bond_energy, angle_energy, dihedral_energy, nonbond_energy, nb14_energy]
    system, atomic_energy = relax_hydrogens(system, neighbour_list, energy_list, steps=steps[0],
                                            grad_clip=grad_clip[0], clip_by_value=clip_by_value[0],
                                            warmup_fraction=warmup_fraction[0], lr_init=lr_init[0], lr_max=lr_max[0])
    try:
        gen_pdb(system.coordinates.asnumpy(), atom_name, init_resname, init_resid, pdb_name=save_pdb_name)
        features = get_pdb_info(save_pdb_name)
        atom14_atom_exists_t = Tensor(features.get("atom14_gt_exists")).astype(ms.float32)
        residue_index_t = Tensor(features.get("residue_index")).astype(ms.float32)
        residx_atom14_to_atom37_t = Tensor(features.get("residx_atom14_to_atom37")).astype(ms.int32)
        atom14_positions_t = Tensor(features.get("atom14_gt_positions")).astype(ms.float32)
        aatype_t = Tensor(features.get("aatype")).astype(ms.int32)
        violations = get_structural_violations(atom14_atom_exists_t, residue_index_t, aatype_t,
                                               residx_atom14_to_atom37_t, atom14_positions_t,
                                               VIOLATION_TOLERANCE_ACTOR, CLASH_OVERLAP_TOLERANCE,
                                               LOWER_BOUND, UPPER_BOUND, ATOMTYPE_RADIUS, C_ONE_HOT, N_ONE_HOT,
                                               DISTS_MASK_I, CYS_SG_IDX)
    except AttributeError:
        violations = get_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                                               atom14_positions, VIOLATION_TOLERANCE_ACTOR, CLASH_OVERLAP_TOLERANCE,
                                               LOWER_BOUND, UPPER_BOUND, ATOMTYPE_RADIUS, C_ONE_HOT, N_ONE_HOT,
                                               DISTS_MASK_I, CYS_SG_IDX)

    violation_loss = violations[-1]
    residue_violations_mask = violations[-2]
    print(f"The violation loss value is: {violation_loss}")

    violation_atom_mask = mnp.matmul(init_resid_onehot, mnp.expand_dims(residue_violations_mask, -1))

    ### 2. Relax All Atoms with Harmonic Restraints:
    ref_crds = coordinates
    restraint_mask = nonh_mask
    for l in range(vio_loops):
        clip_by_value_ = clip_by_value[1]
        if l == 0:
            clip_by_value_ *= 0.5

        ### This term should be adapted @ZhangJ.
        ### ToDo: reduce k_coe according to vio_mask. Shape: [1,Natom,1]
        k_coe = max(10., 40.0*(1./(1.+0.2*l))) # 20.0 # 10.0 # Unit: kcal/mol/A;

        decay_factor = 1./1.5
        vio_mask = mnp.expand_dims(violation_atom_mask, 0)
        if l > 0:
            restraint_mask = restraint_mask*(1.-vio_mask) + restraint_mask*vio_mask*decay_factor

        harmonic_energy = Oscillator(ref_crds, k_coe, restraint_mask)

        energy_list = [bond_energy, angle_energy, dihedral_energy, nonbond_energy, nb14_energy, harmonic_energy]
        system, atomic_energy = relax_violation(system, neighbour_list, energy_list, steps=steps[1],
                                                grad_clip=grad_clip[1], clip_by_value=clip_by_value_,
                                                warmup_fraction=warmup_fraction[1], lr_init=lr_init[1],
                                                lr_max=lr_max[1])
        # @ZhangJ. ToDo: Add violation loss in CallBack (do not need to save pdb.)
        atomic_energy = atomic_energy.asnumpy()
        try:
            gen_pdb(system.coordinates.asnumpy(), atom_name, init_resname, init_resid, pdb_name=save_pdb_name)
            features = get_pdb_info(save_pdb_name)
            atom14_atom_exists_t = Tensor(features.get("atom14_gt_exists")).astype(ms.float32)
            residue_index_t = Tensor(features.get("residue_index")).astype(ms.float32)
            residx_atom14_to_atom37_t = Tensor(features.get("residx_atom14_to_atom37")).astype(ms.int32)
            atom14_positions_t = Tensor(features.get("atom14_gt_positions")).astype(ms.float32)
            aatype_t = Tensor(features.get("aatype")).astype(ms.int32)
            violations = get_structural_violations(atom14_atom_exists_t, residue_index_t, aatype_t,
                                                   residx_atom14_to_atom37_t, atom14_positions_t,
                                                   VIOLATION_TOLERANCE_ACTOR, CLASH_OVERLAP_TOLERANCE,
                                                   LOWER_BOUND, UPPER_BOUND, ATOMTYPE_RADIUS, C_ONE_HOT, N_ONE_HOT,
                                                   DISTS_MASK_I, CYS_SG_IDX)
            violation_loss = violations[-1]
            print(f"The violation loss value is: {violation_loss}; atomic energy is {atomic_energy}")
        except AttributeError:
            violation_loss = 0.
            atomic_energy = 0.

        opt_condition = (atomic_energy < 1.0)
        if violation_loss < 1e-8 and opt_condition: # Early Stopping.
            break

    # Finally, Report Logs:
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print(f"Run Time: {h}:{m}:{s}")


try:
    steps_t = [500, 1000]
    grad_clip_t = [2.0, 4.0]
    clip_by_value_t = [True, False]
    warmup_fraction_t = [0.2, 0.2]
    lr_init_t = [1e-5, 1e-5]
    lr_max_t = [1e-2, 5e-3]

    main(loops=5, steps=steps_t, grad_clip=grad_clip_t, clip_by_value=clip_by_value_t,
         warmup_fraction=warmup_fraction_t, lr_init=lr_init_t, lr_max=lr_max_t)
except RuntimeError as e:
    import traceback

    traceback.print_exc()
    print('MindSponge relax pipeline running failed, please try it again!')
