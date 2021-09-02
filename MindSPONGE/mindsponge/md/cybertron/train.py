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
"""train"""

import os
from shutil import copyfile
from collections import deque

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from mindspore.nn.metrics import Metric
from mindspore.train._utils import _make_directory
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore._checkparam import Validator as validator

_cur_dir = os.getcwd()

__all__ = [
    "DatasetWhitening",
    "OutputScaleShift",
    "WithForceLossCell",
    "WithLabelLossCell",
    "WithForceEvalCell",
    "WithLabelEvalCell",
    "TrainMonitor",
    "MAE",
    "MSE",
    "MLoss",
    "TransformerLR",
]


class DatasetWhitening(nn.Cell):
    """DatasetWhitening"""
    def __init__(self,
                 mol_scale=1,
                 mol_shift=0,
                 atom_scale=1,
                 atom_shift=0,
                 atom_ref=None,
                 axis=-2,
                 ):
        super().__init__()

        self.mol_scale = mol_scale
        self.mol_shift = mol_shift

        self.atom_scale = atom_scale
        self.atom_shift = atom_shift

        self.atom_ref = atom_ref
        self.axis = axis

        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(keep_dims=True)

    def construct(self, label, atoms_number, atom_types=None):
        """construct"""
        ref = 0
        if self.atom_ref is not None:
            ref = F.gather(self.atom_ref, atom_types, 0)
            ref = self.reduce_sum(ref, self.axis)

        whiten_label = (label - self.mol_shift) / self.mol_scale
        whiten_label = (whiten_label - ref - self.atom_shift *
                        atoms_number) / self.atom_scale

        return whiten_label


class OutputScaleShift(nn.Cell):
    """OutputScaleShift"""
    def __init__(self,
                 mol_scale=1,
                 mol_shift=0,
                 atom_scale=1,
                 atom_shift=0,
                 atom_ref=None,
                 axis=-2,
                 ):
        super().__init__()

        self.mol_scale = mol_scale
        self.mol_shift = mol_shift

        self.atom_scale = atom_scale
        self.atom_shift = atom_shift

        self.atom_ref = atom_ref
        self.axis = axis

        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(keep_dims=True)

    def construct(self, outputs, atoms_number, atom_types=None):
        """construct"""
        ref = 0
        if self.atom_ref is not None:
            ref = F.gather(self.atom_ref, atom_types, 0)
            ref = self.reduce_sum(ref, self.axis)

        scaled_outputs = outputs * self.atom_scale + \
            self.atom_shift * atoms_number + ref
        scaled_outputs = scaled_outputs * self.mol_scale + self.mol_shift

        return scaled_outputs


class LossWithEnergyAndForces(nn.loss.loss.LossBase):
    """LossWithEnergyAndForces"""
    def __init__(self,
                 ratio_energy=1,
                 ratio_forces=100,
                 force_aggregate='sum',
                 reduction='mean',
                 scale_dis=1,
                 ratio_normlize=True,
                 ):
        super().__init__(reduction)

        if force_aggregate not in ('mean', 'sum'):
            raise ValueError(
                f"reduction_mol method for {force_aggregate} is not supported")
        self.force_aggregate = force_aggregate

        self.scale_dis = scale_dis
        self.ratio_normlize = ratio_normlize

        self.ratio_energy = ratio_energy
        self.ratio_forces = ratio_forces

        self.norm = 1
        if self.ratio_normlize:
            self.norm = ratio_energy + ratio_forces

        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()

    def _calc_loss(self, diff):
        return diff

    def construct(
            self,
            pred_energy,
            label_energy,
            pred_forces=None,
            label_forces=None,
            atoms_number=1,
            atom_mask=None):
        """construct"""

        if pred_forces is None:
            loss = self._calc_loss(pred_energy - label_energy)
            return self.get_loss(loss)

        eloss = 0
        if self.ratio_forces > 0:
            ediff = (pred_energy - label_energy) / atoms_number
            eloss = self._calc_loss(ediff)

        floss = 0
        if self.ratio_forces > 0:
            fdiff = (pred_forces - label_forces) * self.scale_dis
            fdiff = self._calc_loss(fdiff)
            if self.force_aggregate == 'mean':
                fdiff = self.reduce_mean(fdiff, -1)
            else:
                fdiff = self.reduce_sum(fdiff, -1)

            if atom_mask is None:
                floss = self.reduce_mean(fdiff, -1)
            else:
                fdiff = fdiff * atom_mask
                floss = self.reduce_sum(fdiff, -1)
                floss = floss / atoms_number

        y = (eloss * self.ratio_energy + floss * self.ratio_forces) / self.norm

        natoms = F.cast(atoms_number, pred_energy.dtype)
        weights = natoms / self.reduce_mean(natoms)

        return self.get_loss(y, weights)


class MAELoss(LossWithEnergyAndForces):
    """MAELoss"""
    def __init__(self,
                 ratio_energy=1,
                 ratio_forces=0,
                 force_aggregate='sum',
                 reduction='mean',
                 scale_dis=1,
                 ratio_normlize=True,
                 ):
        super().__init__(
            ratio_energy=ratio_energy,
            ratio_forces=ratio_forces,
            force_aggregate=force_aggregate,
            reduction=reduction,
            scale_dis=scale_dis,
            ratio_normlize=ratio_normlize,
        )
        self.abs = P.Abs()

    def _calc_loss(self, diff):
        return self.abs(diff)


class MSELoss(LossWithEnergyAndForces):
    """MSELoss"""
    def __init__(self,
                 ratio_energy=1,
                 ratio_forces=0,
                 force_aggregate='sum',
                 reduction='mean',
                 scale_dis=1,
                 ratio_normlize=True,
                 ):
        super().__init__(
            ratio_energy=ratio_energy,
            ratio_forces=ratio_forces,
            force_aggregate=force_aggregate,
            reduction=reduction,
            scale_dis=scale_dis,
            ratio_normlize=ratio_normlize,
        )
        self.square = P.Square()

    def _calc_loss(self, diff):
        return self.square(diff)


class WithCell(nn.Cell):
    """WithCell"""
    def __init__(self, datatypes,):
        super().__init__(auto_prefix=False)

        self.fulltypes = 'RZCNnBbLlE'
        self.datatypes = datatypes

        self.r = -1  # positions
        self.z = -1  # atom_types
        self.c = -1  # pbcbox
        self.n = -1  # neighbors
        self.n_mask = -1  # neighbor_mask
        self.b = -1  # bonds
        self.b_mask = -1  # bond_mask
        self.l = -1  # far_neighbors
        self.l_mask = -1  # far_mask
        self.e = -1  # energy

    def _find_type_indexes(self, datatypes):
        """_find_type_indexes"""
        if not isinstance(datatypes, str):
            raise TypeError('Type of "datatypes" must be str')

        for datatype in datatypes:
            if self.fulltypes.count(datatype) == 0:
                raise ValueError('Unknown datatype: ' + datatype)

        for datatype in self.fulltypes:
            num = datatypes.count(datatype)
            if num > 1:
                raise ValueError(
                    'There are ' +
                    str(num) +
                    ' "' +
                    datatype +
                    '" in datatype "' +
                    datatypes +
                    '".')

        self.r = datatypes.find('R')  # positions
        self.z = datatypes.find('Z')  # atom_types
        self.c = datatypes.find('C')  # pbcbox
        self.n = datatypes.find('N')  # neighbors
        self.n_mask = datatypes.find('n')  # neighbor_mask
        self.b = datatypes.find('B')  # bonds
        self.b_mask = datatypes.find('b')  # bond_mask
        self.l = datatypes.find('L')  # far_neighbors
        self.l_mask = datatypes.find('l')  # far_mask
        self.e = datatypes.find('E')  # energy

        if self.e < 0:
            raise TypeError('The datatype "E" must be included!')

        self.keep_sum = P.ReduceSum(keep_dims=True)


class WithForceLossCell(WithCell):
    """WithForceLossCell"""
    def __init__(self,
                 datatypes,
                 backbone,
                 loss_fn,
                 do_whitening=False,
                 mol_scale=1,
                 mol_shift=0,
                 atom_scale=1,
                 atom_shift=0,
                 atom_ref=None,
                 ):
        super().__init__(datatypes=datatypes)

        self.scale = mol_scale * atom_scale
        self.do_whitening = do_whitening
        if do_whitening:
            self.whitening = DatasetWhitening(
                mol_scale=mol_scale,
                mol_shift=mol_shift,
                atom_scale=atom_scale,
                atom_shift=atom_shift,
                atom_ref=atom_ref,
            )
        else:
            self.whitening = None

        self.fulltypes = 'RZCNnBbLlFE'
        self._find_type_indexes(datatypes)
        self.f = datatypes.find('F')  # force

        if self.f < 0:
            raise TypeError(
                'The datatype "F" must be included in WithForceLossCell!')

        self._backbone = backbone
        self._loss_fn = loss_fn

        self.atom_types = self._backbone.atom_types

        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        """construct"""
        inputs = inputs + (None,)

        positions = inputs[self.r]
        atom_types = inputs[self.z]
        pbcbox = inputs[self.c]
        neighbors = inputs[self.n]
        neighbor_mask = inputs[self.n_mask]
        bonds = inputs[self.b]
        bond_mask = inputs[self.b_mask]
        far_neighbors = inputs[self.l]
        far_mask = inputs[self.l_mask]

        energy = inputs[self.e]
        out = self._backbone(
            positions,
            atom_types,
            pbcbox,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        forces = inputs[self.f]
        fout = -1 * self.grad_op(self._backbone)(
            positions,
            atom_types,
            pbcbox,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        if atom_types is None:
            atom_types = self.atom_types

        atoms_number = F.cast(atom_types > 0, out.dtype)
        atoms_number = self.keep_sum(atoms_number, -1)

        if self.do_whitening:
            energy = self.whitening(energy, atoms_number, atom_types)
            forces /= self.scale

        if atom_types is None:
            return self._loss_fn(out, energy, fout, forces)
        atom_mask = atom_types > 0
        return self._loss_fn(
            out,
            energy,
            fout,
            forces,
            atoms_number,
            atom_mask)

    @property
    def backbone_network(self):
        return self._backbone


class WithLabelLossCell(WithCell):
    """WithLabelLossCell"""
    def __init__(self,
                 datatypes,
                 backbone,
                 loss_fn,
                 do_whitening=False,
                 mol_scale=1,
                 mol_shift=0,
                 atom_scale=1,
                 atom_shift=0,
                 atom_ref=None,
                 # with_penalty=False,
                 ):
        super().__init__(datatypes=datatypes)
        self._backbone = backbone
        self._loss_fn = loss_fn
        # self.with_penalty = with_penalty

        self.atom_types = self._backbone.atom_types

        self.do_whitening = do_whitening
        if do_whitening:
            self.whitening = DatasetWhitening(
                mol_scale=mol_scale,
                mol_shift=mol_shift,
                atom_scale=atom_scale,
                atom_shift=atom_shift,
                atom_ref=atom_ref,
            )
        else:
            self.whitening = None

        self._find_type_indexes(datatypes)

    def construct(self, *inputs):
        """construct"""

        inputs = inputs + (None,)

        positions = inputs[self.r]
        atom_types = inputs[self.z]
        pbcbox = inputs[self.c]
        neighbors = inputs[self.n]
        neighbor_mask = inputs[self.n_mask]
        bonds = inputs[self.b]
        bond_mask = inputs[self.b_mask]
        far_neighbors = inputs[self.l]
        far_mask = inputs[self.l_mask]

        out = self._backbone(
            positions,
            atom_types,
            pbcbox,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        label = inputs[self.e]

        if atom_types is None:
            atom_types = self.atom_types

        atoms_number = F.cast(atom_types > 0, out.dtype)
        atoms_number = self.keep_sum(atoms_number, -1)

        if self.do_whitening:
            label = self.whitening(label, atoms_number, atom_types)

        return self._loss_fn(out, label)


class WithForceEvalCell(WithCell):
    """WithForceEvalCell"""
    def __init__(self,
                 datatypes,
                 network,
                 loss_fn=None,
                 add_cast_fp32=False,
                 do_whitening=False,
                 mol_scale=1,
                 mol_shift=0,
                 atom_scale=1,
                 atom_shift=0,
                 atom_ref=None,
                 ):
        super().__init__(datatypes)

        self.scale = mol_scale * atom_scale
        self.do_whitening = do_whitening
        self.scaleshift = None
        self.whitening = None
        if do_whitening:
            self.scaleshift = OutputScaleShift(
                mol_scale=mol_scale,
                mol_shift=mol_shift,
                atom_scale=atom_scale,
                atom_shift=atom_shift,
                atom_ref=atom_ref,
            )
            if loss_fn is not None:
                self.whitening = DatasetWhitening(
                    mol_scale=mol_scale,
                    mol_shift=mol_shift,
                    atom_scale=atom_scale,
                    atom_shift=atom_shift,
                    atom_ref=atom_ref,
                )

        self.fulltypes = 'RZCNnBbLlFE'
        self._find_type_indexes(datatypes)
        self.f = datatypes.find('F')  # force

        if self.f < 0:
            raise TypeError(
                'The datatype "F" must be included in WithForceEvalCell!')

        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = add_cast_fp32

        self.atom_types = self._network.atom_types

        self.reduce_sum = P.ReduceSum()

        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        """construct"""
        inputs = inputs + (None,)

        positions = inputs[self.r]
        atom_types = inputs[self.z]
        pbcbox = inputs[self.c]
        neighbors = inputs[self.n]
        neighbor_mask = inputs[self.n_mask]
        bonds = inputs[self.b]
        bond_mask = inputs[self.b_mask]
        far_neighbors = inputs[self.l]
        far_mask = inputs[self.l_mask]

        outputs = self._network(
            positions,
            atom_types,
            pbcbox,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        foutputs = -1 * self.grad_op(self._network)(
            positions,
            atom_types,
            pbcbox,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        forces = inputs[self.f]
        energy = inputs[self.e]

        if self.add_cast_fp32:
            forces = F.mixed_precision_cast(ms.float32, forces)
            energy = F.mixed_precision_cast(ms.float32, energy)
            outputs = F.cast(outputs, ms.float32)

        if atom_types is None:
            atom_types = self.atom_types

        atoms_number = F.cast(atom_types > 0, outputs.dtype)
        atoms_number = self.keep_sum(atoms_number, -1)

        loss = 0
        if self._loss_fn is not None:
            energy_t = energy
            forces_t = forces
            if self.do_whitening:
                energy_t = self.whitening(energy_t, atoms_number, atom_types)
                forces_t /= self.scale

            atom_mask = atom_types > 0
            loss = self._loss_fn(
                outputs,
                energy_t,
                foutputs,
                forces_t,
                atoms_number,
                atom_mask)

        if self.do_whitening:
            outputs = self.scaleshift(outputs, atoms_number, atom_types)
            foutputs *= self.scale

        return loss, outputs, energy, foutputs, forces, atoms_number


class WithLabelEvalCell(WithCell):
    """WithLabelEvalCell"""
    def __init__(self,
                 datatypes,
                 network,
                 loss_fn=None,
                 add_cast_fp32=False,
                 do_whitening=False,
                 mol_scale=1,
                 mol_shift=0,
                 atom_scale=1,
                 atom_shift=0,
                 atom_ref=None,
                 ):
        super().__init__(datatypes=datatypes)
        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = add_cast_fp32
        self.reducesum = P.ReduceSum(keep_dims=True)

        self.atom_types = self._network.atom_types

        self.do_whitening = do_whitening
        self.scaleshift = None
        self.whitening = None
        if do_whitening:
            self.scaleshift = OutputScaleShift(
                mol_scale=mol_scale,
                mol_shift=mol_shift,
                atom_scale=atom_scale,
                atom_shift=atom_shift,
                atom_ref=atom_ref,
            )
            if loss_fn is not None:
                self.whitening = DatasetWhitening(
                    mol_scale=mol_scale,
                    mol_shift=mol_shift,
                    atom_scale=atom_scale,
                    atom_shift=atom_shift,
                    atom_ref=atom_ref,
                )

        self._find_type_indexes(datatypes)

    def construct(self, *inputs):
        """construct"""
        inputs = inputs + (None,)

        positions = inputs[self.r]
        atom_types = inputs[self.z]
        pbcbox = inputs[self.c]
        neighbors = inputs[self.n]
        neighbor_mask = inputs[self.n_mask]
        bonds = inputs[self.b]
        bond_mask = inputs[self.b_mask]
        far_neighbors = inputs[self.l]
        far_mask = inputs[self.l_mask]

        outputs = self._network(
            positions,
            atom_types,
            pbcbox,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        label = inputs[self.e]
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(ms.float32, label)
            outputs = F.cast(outputs, ms.float32)

        if atom_types is None:
            atom_types = self.atom_types

        atoms_number = F.cast(atom_types > 0, outputs.dtype)
        atoms_number = self.keep_sum(atoms_number, -1)

        loss = 0
        if self._loss_fn is not None:
            label_t = label
            if self.do_whitening:
                label_t = self.whitening(label, atoms_number, atom_types)
            loss = self._loss_fn(outputs, label_t)

        if self.do_whitening:
            outputs = self.scaleshift(outputs, atoms_number, atom_types)

        return loss, outputs, label, atoms_number


class TrainMonitor(Callback):
    """TrainMonitor"""
    def __init__(
            self,
            model,
            name,
            directory=None,
            per_epoch=1,
            per_step=0,
            avg_steps=0,
            eval_dataset=None,
            best_ckpt_metrics=None):
        super().__init__()
        if not isinstance(per_epoch, int) or per_epoch < 0:
            raise ValueError("per_epoch must be int and >= 0.")
        if not isinstance(per_step, int) or per_step < 0:
            raise ValueError("per_step must be int and >= 0.")

        self.avg_steps = avg_steps
        self.loss_record = 0
        self.train_num = 0
        if avg_steps > 0:
            self.train_num = deque(maxlen=avg_steps)
            self.loss_record = deque(maxlen=avg_steps)

        if per_epoch * per_step != 0:
            if per_epoch == 1:
                per_epoch = 0
            else:
                raise ValueError(
                    "per_epoch and per_step cannot larger than 0 at same time.")
        self.model = model
        self._per_epoch = per_epoch
        self._per_step = per_step
        self.eval_dataset = eval_dataset

        if directory is not None:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir

        self._filename = name + '-info.data'
        self._ckptfile = name + '-best'
        self._ckptdata = name + '-cpkt.data'

        self.num_ckpt = 1
        self.best_value = 5e4
        self.best_ckpt_metrics = best_ckpt_metrics

        self.last_loss = 0
        self.record = []

        self.output_title = True
        filename = os.path.join(self._directory, self._filename)
        if os.path.exists(filename):
            with open(filename, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    os.remove(filename)

    def _write_cpkt_file(self, filename, info, network):
        ckptfile = os.path.join(self._directory, filename + '.ckpt')
        ckptbck = os.path.join(self._directory, filename + '.bck.ckpt')
        ckptdata = os.path.join(self._directory, self._ckptdata)

        if os.path.exists(ckptfile):
            os.rename(ckptfile, ckptbck)
        save_checkpoint(network, ckptfile)
        with open(ckptdata, "a") as f:
            f.write(info + os.linesep)

    def _output_data(self, cb_params):
        """_output_data"""
        cur_epoch = cb_params.cur_epoch_num

        opt = cb_params.optimizer
        if opt is None:
            opt = cb_params.train_network.optimizer

        if opt.dynamic_lr:
            step = opt.global_step
            if not isinstance(step, int):
                step = step.asnumpy()[0]
        else:
            step = cb_params.cur_step_num

        if self.avg_steps > 0:
            mov_avg = sum(self.loss_record) / sum(self.train_num)
        else:
            mov_avg = self.loss_record / self.train_num

        title = "#! FIELDS step"
        info = 'Epoch: ' + str(cur_epoch) + ', Step: ' + str(step)
        outdata = '{:>10d}'.format(step)

        lr = opt.learning_rate
        if opt.dynamic_lr:
            step = F.cast(step, ms.int32)
            if opt.is_group_lr:
                lr = ()
                for learning_rate in opt.learning_rate:
                    current_dynamic_lr = learning_rate(step - 1)
                    lr += (current_dynamic_lr,)
            else:
                lr = opt.learning_rate(step - 1)
        lr = lr.asnumpy()

        title += ' learning_rate'
        info += ', Learning_rate: ' + str(lr)
        outdata += '{:>15e}'.format(lr)

        title += " last_loss avg_loss"
        info += ', Last_Loss: ' + \
            str(self.last_loss) + ', Avg_loss: ' + str(mov_avg)
        outdata += '{:>15e}'.format(self.last_loss) + '{:>15e}'.format(mov_avg)

        _make_directory(self._directory)

        if self.eval_dataset is not None:
            eval_metrics = self.model.eval(
                self.eval_dataset, dataset_sink_mode=False)
            for k, v in eval_metrics.items():
                info += ', '
                info += k
                info += ': '
                info += str(v)

                if isinstance(v, np.ndarray) and v.size > 1:
                    for i in range(v.size):
                        title += (' ' + k + str(i))
                        outdata += '{:>15e}'.format(v[i])
                else:
                    title += (' ' + k)
                    outdata += '{:>15e}'.format(v)
            if self.best_ckpt_metrics in eval_metrics.keys():
                self.eval_dataset_process(eval_metrics, info, cb_params)


        print(info, flush=True)
        filename = os.path.join(self._directory, self._filename)
        if self.output_title:
            with open(filename, "a") as f:
                f.write(title + os.linesep)
            self.output_title = False
        with open(filename, "a") as f:
            f.write(outdata + os.linesep)

    def eval_dataset_process(self, eval_metrics, info, cb_params):
        """eval_dataset_process"""
        vnow = eval_metrics[self.best_ckpt_metrics]
        if isinstance(vnow, np.ndarray) and len(vnow) > 1:
            output_ckpt = vnow < self.best_value
            num_best = np.count_nonzero(output_ckpt)
            if num_best > 0:
                self._write_cpkt_file(
                    self._ckptfile, info, cb_params.train_network)
                source_ckpt = os.path.join(
                    self._directory, self._ckptfile + '.ckpt')
                for i in range(len(vnow)):
                    if output_ckpt[i]:
                        dest_ckpt = os.path.join(
                            self._directory, self._ckptfile + '-' + str(i) + '.ckpt')
                        bck_ckpt = os.path.join(
                            self._directory, self._ckptfile + '-' + str(i) + '.ckpt.bck')
                        if os.path.exists(dest_ckpt):
                            os.rename(dest_ckpt, bck_ckpt)
                        copyfile(source_ckpt, dest_ckpt)
                self.best_value = np.minimum(vnow, self.best_value)
        else:
            if vnow < self.best_value:
                self._write_cpkt_file(
                    self._ckptfile, info, cb_params.train_network)
                self.best_value = vnow

    def step_end(self, run_context):
        """step_end"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        nbatch = len(cb_params.train_dataset_element[0])
        batch_loss = loss * nbatch

        self.last_loss = loss
        if self.avg_steps > 0:
            self.loss_record.append(batch_loss)
            self.train_num.append(nbatch)
        else:
            self.loss_record += batch_loss
            self.train_num += nbatch

        if self._per_step > 0 and cb_params.cur_step_num % self._per_step == 0:
            self._output_data(cb_params)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if self._per_epoch > 0 and cur_epoch % self._per_epoch == 0:
            self._output_data(cb_params)


class MaxError(Metric):
    """MaxError"""
    def __init__(self, indexes=None, reduce_all_dims=True):
        super().__init__()
        self.clear()
        self._indexes = [1, 2] if not indexes else indexes
        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

    def clear(self):
        self._max_error = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        diff = y.reshape(y_pred.shape) - y_pred
        max_error = diff.max() - diff.min()
        if max_error > self._max_error:
            self._max_error = max_error

    def eval(self):
        return self._max_error


class Error(Metric):
    """Error"""
    def __init__(self,
                 indexes=None,
                 reduce_all_dims=True,
                 averaged_by_atoms=False,
                 atom_aggregate='mean',
                 ):
        super().__init__()
        self.clear()
        self._indexes = [1, 2] if not indexes else indexes
        self.read_atoms_number = False
        if len(self._indexes) > 2:
            self.read_atoms_number = True

        self.reduce_all_dims = reduce_all_dims

        if atom_aggregate.lower() not in ('mean', 'sum'):
            raise ValueError(
                'aggregate_by_atoms method must be "mean" or "sum"')
        self.atom_aggregate = atom_aggregate.lower()

        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

        if averaged_by_atoms and not self.read_atoms_number:
            raise ValueError(
                'When to use averaged_by_atoms, the index of atom number must be set at "indexes".')

        self.averaged_by_atoms = averaged_by_atoms

        self._error_sum = 0
        self._samples_num = 0

    def clear(self):
        self._error_sum = 0
        self._samples_num = 0

    def _calc_error(self, y, y_pred):
        return y.reshape(y_pred.shape) - y_pred

    def update(self, *inputs):
        """update"""
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])

        error = self._calc_error(y, y_pred)
        if len(error.shape) > 2:
            axis = tuple(range(2, len(error.shape)))
            if self.atom_aggregate == 'mean':
                error = np.mean(error, axis=axis)
            else:
                error = np.sum(error, axis=axis)

        tot = y.shape[0]
        if self.read_atoms_number:
            natoms = self._convert_data(inputs[self._indexes[2]])
            if self.averaged_by_atoms:
                error /= natoms
            elif self.reduce_all_dims:
                tot = np.sum(natoms)
                if natoms.shape[0] != y.shape[0]:
                    tot *= y.shape[0]
        elif self.reduce_all_dims:
            tot = error.size

        self._error_sum += np.sum(error, axis=self.axis)
        self._samples_num += tot

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._error_sum / self._samples_num

# mean absolute error


class MAE(Error):
    """MAE"""
    def __init__(self,
                 indexes=None,
                 reduce_all_dims=True,
                 averaged_by_atoms=False,
                 atom_aggregate='mean',
                 ):
        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y, y_pred):
        return np.abs(y.reshape(y_pred.shape) - y_pred)

# mean square error


class MSE(Error):
    """MSE"""
    def __init__(self,
                 indexes=None,
                 reduce_all_dims=True,
                 averaged_by_atoms=False,
                 atom_aggregate='mean',
                 ):
        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y, y_pred):
        return np.square(y.reshape(y_pred.shape) - y_pred)

# mean norm error


class MNE(Error):
    """MNE"""
    def __init__(self,
                 indexes=None,
                 reduce_all_dims=True,
                 averaged_by_atoms=False,
                 atom_aggregate='mean',
                 ):
        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y, y_pred):
        diff = y.reshape(y_pred.shape) - y_pred
        return np.linalg.norm(diff, axis=-1)

# root mean square error


class RMSE(Error):
    """RMSE"""
    def __init__(self,
                 indexes=None,
                 reduce_all_dims=True,
                 averaged_by_atoms=False,
                 atom_aggregate='mean',
                 ):
        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y, y_pred):
        return np.square(y.reshape(y_pred.shape) - y_pred)

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return np.sqrt(self._error_sum / self._samples_num)


class MLoss(Metric):
    """MLoss"""
    def __init__(self, index=0):
        super().__init__()
        self.clear()
        self._index = index

    def clear(self):
        self._sum_loss = 0
        self._total_num = 0

    def update(self, *inputs):
        """update"""

        loss = self._convert_data(inputs[self._index])

        if loss.ndim == 0:
            loss = loss.reshape(1)

        if loss.ndim != 1:
            raise ValueError(
                "Dimensions of loss must be 1, but got {}".format(
                    loss.ndim))

        loss = loss.mean(-1)
        self._sum_loss += loss
        self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Total number can not be 0.')
        return self._sum_loss / self._total_num


class TransformerLR(LearningRateSchedule):
    """TransformerLR"""
    def __init__(self, learning_rate=1.0, warmup_steps=4000, dimension=1):
        super().__init__()
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float.")
        validator.check_non_negative_float(
            learning_rate, "learning_rate", self.cls_name)
        validator.check_positive_int(
            warmup_steps, 'warmup_steps', self.cls_name)

        self.learning_rate = learning_rate

        self.pow = P.Pow()
        self.warmup_scale = self.pow(F.cast(warmup_steps, ms.float32), -1.5)
        self.dim_scale = self.pow(F.cast(dimension, ms.float32), -0.5)

        self.min = P.Minimum()

    def construct(self, global_step):
        step_num = F.cast(global_step, ms.float32)
        lr_percent = self.dim_scale * \
            self.min(self.pow(step_num, -0.5), step_num * self.warmup_scale)
        return self.learning_rate * lr_percent
