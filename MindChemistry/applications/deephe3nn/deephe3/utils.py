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
"""
utils
"""
import os
import shutil
import random
import numpy as np
from mindspore.train import ReduceLROnPlateau
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore import ops
from mindchemistry.e3.o3.irreps import Irreps, Irrep

from scipy.optimize import brentq
from scipy import special as sp
import sympy as sym


def set_random_seed(seed):
    """
    set random seed of network
    """
    np.random.seed(seed)
    random.seed(seed)


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    """
    check if tp_path exists
    """
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

def jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    two_r = 2 * r
    if two_r != 0:
        result = np.sqrt(np.pi / two_r) * sp.jv(n + 0.5, r)
    else:
        raise ValueError
    return result


def jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols('x')

    if x != 0:
        f = [sym.sin(x) / x]
        a = sym.sin(x) / x
    else:
        raise ValueError
    for i in range(1, n):
        if x != 0:
            b = sym.diff(a, x) / x
        else:
            raise ValueError
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """

    zeros = jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = np.divide(1, np.array(normalizer_tmp) ** 0.5)
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


class GaussianBasis(nn.Cell):
    """
    GaussianBasis class
    """

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super(GaussianBasis, self).__init__()
        # compute offset and width of Gaussian functions
        start = ms.Tensor(start, ms.float32)
        stop = ms.Tensor(stop, ms.float32)

        offset = msnp.linspace(start, stop, n_gaussians)
        widths = ms.Tensor((offset[1] - offset[0]) * ms.numpy.ones_like(offset))  # FloatTensor
        if trainable:
            self.width = ms.Parameter(widths)
            self.offsets = ms.Parameter(offset)
        else:
            self.width = ms.Parameter(widths, name="width", requires_grad=False)
            self.offsets = ms.Parameter(offset, name="offsets", requires_grad=False)

        self.centered = centered

    def construct(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        if not self.centered:
            # compute width of Gaussian functions (using an overlap of 1 STDDEV)
            coeff = ops.div(-0.5, ops.pow(self.width, 2))

            # Use advanced indexing to compute the individual components
            diff = ops.unsqueeze(distances, -1) - self.offsets
        else:
            # if Gaussian functions are centered, use offsets to compute widths
            coeff = ops.div(-0.5, ops.pow(self.width, 2))
            # if Gaussian functions are centered, no offset is subtracted
            diff = ops.unsqueeze(distances, -1)
        gauss = ops.exp(coeff * ops.pow(diff, 2))
        return gauss


class MaskMSELoss(nn.Cell):
    """
    Masked MSELoss class
    """

    def __init__(self) -> None:
        pass
    def construct(self, inputs: ms.Tensor, target: ms.Tensor, mask: ms.Tensor) -> ms.Tensor:
        """
        MaskMSELoss class construct process
        """
        mse = ops.pow(ops.abs(inputs - target), 2)
        mse = ops.masked_select(mse, mask).mean()

        return mse


class MaskMAELoss(nn.Cell):
    """
    Masked MSELoss class
    """

    def __init__(self) -> None:
        pass
    def construct(self, inputs: ms.Tensor, target: ms.Tensor, mask: ms.Tensor) -> ms.Tensor:
        """
        Masked MSELoss class construct process
        """
        mae = ops.abs(inputs - target)
        mae = ops.masked_select(mae, mask).mean()

        return mae


class LossRecord:
    """
    LossRecord class
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        LossRecord class reset all the parameter
        """
        self.last_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        LossRecord class update all the parameter
        """
        self.last_val = val
        self.sum += val * num
        self.count += num
        count_div = self.count
        if count_div != 0:
            self.avg = self.sum / count_div
        else:
            raise ValueError


def convert2numpyt(original_dtype):
    """
    convert the dtype to numpy dtype
    """
    if original_dtype == ms.float32:
        numpy_dtype = np.float32
    elif original_dtype == ms.float64:
        numpy_dtype = np.float64
    elif original_dtype == np.float32:
        numpy_dtype = np.float32
    elif original_dtype == np.float64:
        numpy_dtype = np.float64
    else:
        raise NotImplementedError(f'Unsupported original dtype: {original_dtype}')
    return numpy_dtype


def flt2cplx(flt_dtype):
    """
    convert float to complex
    """
    if flt_dtype == ms.float32:
        cplx_dtype = np.complex64
    elif flt_dtype == ms.float64:
        cplx_dtype = np.complex128
    elif flt_dtype == np.float32:
        cplx_dtype = np.complex64
    elif flt_dtype == np.float64:
        cplx_dtype = np.complex128
    else:
        raise NotImplementedError(f'Unsupported float dtype: {flt_dtype}')
    return cplx_dtype


class SlipSlopLR:
    """
    SlipSlop learning rate scheduler class
    """

    def __init__(self, optimizer, start=1400, interval=200, decay_rate=0.5) -> None:
        self.optimizer = optimizer

        self.start = start
        self.interval = interval
        self.decay_rate = decay_rate

        self.next_epoch = 0
        self.last_decayed = None
        self.curr_val_loss = 0

    def step(self, val_loss=None):
        """
        SlipSlop learning rate scheduler step
        """
        self.curr_val_loss = val_loss
        epoch = self.next_epoch
        self.next_epoch += 1

        next_decay = -1
        if self.last_decayed is not None:
            next_decay = self.last_decayed + self.interval

        if epoch in (self.start, next_decay):
            self.decay()
            self.last_decayed = epoch

    def decay(self):
        """
        SlipSlop learning rate scheduler decay
        """
        num_lr = 0
        last_lr = self.optimizer.learning_rate.value()
        self.optimizer.learning_rate.set_data(last_lr * self.decay_rate)
        new_lr = self.optimizer.learning_rate.value()
        print(f'Learning rate {num_lr} is decayed from {last_lr} to {new_lr}.')

    def state_dict(self):
        """
        SlipSlop learning rate scheduler get state_dict
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """
        SlipSlop learning rate scheduler load state_dict
        """
        self.__dict__.update(state_dict)


class RevertDecayLR:
    """
    RevertDecayLR learning rate scheduler class
    """

    def __init__(self,
                 model,
                 optimizer,
                 save_model_dir,
                 decay_patience=20,
                 decay_rate=0.8,
                 scheduler_type=0,
                 scheduler_params=None):
        self.model = model
        self.optimizer = optimizer

        os.makedirs(save_model_dir, exist_ok=True)
        self.save_model_dir = save_model_dir

        self.next_epoch = 0
        self.best_epoch = 0
        self.best_loss = 1e10

        self.decay_patience = decay_patience
        self.decay_rate = decay_rate

        self.bad_epochs = 0

        self.scheduler = None
        self.alpha = None
        self.loss_smoothed = None

        self.scheduler_type = scheduler_type
        if scheduler_type == 1:
            self.alpha = scheduler_params.pop('alpha', 0.1)
            self.scheduler = ReduceLROnPlateau(**scheduler_params)
        elif scheduler_type == 2:
            self.scheduler = SlipSlopLR(optimizer=optimizer, **scheduler_params)
        elif scheduler_type == 0:
            pass
        else:
            raise ValueError(f'Unknown scheduler type: {scheduler_type}')

    def step(self, val_loss):
        """
        RevertDecayLR learning rate scheduler step
        """
        epoch = self.next_epoch
        self.next_epoch += 1

        if val_loss > 2 * self.best_loss:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0
        if self.bad_epochs >= self.decay_patience:
            self.revert()
            self.decay()

        if self.scheduler_type == 1:
            if self.loss_smoothed is None:
                self.loss_smoothed = val_loss
            else:
                self.loss_smoothed = self.alpha * val_loss + (1.0 - self.alpha) * self.loss_smoothed
            self.scheduler.step(self.loss_smoothed)
        elif self.scheduler_type == 2:
            self.scheduler.step()

        # = check is best =
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss
            self.best_epoch = epoch

        # = save model =
        save_complete = False
        while not save_complete:
            try:
                self.save_model(epoch, val_loss, is_best=is_best)
                save_complete = True
            except KeyboardInterrupt:
                print('Interrupting while saving model might cause the saved model to be deprecated')

    def revert(self):
        """
        RevertDecayLR learning rate scheduler revert
        """
        param_dict = ms.load_checkpoint(os.path.join(self.save_model_dir, 'best_model.ckpt'))
        _, _ = ms.load_param_into_net(self.model, param_dict)

    def decay(self):
        """
        RevertDecayLR learning rate scheduler revert
        """
        last_lr = self.optimizer.learning_rate.value()
        self.optimizer.learning_rate.set_data(last_lr * self.decay_rate)

    def save_model(self, epoch, val_loss, is_best=False, **kwargs):
        """
        RevertDecayLR learning rate scheduler save model process
        """
        model_ckpt = "model.ckpt"
        state = {
            'epoch': str(epoch),
            'val_loss': str(val_loss),
        }
        state.update(kwargs)
        ms.save_checkpoint(self.model, os.path.join(self.save_model_dir, model_ckpt), append_dict=state)
        if is_best:
            shutil.copyfile(os.path.join(self.save_model_dir, model_ckpt),
                            os.path.join(self.save_model_dir, 'best_model.ckpt'))

def process_targets(orbital_types, index_to_z, targets):
    """
    process the targets
    """
    z_to_index = np.full((100,), -1, dtype=np.int64)
    z_to_index[index_to_z] = np.arange(len(index_to_z))

    orbital_types = list(map(lambda x: np.array(x, dtype=np.int32), orbital_types))
    orbital_types_cumsum = list(
        map(lambda x: np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(2 * x + 1)]), orbital_types))

    # = process the orbital indices into block slices =
    equivariant_blocks, out_js_list = [], []
    out_slices = [0]
    for target in targets:
        out_js = None
        equivariant_block = dict()
        for n_m_str, block_indices in target.items():
            i, j = map(lambda x: z_to_index[int(x)], n_m_str.split())
            block_slice = [
                orbital_types_cumsum[i][block_indices[0]], orbital_types_cumsum[i][block_indices[0] + 1],
                orbital_types_cumsum[j][block_indices[1]], orbital_types_cumsum[j][block_indices[1] + 1]
            ]
            equivariant_block.update({n_m_str: block_slice})
            if out_js is None:
                out_js = (orbital_types[i][block_indices[0]], orbital_types[j][block_indices[1]])

        equivariant_blocks.append(equivariant_block)
        out_js_list.append(tuple(map(int, out_js)))
        out_slices.append(out_slices[-1] + (2 * out_js[0] + 1) * (2 * out_js[1] + 1))

    return equivariant_blocks, out_js_list, out_slices


def irreps_from_l1l2(l1, l2, mul, spinful, no_parity=False):
    r'''
    non-spinful example: l1=1, l2=2 (1x2) ->
    required_irreps_full=1+2+3, required_irreps=1+2+3, required_irreps_x1=None
    spinful example: l1=1, l2=2 (1x0.5)x(2x0.5) ->
    required_irreps_full = 1+2+3 + 0+1+2 + 1+2+3 + 2+3+4
    required_irreps = (1+2+3)x0 = 1+2+3
    required_irreps_x1 = (1+2+3)x1 = [0+1+2, 1+2+3, 2+3+4]
    notice that required_irreps_x1 is a list of Irreps
    '''
    p = 1
    if not no_parity:
        p = (-1) ** (l1 + l2)
    required_ls = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = Irreps([(mul, (l, p)) for l in required_ls])
    required_irreps_full = required_irreps
    required_irreps_x1 = None
    if spinful:
        required_irreps_x1 = []
        for _, ir in required_irreps:
            required_ls_irx1 = range(abs(ir.l - 1), ir.l + 1 + 1)
            irx1 = Irreps([(mul, (l, p)) for l in required_ls_irx1])
            required_irreps_x1.append(irx1)
            required_irreps_full += irx1
    return required_irreps_full, required_irreps, required_irreps_x1


def get_net_out_irreps(no_parity, spinful, required_block_type, il_list, hoppings_list):
    """
    helper function to get net_out_irreps
    """
    hoppings_list_mask = [False
                          for _ in range(len(hoppings_list))]  # if that hopping is already included, then it is True
    targets = []
    net_out_irreps_list = []
    net_out_irreps = Irreps(None)
    length = len(hoppings_list)
    for hopping1_index in range(length):
        target = {}
        if not hoppings_list_mask[hopping1_index]:
            is_diagonal = il_list[hopping1_index][0:2] == il_list[hopping1_index][2:4]
            hoppings_list_mask[hopping1_index] = True
            if is_diagonal and required_block_type == 'o':
                continue
            if not is_diagonal and required_block_type == 'd':
                continue
            target.update(hoppings_list[hopping1_index])
            length2 = len(hoppings_list)
            for hopping2_index in range(length2):
                if not hoppings_list_mask[hopping2_index]:
                    if il_list[hopping1_index] == il_list[hopping2_index]:
                        target.update(hoppings_list[hopping2_index])
                        hoppings_list_mask[hopping2_index] = True
            targets.append(target)

            l1, l2 = il_list[hopping1_index][0], il_list[hopping1_index][2]
            irreps_new = irreps_from_l1l2(l1, l2, 1, spinful, no_parity=no_parity)[0]
            net_out_irreps_list.append(irreps_new)
            net_out_irreps = net_out_irreps + irreps_new
    return net_out_irreps, targets


def orbital_analysis(atom_orbitals,
                     required_block_type,
                     spinful,
                     targets=None,
                     element_pairs=None,
                     no_parity=False):
    r'''
    example of atom_orbitals: {'42': [0, 0, 0, 1, 1, 2, 2], '16': [0, 0, 1, 1, 2]}
    required_block_type: 's' - specify; 'a' - all; 'o' - off-diagonal; 'd' - diagonal;
    '''

    if required_block_type == 's':
        net_out_irreps = Irreps(None)
        for target in targets:
            l1, l2 = None, None
            for n_m_str, block_indices in target.items():
                atom1, atom2 = n_m_str.split()
                if l1 is None and l2 is None:
                    l1 = atom_orbitals[atom1][block_indices[0]]
                    l2 = atom_orbitals[atom2][block_indices[1]]
                    net_out_irreps += irreps_from_l1l2(l1, l2, 1, spinful, no_parity=no_parity)[0]

    else:
        hoppings_list = []  # [{'42 16': [4, 3]}, ...]
        for atom1, orbitals1 in atom_orbitals.items():
            for atom2, orbitals2 in atom_orbitals.items():
                hopping_key = atom1 + ' ' + atom2
                if element_pairs:
                    if hopping_key not in element_pairs:
                        continue
                for orbital1 in range(len(orbitals1)):
                    for orbital2 in range(len(orbitals2)):
                        hopping_orbital = [orbital1, orbital2]
                        hoppings_list.append({hopping_key: hopping_orbital})

        il_list = []  # [[1, 1, 2, 0], ...] this means the hopping is from 1st l=1 orbital to 0th l=2 orbital.
        for hopping in hoppings_list:
            for n_m_str, block in hopping.items():
                atom1, atom2 = n_m_str.split()
                l1 = atom_orbitals[atom1][block[0]]
                l2 = atom_orbitals[atom2][block[1]]
                il1 = block[0] - atom_orbitals[atom1].index(l1)
                il2 = block[1] - atom_orbitals[atom2].index(l2)
            il_list.append([l1, il1, l2, il2])

        net_out_irreps, targets = get_net_out_irreps(no_parity, spinful, required_block_type, il_list, hoppings_list)

    if spinful:
        net_out_irreps = net_out_irreps + net_out_irreps

    return targets, net_out_irreps, net_out_irreps.sort()[0].simplify()
