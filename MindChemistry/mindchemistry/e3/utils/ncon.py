# Copyright 2022 Huawei Technologies Co., Ltd
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
from copy import deepcopy

import numpy as np

from mindspore import ops, nn, vmap
from mindspore.numpy import tensordot, trace


def nest_vmap(fn, in_list, out_list, pt):
    """nest vmap function"""
    if pt == len(in_list) - 1:
        return vmap(fn, in_list[pt], out_list[pt])
    else:
        return vmap(nest_vmap(fn, in_list, out_list, pt + 1), in_list[pt], out_list[pt])


def _create_order(con_list):
    """ Identify all unique, positive indices and return them sorted. """
    flat_con = np.concatenate(con_list)
    return np.unique(flat_con[flat_con > 0]).tolist()


def _single_trace(con, leg):
    leg = np.where(np.array(con) == leg)[0]
    con = np.delete(con, leg).tolist()
    return con, leg.tolist()


def _find_sum(con_list):
    _flat = []
    for item in con_list:
        _flat += item
    legs = []
    for leg in np.unique(_flat):
        if leg < 0:
            continue
        if np.sum(np.array(_flat) == leg) == 1:
            legs.append(leg)
    return legs


def _find_trace(con_list):
    legs_list = []
    for i in range(len(con_list)):
        tr_num = len(con_list[i]) - len(np.unique(con_list[i]))
        legs = []
        if tr_num:
            for leg in np.unique(con_list[i]):
                if sum(con_list[i] == leg) > 1 and leg > 0:
                    leg = np.where(con_list[i] == leg)[0].tolist()
                    legs.append(leg)
                    con_list[i] = np.delete(con_list[i], leg).tolist()

        legs_list.append(legs)
    return legs_list


def _find_batch(con_list):
    outer = []
    for i in con_list:
        if not isinstance(i, np.ndarray):
            i = np.array(i)
        outer.extend(i[i < 0].tolist())
    if not len(outer):
        return None
    if - len(outer) == min(outer):
        return None

    for leg in np.unique(outer):
        if sum(outer == leg) == 1:
            outer = np.delete(outer, outer.index(leg)).tolist()

    return outer


def _process_perm(con, batch_leg):
    p = list(range(len(con)))
    for i, ind in enumerate(batch_leg):
        j = con.index(ind)
        if i == j:
            continue
        con[i], con[j] = con[j], con[i]
        p[i], p[j] = p[j], p[i]

    return con, tuple(p)


def _make_dict(mode, inds=None, legs=None, batch_leg=None, p_list=None):
    d = {}
    d['mode'] = mode

    if d['mode'] == 'permute':
        d['perms'] = p_list

    elif d['mode'] == 'outer':
        d['inds'] = inds

    elif d['mode'] in ('diag', 'sum', 'trace'):
        d['inds'] = inds
        d['legs'] = legs

    elif d['mode'] == 'ndot':
        d['inds'] = inds
        d['legs'] = legs
        d['batch_leg'] = batch_leg

    else:
        raise ValueError

    return d


def _process_commands(con_list):
    conmmands = []
    operators = []

    # find sum index
    sum_legs = _find_sum(con_list)
    for leg in sum_legs:
        for i, con in enumerate(con_list):
            if leg in con:
                leg_ind = con.index(leg)
                con_list[i].remove(leg)
                conmmands.append(_make_dict('sum', [i], [leg_ind]))
                operators.append(ops.sum)

    # find trace
    trace_legs = _find_trace(con_list)
    for i, leg_list in enumerate(trace_legs):
        if len(leg_list):
            for legs in leg_list:
                conmmands.append(_make_dict('trace', [i], legs))
                operators.append(trace)

    order = _create_order(con_list)
    batch_legs = _find_batch(con_list)

    if not len(con_list[0]):
        return conmmands, operators

    while len(order):
        leg_now = order[-1]
        inds = []
        legs = []
        batch_legs_now = []

        # find the two tensors' indices
        for i, item in enumerate(con_list):
            if leg_now in item:
                inds.append(i)

        # check trace
        if len(inds) == 1:
            con_list[inds[0]], legs = _single_trace(con_list[inds[0]], leg_now)
            conmmands.append(_make_dict('trace', inds, legs))
            operators.append(trace)

        else:

            # find batch legs
            batch_leg_inds = []
            if batch_legs is not None:
                _tmp = np.intersect1d(con_list[inds[0]], con_list[inds[1]])
                batch_legs_now = np.intersect1d(
                    _tmp,
                    batch_legs,
                    False
                ).tolist()

                # find indices of batch legs 
                for batch_leg in batch_legs_now:
                    i_leg_0 = con_list[inds[0]].index(batch_leg)
                    i_leg_1 = con_list[inds[1]].index(batch_leg)
                    con_list[inds[0]].remove(batch_leg)
                    con_list[inds[1]].remove(batch_leg)
                    batch_leg_inds.append((i_leg_0, i_leg_1, None))

            ndot_legs = []
            ndot_leg_inds = []
            # find all ndot legs and their indices
            for _leg in con_list[inds[0]]:
                if _leg in con_list[inds[1]]:
                    i_leg_0 = con_list[inds[0]].index(_leg)
                    i_leg_1 = con_list[inds[1]].index(_leg)
                    ndot_legs.append(_leg)
                    ndot_leg_inds.append([i_leg_0, i_leg_1])

            # do ndot contraction and update order
            for _leg in ndot_legs:
                con_list[inds[0]].remove(_leg)
                con_list[inds[1]].remove(_leg)
            for _leg in ndot_legs:
                if _leg != leg_now:
                    order.remove(_leg)

            ndot_leg_inds = ndot_leg_inds[0] if len(ndot_leg_inds) == 1 else ndot_leg_inds
            conmmands.append(_make_dict('ndot', inds, ndot_leg_inds, batch_leg_inds))
            operators.append(nest_vmap(tensordot, batch_leg_inds, [0] * len(batch_leg_inds), 0) if len(
                batch_leg_inds) else tensordot)

            # merge two con_list
            for leg in con_list[inds[1]]:
                if leg not in batch_legs_now:
                    con_list[inds[0]].append(leg)
            con_list[inds[1]] = []
            con_list[inds[0]] = batch_legs_now + con_list[inds[0]]

        order = order[:-1]

    # do outer product
    for i, con in enumerate(con_list):
        if not i:
            continue
        if len(con):
            inds = [0, i]
            for leg in con:
                con_list[0].append(leg)
            con_list[i] = []
            conmmands.append(_make_dict('outer', inds))
            operators.append(tensordot)

    # do diagonal
    min_leg = min(con_list[0])
    for leg in range(-1, min_leg - 1, -1):
        num_leg = con_list[0].count(leg)
        while num_leg > 1:
            i = con_list[0].index(leg)
            j = con_list[0].index(leg, i + 1)
            conmmands.append(_make_dict('diag', [0], [i, j]))
            operators.append(ops.diagonal)
            con_list[0] = con_list[0][:i] + con_list[0][i + 1:j] + con_list[0][j + 1:] + [leg]
            num_leg = con_list[0].count(leg)

    # do final permutation
    fin_con = list(range(-1, -1 - len(con_list[0]), -1))
    con_list[0], p = _process_perm(con_list[0], fin_con)
    conmmands.append(_make_dict('permute', p_list=[p]))
    operators.append(ops.permute)

    return conmmands, operators


def _prod(x):
    out = 1
    for i in x:
        out *= i
    return out


class Ncon(nn.Cell):
    r"""
    Multiple-tensor contraction operator which has similar function to Einsum.

    Args:
        con_list (List[List[int]]): lists of indices for each tensor. 
             - The the number of each list in `con_list` should coincide with the corresponding tensor's dimensions.
             - The positive indices indicate the dimensions to be contracted or summed.
             - The negative indices indicate the dimensions to be keeped (as batch dimensions).

    Raises:
        ValueError: If the number of commands is not match the number of operations.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        Trace of a matrix:
        >>> a = ops.ones((3, 3))
        >>> Ncon([[1, 1]])([a])
        3.0

        Diagonal of a matrix:
        >>> Ncon([[-1, -1]])([a])
        [1. 1. 1.]

        Outer product:
        >>> b = ops.ones((2))
        >>> c = ops.ones((3))
        >>> Ncon([[-1], [-2]])([b, c]).shape
        (2, 3)

        Batch matrix multiplication
        >>> d = ops.ones((2, 3, 4))
        >>> e = ops.ones((2, 4, 1))
        >>> Ncon([[-1, -2, 1], [-1, 1, -3]])([d, e]).shape
        (2, 3, 1)
    """

    def __init__(self, con_list):
        super().__init__()
        self.con_list = tuple(con_list)
        _con_list = deepcopy(con_list)
        self.commands, self.ops = _process_commands(_con_list)
        if len(self.commands) != len(self.ops):
            raise ValueError(f'{self.commands} is not match {len(self.ops)}')

    def construct(self, ten_list):
        """
        The list of tensors to be conctracted.
        """
        i = 0
        for d in self.commands:
            if d['mode'] == 'diag':
                ten_list[0] = self.ops[i](ten_list[0], 0, *d['legs'])
            elif d['mode'] == 'permute':
                ten_list[0] = self.ops[i](ten_list[0], d['perms'][0])
            elif d['mode'] == 'sum':
                i1 = d['inds'][0]
                ten_list[i1] = self.ops[i](ten_list[i1], d['legs'][0])
            elif d['mode'] == 'trace':
                i1 = d['inds'][0]
                ten_list[i1] = self.ops[i](ten_list[i1], 0, d['legs'][0], d['legs'][1])
            elif d['mode'] == 'outer':
                i1, i2 = d['inds']
                ten_list[i1] = self.ops[i](ten_list[i1], ten_list[i2], 0)
            elif d['mode'] == 'ndot':
                i1, i2 = d['inds']
                ten_list[i1] = self.ops[i](ten_list[i1], ten_list[i2], d['legs'])
            else:
                i += 1
                continue
            i += 1

        return ten_list[0]

    def __repr__(self):
        s = f'Ncon: {self.con_list}\n'
        for d in self.commands:
            s += str(d) + '\n'
        return s


if __name__ == '__main__':
    import mindspore as ms

    ms.set_context(device_target="CPU", device_id=4,
                   mode=ms.GRAPH_MODE, save_graphs=False)
    np.random.seed(123)

    ncon = Ncon([[5, -1, 1, 4, 3, -2], [3, -2, -1, 4, 2], [2, -3], [-3, -4]])
    v1 = ops.ones((3, 1, 3, 4, 5, 2))
    v2 = ops.ones((5, 2, 1, 4, 6))
    v3 = ops.ones((6, 3))
    v4 = ops.ones((3, 4))
    print(ncon)
    out = ncon([v1, v2, v3, v4])
    print(out.shape)

    ncon = Ncon([[-1, 2], [-1, 1], [2, 1, -2]])
    v1 = ops.ones((20, 50))
    v2 = ops.ones((20, 2))
    v3 = ops.ones((50, 2, 7))
    print(ncon)
    out = ncon([v1, v2, v3])
    print(out.shape)

    ncon = Ncon([[-1, -2, 1], [-1, 1]])
    v1 = ops.ones((3, 4, 5))
    v2 = ops.ones((3, 5))
    print(ncon)
    out = ncon([v1, v2])
    print(out.shape)
