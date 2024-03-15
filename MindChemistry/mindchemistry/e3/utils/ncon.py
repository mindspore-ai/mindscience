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
"""ncon"""
from copy import deepcopy
import numpy as np

from mindspore import ops, nn, vmap
from mindspore.numpy import tensordot, trace, expand_dims


def list_to_tuple(lst):
    """list_to_tuple"""
    return tuple(list_to_tuple(item) if isinstance(item, list) else item for item in lst)


def nest_vmap(fn, in_list, out_list, pt):
    """nest vmap function"""
    if pt == len(in_list) - 1:
        return vmap(fn, in_list[pt], out_list[pt])
    return vmap(nest_vmap(fn, in_list, out_list, pt + 1), in_list[pt], out_list[pt])


def _create_order(con_list):
    """ Identify all unique, positive indices and return them sorted. """
    flat_con = np.concatenate(con_list)
    return np.unique(flat_con[flat_con > 0]).tolist()


def _single_trace(con, leg):
    """_single_trace"""
    leg = np.where(np.array(con) == leg)[0]
    con = np.delete(con, leg).tolist()
    return con, leg.tolist()


def _find_sum(con_list):
    """_find_sum

    Args:
        con_list: con_list

    Returns:
        legs
    """
    flat = []
    for item in con_list:
        flat += item
    legs = []
    for leg in np.unique(flat):
        if leg < 0:
            continue
        if np.sum(np.array(flat) == leg) == 1:
            legs.append(leg)
    return legs


def _find_trace(con_list):
    """_find_trace

    Args:
        con_list: con_list

    Returns:
        legs_list
    """
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
    """_find_batch

    Args:
        con_list: con_list

    Returns:
        outer
    """
    outer = []
    for i in con_list:
        if not isinstance(i, np.ndarray):
            i = np.array(i)
        outer.extend(i[i < 0].tolist())
    if not outer:
        return None
    if -len(outer) == min(outer):
        return None

    for leg in np.unique(outer):
        if sum(outer == leg) == 1:
            outer = np.delete(outer, outer.index(leg)).tolist()

    return outer


def _process_perm(con, batch_leg):
    """_process_perm"""
    p = list(range(len(con)))
    for i, ind in enumerate(batch_leg):
        j = con.index(ind)
        if i == j:
            continue
        con[i], con[j] = con[j], con[i]
        p[i], p[j] = p[j], p[i]

    return con, tuple(p)


def _make_dict(mode,
               inds=None,
               legs=None,
               batch_leg=None,
               p_list=None,
               res_legs=None,
               permute_index=None,
               expand_axis=None):
    """_summary_

    Args:
        mode: mode
        inds: inds. Defaults to None.
        legs: legs. Defaults to None.
        batch_leg: batch_leg. Defaults to None.
        p_list: p_list. Defaults to None.
        res_legs: res_legs. Defaults to None.
        permute_index: permute_index. Defaults to None.
        expand_axis: expand_axis. Defaults to None.

    Raises:
        ValueError: ValueError

    Returns:
        d
    """
    d = {}
    calculate_mode = 'mode'
    indices = 'inds'
    indices_legs = 'legs'
    d[calculate_mode] = mode

    if d[calculate_mode] == 'permute':
        d['perms'] = p_list

    elif d[calculate_mode] == 'outer':
        d[indices] = inds

    elif d[calculate_mode] in ('diag', 'sum', 'trace'):
        d[indices] = inds
        d[indices_legs] = legs

    elif d[calculate_mode] == 'ndot':
        d[indices] = inds
        d[indices_legs] = legs
        d['batch_leg'] = batch_leg

    elif d[calculate_mode] == 'hadamard':
        d[indices] = inds
        d[indices_legs] = legs
        d['res_legs'] = res_legs
        d['permute_index'] = permute_index
        d['expand_axis'] = expand_axis

    else:
        raise ValueError

    return d


def _process_commands(con_list):
    """_process_commands

    Args:
        con_list: con_list

    Returns:
        conmmands, operators
    """
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
        if leg_list:
            for legs in leg_list:
                conmmands.append(_make_dict('trace', [i], legs))
                operators.append(trace)

    order = _create_order(con_list)
    batch_legs = _find_batch(con_list)

    if not con_list[0]:
        return conmmands, operators

    do_ndot(con_list, conmmands, operators, order, batch_legs)

    # do Hadamard(alike) product
    do_hadamard(con_list, conmmands, operators)

    # do outer product
    for i, con in enumerate(con_list):
        if not i:
            continue
        if con:
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


def do_ndot(con_list, conmmands, operators, order, batch_legs):
    """do_ndot

    Args:
        con_list: con_list
        conmmands: conmmands
        operators: operators
        order: order
        batch_legs: batch_legs
    """
    while order:
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
                tmp = np.intersect1d(con_list[inds[0]], con_list[inds[1]])
                batch_legs_now = np.intersect1d(tmp, batch_legs, False).tolist()

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
            for leg in con_list[inds[0]]:
                if leg in con_list[inds[1]]:
                    i_leg_0 = con_list[inds[0]].index(leg)
                    i_leg_1 = con_list[inds[1]].index(leg)
                    ndot_legs.append(leg)
                    ndot_leg_inds.append([i_leg_0, i_leg_1])

            # do ndot contraction and update order
            for leg in ndot_legs:
                con_list[inds[0]].remove(leg)
                con_list[inds[1]].remove(leg)
            for leg in ndot_legs:
                if leg != leg_now:
                    order.remove(leg)

            ndot_leg_inds = ndot_leg_inds[0] if len(ndot_leg_inds) == 1 else np.array(
                ndot_leg_inds).transpose().tolist()
            conmmands.append(_make_dict('ndot', inds, list_to_tuple(ndot_leg_inds), batch_leg_inds))
            operators.append(
                nest_vmap(tensordot, batch_leg_inds, [0] * len(batch_leg_inds), 0) if batch_leg_inds else tensordot)

            # merge two con_list
            for leg in con_list[inds[1]]:
                if leg not in batch_legs_now:
                    con_list[inds[0]].append(leg)
            con_list[inds[1]] = []
            con_list[inds[0]] = batch_legs_now + con_list[inds[0]]

        order = order[:-1]


def do_hadamard(con_list, conmmands, operators):
    """do_hadamard

    Args:
        con_list: con_list
        conmmands: conmmands
        operators: operators
    """
    is_con_list_not_none = len(con_list) == 2 and con_list[1]
    if  is_con_list_not_none and not [i for i in con_list[0] if i > 0] and not [i for i in con_list[1] if i > 0]:
        con_list_all = []
        for con in con_list:
            con_list_all.extend(con)
        con_min_leg = min(con_list_all)
        out_list = [i for i in range(-1, con_min_leg - 1, -1)]

        res_legs = []
        for ind in out_list:
            for i, con in enumerate(con_list):
                if ind in con:
                    res_legs.append((i, con.index(ind)))
                    break

        hadamard_legs = [[], []]
        con_raw = deepcopy(con_list)
        handle_inds(con_list, out_list, hadamard_legs)

        expand_axis = deepcopy(hadamard_legs)
        for i, axis in enumerate(expand_axis):
            if axis and len(axis) <= 1:
                expand_axis[i] = axis[0]

        # input permute
        permute_index = [[], []]
        con_sort = deepcopy(con_raw)
        for i, con in enumerate(con_raw):
            con_sort[i].sort(reverse=True)
            _, permute_index[i] = _process_perm(con, con_sort[i])

        conmmands.append(
            _make_dict('hadamard',
                       inds=[0, 1],
                       legs=hadamard_legs,
                       res_legs=res_legs,
                       permute_index=permute_index,
                       expand_axis=expand_axis))
        operators.append([ops.permute, ops.tile, ops.mul, expand_dims])


def handle_inds(con_list, out_list, hadamard_legs):
    """handle_inds"""
    for i, con in enumerate(con_list):
        if con:
            for ind in out_list:
                if ind not in con:
                    hadamard_legs[i].append((out_list.index(ind)))
            if i:
                con_list[i] = []
            else:
                con_list[i] = out_list


class Ncon(nn.Cell):
    r"""
    Multiple-tensor contraction operator which has similar function to Einsum.

    Args:
        con_list (List[List[int]]): lists of indices for each tensor.
            The the number of each list in `con_list` should coincide with the corresponding tensor's dimensions.
            The positive indices indicate the dimensions to be contracted or summed.
            The negative indices indicate the dimensions to be keeped (as batch dimensions).

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
        con_list_copy = deepcopy(con_list)
        self.commands, self.ops = _process_commands(con_list_copy)
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
            elif d['mode'] == 'hadamard':
                i1, i2 = d['inds']
                a = ten_list[i1]
                b = ten_list[i2]
                res_legs = d['res_legs']

                a = ops.permute(a, d['permute_index'][i1])
                b = ops.permute(b, d['permute_index'][i2])

                if d['expand_axis'][i1]:
                    a = expand_dims(a, d['expand_axis'][i1])
                if d['expand_axis'][i2]:
                    b = expand_dims(b, d['expand_axis'][i2])

                tile_index = [[1 for _ in res_legs], [1 for _ in res_legs]]
                for j in range(len(d['legs'][i1])):
                    tile_index[0][d['legs'][i1][j]] = ten_list[res_legs[d['legs'][i1][j]][0]].shape[res_legs[
                        d['legs'][i1][j]][1]]
                for j in range(len(d['legs'][i2])):
                    tile_index[1][d['legs'][i2][j]] = ten_list[res_legs[d['legs'][i2][j]][0]].shape[res_legs[
                        d['legs'][i2][j]][1]]
                a = ops.tile(a, tuple(tile_index[0]))
                b = ops.tile(b, tuple(tile_index[1]))

                ten_list[i1] = ops.mul(a, b)
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


def test_other():
    """test_other"""
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


def test_diagonal():
    """test_diagonal"""
    ncon = Ncon([[-1, -1]])
    v1 = ops.ones((3, 3))
    print(ncon)
    out = ncon([v1])
    print(out.shape)
    print(out)


def test_outer():
    """test_other"""
    ncon = Ncon([[-1], [-2]])
    v1 = ops.ones((2))
    v2 = ops.ones((3))
    print(ncon)
    out = ncon([v1, v2])
    print(out.shape)
    print(out)


def test_outer_multi_input():
    """test_other"""
    ncon = Ncon([[-1], [-2], [-3]])
    v1 = ops.ones((2))
    v2 = ops.ones((3))
    v3 = ops.ones((4))
    print(ncon)
    out = ncon([v1, v2, v3])
    print(out.shape)
    print(out)


def test_ndot():
    """test_other"""
    ncon = Ncon([[-1, -2, 1], [-1, 1]])
    v1 = ops.ones((3, 4, 5))
    v2 = ops.ones((3, 5))
    print(ncon)
    out = ncon([v1, v2])
    print(out.shape)
    print(out)


def test_ndot_2():
    """test_other"""
    ncon = Ncon([[-1, -2, 1, 2], [-1, 1, 2]])
    v1 = ops.ones((3, 4, 5, 6))
    v2 = ops.ones((3, 5, 6))
    print(ncon)
    out = ncon([v1, v2])
    print(out.shape)
    print(out)


def test_hadamard():
    """test_hadamard"""
    a = np.arange(6).reshape((2, 3))
    b = np.arange(6).reshape((2, 3))
    print(a)
    print(b)
    einstr = f"zu,zu->zu"
    d = np.einsum(einstr, a, b)
    print(d)
    print(d.shape)

    ma = ms.Tensor(a, dtype=ms.float32)
    mb = ms.Tensor(b, dtype=ms.float32)
    ncon = Ncon([[-1, -2], [-1, -2]])
    print(ncon)
    md = ncon([ma, mb])
    print(md.shape)
    print(np.allclose(md.asnumpy(), d))


def test_hadamard_alike():
    """test_hadamard_alike"""
    a = np.arange(8).reshape((2, 4))
    b = np.arange(24).reshape((2, 3, 4))
    print(a)
    print(b)
    einstr = f"zi,zui->zui"
    d = np.einsum(einstr, a, b)
    print(d)
    print(d.shape)

    ma = ms.Tensor(a, dtype=ms.float32)
    mb = ms.Tensor(b, dtype=ms.float32)
    ncon = Ncon([[-1, -3], [-1, -2, -3]])
    print(ncon)
    md = ncon([ma, mb])
    print(md.shape)
    print(np.allclose(md.asnumpy(), d))


def test_hadamard_with_outer():
    """test_hadamard_with_outer"""
    a = np.arange(24).reshape((2, 3, 4))
    b = np.arange(30).reshape((2, 3, 5))
    print(f"a:\n {a}")
    print(f"b:\n {b}")

    einstr = f"zui,zuj->zuij"

    d = np.einsum(einstr, a, b)
    print(f"d:\n {d}")
    print(f"d.shape:\n {d.shape}")

    ma = ms.Tensor(a, dtype=ms.float32)
    mb = ms.Tensor(b, dtype=ms.float32)

    ncon = Ncon([[-1, -2, -3], [-1, -2, -4]])
    print(ncon)
    md = ncon([ma, mb])
    print(md.shape)
    print(np.allclose(md.asnumpy(), d))


def test_hadamard_outer_nosequential():
    """test_hadamard_outer_nosequential"""
    a = np.arange(8).reshape((2, 4))
    b = np.arange(30).reshape((2, 5, 3))
    print(f"a:\n {a}")
    print(f"b:\n {b}")

    einstr = f"ac,adb->abcd"

    d = np.einsum(einstr, a, b)
    print(f"d:\n {d}")
    print(f"d.shape:\n {d.shape}")
    ma = ms.Tensor(a, dtype=ms.float32)
    mb = ms.Tensor(b, dtype=ms.float32)

    ncon = Ncon([[-1, -3], [-1, -4, -2]])
    print(ncon)
    md = ncon([ma, mb])
    print(md.shape)
    print(np.allclose(md.asnumpy(), d))


def test_sum():
    """test_other"""
    ncon = Ncon([[1, 2]])
    v1 = ops.ones((2, 3))
    print(ncon)
    out = ncon([v1])
    print(out.shape)
    print(out)


if __name__ == '__main__':
    import mindspore as ms

    ms.set_context(device_target="GPU", device_id=4, mode=ms.GRAPH_MODE, save_graphs=False)
    np.random.seed(123)

    test_hadamard_outer_nosequential()
