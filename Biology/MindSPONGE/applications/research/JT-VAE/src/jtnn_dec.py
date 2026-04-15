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
"""jtnn_dec"""
import random
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from .mol_tree import MolTreeNode
from .nnutils import gru
from .chemutils import enum_assemble
from .utils import squeeze

MAX_NB = 8
MAX_DECODE_LEN = 100


class JTNNDecoder(nn.Cell):
    """jtnndecoder"""

    def __init__(self, vocab, hidden_size, latent_size, embedding=None):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        # GRU Weights
        self.w_z = nn.Dense(2 * hidden_size, hidden_size)
        self.u_r = nn.Dense(hidden_size, hidden_size, has_bias=False)
        self.w_r = nn.Dense(hidden_size, hidden_size)
        self.w_h = nn.Dense(2 * hidden_size, hidden_size)

        # Feature Aggregate Weights
        self.w = nn.Dense(latent_size + hidden_size, hidden_size)
        self.u = nn.Dense(latent_size + 2 * hidden_size, hidden_size)

        # Output Weights
        self.w_o = nn.Dense(hidden_size, self.vocab_size)
        self.u_s = nn.Dense(hidden_size, 1)

        # Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(reduction="sum")
        self.stop_loss = nn.BCEWithLogitsLoss(reduction="sum")

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.sum = ops.ReduceSum()
        self.max_1 = ops.ArgMaxWithValue(1)

    @staticmethod
    def get_nei_trace(prop_list, h_list, padding, h):
        """get nei trace"""
        cur_x = []
        cur_h_nei, cur_o_nei = [], []

        for node_x, real_y, _ in prop_list:
            # Neighbors for message passing (target not included)
            cur_nei = []
            for node_y in node_x.neighbors:
                if node_y.idx != real_y.idx:
                    cur_nei.append(h_list[h[(node_y.idx, node_x.idx)]])
            pad_len = MAX_NB - len(cur_nei)
            cur_h_nei.extend(cur_nei)
            cur_h_nei.extend([padding] * pad_len)

            # Neighbors for stop prediction (all neighbors)
            cur_nei = []
            for node_y in node_x.neighbors:
                cur_nei.append(h_list[h[(node_y.idx, node_x.idx)]])
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

            # Current clique embedding
            cur_x.append(node_x.wid)
        return cur_x, cur_h_nei, cur_o_nei

    def get_trace(self, node):
        super_root = MolTreeNode("")
        super_root.set_idx(-1)
        trace = []
        dfs(trace, node, super_root)
        return [(x.smiles, y.smiles, z) for x, y, z in trace]

    def construct(self, mol_batch, mol_vec):
        """construct"""
        super_root = MolTreeNode("")
        super_root.set_idx(-1)

        # Initialize
        pred_hiddens, pred_mol_vecs, pred_targets = [], [], []
        stop_hiddens, stop_targets = [], []
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], super_root)
            traces.append(s)
            for node in mol_tree.nodes:
                node.empty_neighbors()

        # Predict Root
        pred_hiddens.append(ops.zeros((len(mol_batch), self.hidden_size), ms.float32))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_mol_vecs.append(mol_vec)

        max_iter = max([len(tr) for tr in traces])
        padding = ops.zeros(self.hidden_size, ms.float32)
        h = {}
        h_list = []

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x, cur_h_nei, cur_o_nei = self.get_nei_trace(prop_list, h_list, padding, h)

            # Clique embedding
            cur_x = ms.Tensor(cur_x, ms.int32)
            cur_x = self.embedding(cur_x)

            # Message passing
            cur_h_nei = ops.stack(cur_h_nei, 0).view(-1, MAX_NB, self.hidden_size)
            new_h = gru(cur_x, cur_h_nei, self.w_z, self.w_r, self.u_r, self.w_h)

            # Node Aggregate
            cur_o_nei = ops.stack(cur_o_nei, 0).view(-1, MAX_NB, self.hidden_size)
            cur_o = cur_o_nei.sum(axis=1)

            # Gather targets
            pred_target, pred_list = [], []
            stop_target = []
            for i, m in enumerate(prop_list):
                node_x, node_y, direction = m
                x, y = node_x.idx, node_y.idx
                h_list.append(new_h[i])
                h[(x, y)] = len(h_list) - 1
                node_y.add_neighbor(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i)
                stop_target.append(direction)

            # Hidden states for stop prediction
            cur_batch = ms.Tensor(batch_list, ms.int32)
            cur_mol_vec = mol_vec.take(cur_batch, axis=0)
            stop_hidden = ops.concat([cur_x, cur_o, cur_mol_vec], 1)
            stop_hiddens.append(stop_hidden)
            stop_targets.extend(stop_target)

            # Hidden states for clique prediction
            if pred_list:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = ms.Tensor(batch_list, ms.int32)
                pred_mol_vecs.append(mol_vec.take(cur_batch, axis=0))

                cur_pred = ms.Tensor(pred_list, ms.int32)
                pred_hiddens.append(new_h.take(cur_pred, axis=0))
                pred_targets.extend(pred_target)

        # Last stop at root
        cur_x, cur_o_nei = [], []
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = []
            for node_y in node_x.neighbors:
                cur_nei.append(h_list[h.get((node_y.idx, node_x.idx))])
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = ms.Tensor(cur_x, ms.int32)
        cur_x = self.embedding(cur_x)
        cur_o_nei = ops.stack(cur_o_nei, 0).view(-1, MAX_NB, self.hidden_size)
        cur_o = cur_o_nei.sum(axis=1)

        stop_hidden = ops.concat([cur_x, cur_o, mol_vec], 1)
        stop_hiddens.append(stop_hidden)
        stop_targets.extend([0] * len(mol_batch))

        # Predict next clique
        pred_hiddens = ops.concat(pred_hiddens, 0)
        pred_mol_vecs = ops.concat(pred_mol_vecs, 0)
        pred_vecs = ops.concat([pred_hiddens, pred_mol_vecs], 1)
        pred_vecs = self.relu(self.w(pred_vecs))
        pred_scores = self.w_o(pred_vecs)
        pred_targets = ms.Tensor(pred_targets, ms.int32)

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        preds, _ = self.max_1(pred_scores)
        pred_acc = ops.cast(ops.equal(preds, pred_targets), ms.float32)
        pred_acc = self.sum(pred_acc) / pred_targets.size

        # Predict stop
        stop_hiddens = ops.concat(stop_hiddens, 0)
        stop_vecs = self.relu(self.u(stop_hiddens))
        stop_scores = squeeze(self.u_s(stop_vecs))
        stop_targets = ms.Tensor(stop_targets, ms.float32)

        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = ops.cast(ops.ge(stop_scores, 0), ms.float32)
        stop_acc = ops.cast(ops.equal(stops, stop_targets), ms.float32)
        stop_acc = self.sum(stop_acc) / stop_targets.size

        res_value = pred_loss, stop_loss, float(pred_acc), float(stop_acc)

        return res_value

    def decode(self, mol_vec, prob_decode):
        """decode"""
        stack = []
        init_hidden = ops.zeros((1, self.hidden_size), ms.float32)
        zero_pad = ops.zeros((1, 1, self.hidden_size), ms.float32)

        # Root Prediction
        root_hidden = ops.concat([init_hidden, mol_vec], 1)
        root_hidden = self.relu(self.w(root_hidden))
        root_score = self.w_o(root_hidden)
        root_wid, _ = self.max_1(root_score)
        root_wid = root_wid.item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.set_wid(root_wid)
        root.set_idx(0)
        stack.append((root, self.vocab.get_slots(root.wid)))

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, fa_slot = stack[-1]
            cur_h_nei = []
            for node_y in node_x.neighbors:
                cur_h_nei.append(h.get((node_y.idx, node_x.idx)))
            if cur_h_nei:
                cur_h_nei = ops.stack(cur_h_nei, 0).view(1, -1, self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = ms.Tensor([node_x.wid], ms.int32)
            cur_x = self.embedding(cur_x)

            # Predict stop
            cur_h = cur_h_nei.sum(axis=1)
            stop_hidden = ops.concat([cur_x, cur_h, mol_vec], 1)
            stop_hidden = self.relu(self.u(stop_hidden))
            stop_score = squeeze(self.sigmoid(self.u_s(stop_hidden) * 20))

            if prob_decode:
                backtrack = (random.random() < (1.0 - stop_score[0]))
            else:
                backtrack = (stop_score < 0.5)

            if not backtrack:
                new_h = gru(cur_x, cur_h_nei, self.w_z, self.w_r, self.u_r, self.w_h)
                pred_hidden = ops.concat([new_h, mol_vec], 1)
                pred_hidden = self.relu(self.w(pred_hidden))
                pred_score = self.softmax(self.w_o(pred_hidden) * 20)
                if prob_decode:
                    sort_wid = ops.multinomial(squeeze(pred_score), 5)
                else:
                    _, sort_wid = ops.Sort(axis=-1, descending=True)(pred_score)
                    sort_wid = squeeze(sort_wid)

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.set_wid(next_wid)
                    node_y.set_idx(step + 1)
                    node_y.add_neighbor(node_x)
                    h[(node_x.idx, node_y.idx)] = new_h[0]
                    stack.append((node_y, next_slots))
                    all_nodes.append(node_y)

            if backtrack:
                if len(stack) == 1:
                    break

                node_fa, _ = stack[-2]
                cur_h_nei = []
                for node_y in node_x.neighbors:
                    if node_y.idx != node_fa.idx:
                        cur_h_nei.append(h.get((node_y.idx, node_x.idx)))

                if cur_h_nei:
                    cur_h_nei = ops.stack(cur_h_nei, 0).view(1, -1, self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = gru(cur_x, cur_h_nei, self.w_z, self.w_r, self.u_r, self.w_h)
                h[(node_x.idx, node_fa.idx)] = new_h[0]
                node_fa.add_neighbor(node_x)

                stack.pop()

        return root, all_nodes


def dfs(stack, x, fa):
    """dfs method"""
    for y in x.neighbors:
        if y.idx == fa.idx:
            continue
        stack.append((x, y, 1))
        dfs(stack, y, x)
        stack.append((y, x, 0))


def have_slots(fa_slots, ch_slots):
    """have slots"""
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i, s1 in enumerate(fa_slots):
        a1, c1, h1 = s1
        for j, s2 in enumerate(ch_slots):
            a2, c2, h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append((i, j))

    if not matches:
        return False

    fa_match, ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2:
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2:
        ch_slots.pop(ch_match[0])

    return True


def can_assemble(node_x, node_y):
    """can assemble"""
    neis = node_x.neighbors + [node_y]
    for i, nei in enumerate(neis):
        nei.nid = i

    neighbors = []
    for nei in neis:
        if nei.mol.GetNumAtoms() > 1:
            neighbors.append(nei)
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = []
    for nei in neis:
        if nei.mol.GetNumAtoms() == 1:
            singletons.append(nei)
    neighbors = singletons + neighbors
    prev_nodes, prev_amap = [], []
    cands = enum_assemble(node_x, neighbors, prev_nodes, prev_amap)
    return len(cands) > 0
