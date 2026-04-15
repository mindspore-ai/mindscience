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
"""jtnn_vae"""
import copy
import rdkit.Chem as Chem
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .mol_tree import MolTree
from .jtnn_enc import JTNNEncoder
from .jtnn_dec import JTNNDecoder
from .mpn import MPN, mol2graph
from .jtmpn import JTMPN
from .utils import CosineSimilarity, mv, squeeze
from .chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, decode_stereo


def set_batch_node_id(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.set_idx(tot)
            node.set_wid(vocab.get_index(node.smiles))
            tot += 1


class JTNNVAE(nn.Cell):
    """jitnnvae"""

    def __init__(self, vocab, hidden_size, latent_size, depth, stereo=True, beta=0.0):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth

        self.beta = beta

        self.embedding = nn.Embedding(vocab.size(), hidden_size)
        self.jtnn = JTNNEncoder(vocab, hidden_size, self.embedding)
        self.jtmpn = JTMPN(hidden_size, depth)
        self.mpn = MPN(hidden_size, depth)
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size // 2, self.embedding)

        self.t_mean = nn.Dense(hidden_size, latent_size // 2)
        self.t_var = nn.Dense(hidden_size, latent_size // 2)
        self.g_mean = nn.Dense(hidden_size, latent_size // 2)
        self.g_var = nn.Dense(hidden_size, latent_size // 2)

        self.assm_loss = nn.CrossEntropyLoss(reduction="sum")
        self.use_stereo = stereo
        if stereo:
            self.stereo_loss = nn.CrossEntropyLoss(reduction="sum")

        self.softmax = nn.Softmax()
        self.sort = ops.Sort(descending=True)
        self.sum = ops.ReduceSum()
        self.bmm = ops.BatchMatMul()
        self.cosine_similarity = CosineSimilarity()
        self.max_0 = ops.ArgMaxWithValue(0)

    @staticmethod
    def update_global_amap(fa_nid, new_global_amap, pred_amap, cur_node):
        """update_global_amap"""
        for nei_id, ctr_atom, nei_atom in pred_amap:
            if nei_id == fa_nid:
                continue
            new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]
        return new_global_amap

    def set_beta(self, beta):
        self.beta = beta

    def encode(self, mol_batch):
        set_batch_node_id(mol_batch, self.vocab)
        root_batch = [mol_tree.nodes[0] for mol_tree in mol_batch]
        tree_mess, tree_vec = self.jtnn(root_batch)

        smiles_batch = [mol_tree.smiles for mol_tree in mol_batch]
        mol_vec = self.mpn(mol2graph(smiles_batch))
        return tree_mess, tree_vec, mol_vec

    def encode_latent_mean(self, smiles_list):
        mol_batch = [MolTree(s) for s in smiles_list]
        for mol_tree in mol_batch:
            mol_tree.recover()

        _, tree_vec, mol_vec = self.encode(mol_batch)
        tree_mean = self.t_mean(tree_vec)
        mol_mean = self.g_mean(mol_vec)
        return ops.concat([tree_mean, mol_mean], 1)

    def construct(self, smiles_input):
        """construct"""
        mol_batch = []
        for smiles in smiles_input.asnumpy():
            mol_tree = MolTree(smiles)
            mol_tree.recover()
            mol_tree.assemble()
            mol_tree.set_cands()
            mol_batch.append(mol_tree)

        batch_size = len(mol_batch)

        tree_mess, tree_vec, mol_vec = self.encode(mol_batch)

        tree_mean = self.t_mean(tree_vec)
        tree_log_var = -ops.abs(self.t_var(tree_vec))
        mol_mean = self.g_mean(mol_vec)
        mol_log_var = -ops.abs(self.g_var(mol_vec))

        z_mean = ops.concat([tree_mean, mol_mean], 1)
        z_log_var = ops.concat([tree_log_var, mol_log_var], 1)
        kl_loss = -0.5 * self.sum(1.0 + z_log_var - z_mean * z_mean - ops.exp(z_log_var)) / batch_size

        epsilon = ops.standard_normal((batch_size, self.latent_size // 2))
        tree_vec = tree_mean + ops.exp(tree_log_var / 2) * epsilon
        epsilon = ops.standard_normal((batch_size, self.latent_size // 2))
        mol_vec = mol_mean + ops.exp(mol_log_var / 2) * epsilon

        word_loss, topo_loss, _, _ = self.decoder(mol_batch, tree_vec)
        assm_loss = self.assm(mol_batch, mol_vec, tree_mess)
        if self.use_stereo:
            stereo_loss, _ = self.stereo(mol_batch, mol_vec)
        else:
            stereo_loss = 0

        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + self.beta * kl_loss

        return loss

    def assm(self, mol_batch, mol_vec, tree_mess):
        """assm"""
        cands = []
        batch_idx = []
        for i, mol_tree in enumerate(mol_batch):
            for node in mol_tree.nodes:
                # Leaf node's attachment is determined by neighboring node's attachment
                if node.is_leaf or len(node.cands) == 1:
                    continue
                temp_list = []
                for cand in node.cand_mols:
                    temp_list.append((cand, mol_tree.nodes, node))
                cands.extend(temp_list)
                batch_idx.extend([i] * len(node.cands))

        cand_vec = self.jtmpn(cands, tree_mess)
        cand_vec = self.g_mean(cand_vec)
        batch_idx = ms.Tensor(batch_idx, ms.int32)
        mol_vec = mol_vec.take(batch_idx, 0)

        mol_vec = mol_vec.view(-1, 1, self.latent_size // 2)
        cand_vec = cand_vec.view(-1, self.latent_size // 2, 1)
        scores = squeeze(self.bmm(mol_vec, cand_vec))

        tot = 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = []
            for node in mol_tree.nodes:
                if len(node.cands) > 1 and not node.is_leaf:
                    comp_nodes.append(node)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                label = ms.Tensor([label], ms.int32)
                cur_loss = self.assm_loss(cur_score.view(1, -1), label)
                all_loss.append(cur_loss)

        all_loss = ops.stack(all_loss).sum() / len(mol_batch)
        return all_loss

    def stereo(self, mol_batch, mol_vec):
        """stereo"""
        stereo_cands, batch_idx = [], []
        labels = []
        for i, mol_tree in enumerate(mol_batch):
            cands = mol_tree.stereo_cands
            if len(cands) == 1:
                continue
            if mol_tree.smiles3D not in cands:
                cands.append(mol_tree.smiles3D)
            stereo_cands.extend(cands)
            batch_idx.extend([i] * len(cands))
            labels.append((cands.index(mol_tree.smiles3D), len(cands)))

        if not labels:
            return ops.zeros(1, ms.float32), 1.0

        batch_idx = ms.Tensor(batch_idx, ms.int32)
        stereo_cands = self.mpn(mol2graph(stereo_cands))
        stereo_cands = self.g_mean(stereo_cands)
        stereo_labels = mol_vec.take(batch_idx, 0)
        scores = self.cosine_similarity(stereo_cands, stereo_labels)

        st, acc = 0, 0
        all_loss = []
        for label, le in labels:
            cur_scores = scores.narrow(0, st, le)
            label = ms.Tensor([label], ms.int32)
            all_loss.append(self.stereo_loss(cur_scores.view(1, -1), label))
            st += le

        all_loss = ops.stack(all_loss).sum() / len(labels)
        return all_loss, acc * 1.0 / len(labels)

    def reconstruct(self, smiles, prob_decode=False):
        """reconstruct"""

        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])
        tree_mean = self.t_mean(tree_vec)
        tree_log_var = -ops.abs(self.t_var(tree_vec))
        mol_mean = self.g_mean(mol_vec)
        mol_log_var = -ops.abs(self.g_var(mol_vec))

        epsilon = ops.standard_normal((1, self.latent_size // 2))
        tree_vec = tree_mean + ops.exp(tree_log_var / 2) * epsilon
        epsilon = ops.standard_normal((1, self.latent_size // 2))
        mol_vec = mol_mean + ops.exp(mol_log_var / 2) * epsilon

        return self.decode(tree_vec, mol_vec, prob_decode)

    def recon_eval(self, smiles):
        """recon_eval"""
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])

        tree_mean = self.t_mean(tree_vec)
        tree_log_var = -ops.abs(self.t_var(tree_vec))
        mol_mean = self.g_mean(mol_vec)
        mol_log_var = -ops.abs(self.g_var(mol_vec))

        all_smiles = []
        for _ in range(10):
            epsilon = ops.standard_normal((1, self.latent_size // 2))
            tree_vec = tree_mean + ops.exp(tree_log_var / 2) * epsilon
            epsilon = ops.standard_normal((1, self.latent_size // 2))
            mol_vec = mol_mean + ops.exp(mol_log_var / 2) * epsilon
            for _ in range(10):
                new_smiles = self.decode(tree_vec, mol_vec, prob_decode=True)
                all_smiles.append(new_smiles)
        return all_smiles

    def sample_prior(self, prob_decode=False):
        tree_vec = ops.standard_normal((1, self.latent_size // 2))
        mol_vec = ops.standard_normal((1, self.latent_size // 2))
        return self.decode(tree_vec, mol_vec, prob_decode)

    def sample_eval(self):
        tree_vec = ops.standard_normal((1, self.latent_size // 2))
        mol_vec = ops.standard_normal((1, self.latent_size // 2))
        all_smiles = []
        for _ in range(100):
            s = self.decode(tree_vec, mol_vec, prob_decode=True)
            all_smiles.append(s)

        return all_smiles

    def decode(self, tree_vec, mol_vec, prob_decode):
        """decode"""
        pred_root, pred_nodes = self.decoder.decode(tree_vec, prob_decode)

        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        tree_mess = self.jtnn([pred_root])[0]

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, mol_vec, pred_nodes, cur_mol, global_amap, [], pred_root, None,
                                    prob_decode)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        if cur_mol is None:
            return None
        if not self.use_stereo:
            return Chem.MolToSmiles(cur_mol)

        smiles_2d = Chem.MolToSmiles(cur_mol)
        stereo_cands = decode_stereo(smiles_2d)
        if len(stereo_cands) == 1:
            return stereo_cands[0]
        stereo_vecs = self.mpn(mol2graph(stereo_cands))
        stereo_vecs = self.g_mean(stereo_vecs)
        scores = self.cosine_similarity(stereo_vecs, mol_vec)
        max_id, _ = self.max_0(scores)

        return stereo_cands[max_id]

    def dfs_assemble(self, tree_mess, mol_vec, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node,
                     prob_decode):
        """dfs assemble"""
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if not cands:
            return None
        _, cand_mols, cand_amap = zip(*cands)

        cands = [(candmol, all_nodes, cur_node) for candmol in cand_mols]

        cand_vecs = self.jtmpn(cands, tree_mess)
        cand_vecs = self.g_mean(cand_vecs)
        mol_vec = squeeze(mol_vec)
        scores = mv(cand_vecs, mol_vec) * 20

        if prob_decode:
            probs = squeeze(self.softmax(scores.view(1, -1))) + 1e-5
            cand_idx = ops.multinomial(probs, ops.size(probs), replacement=False)
        else:
            _, cand_idx = self.sort(scores)

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(ops.size(cand_idx)):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i]]
            new_global_amap = copy.deepcopy(global_amap)

            new_global_amap = self.update_global_amap(fa_nid, new_global_amap, pred_amap, cur_node)

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            result = True
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                cur_mol = self.dfs_assemble(tree_mess, mol_vec, all_nodes, cur_mol, new_global_amap, pred_amap,
                                            nei_node, cur_node, prob_decode)
                if cur_mol is None:
                    result = False
                    break

            if result:
                return cur_mol

        return None
