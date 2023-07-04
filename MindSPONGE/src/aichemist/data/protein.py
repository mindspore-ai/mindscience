# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
"""
protein
"""

import os
import warnings
from collections import defaultdict
import numpy as np
from rdkit import Chem
from ..util import scatter_add, scatter_max, scatter_min
from .molecule import Molecule, MoleculeBatch
from . import feature
from ..core import Registry as R
from ..util import pretty


class Protein(Molecule):
    """
    Proteins with predefined chemical features.
    Support both residue-level and atom-level operations and ensure consistency between two views.

    .. warning::

        The order of residues must be the same as the protein sequence.
        However, this class doesn't enforce any order on nodes or edges.
        Nodes may have a different order with residues.

    Parameters:
        edges (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        resi_type (array_like, optional): residue types of shape :math:`(|V_{res}|,)`
        view (str, optional): default view for this protein. Can be ``atom`` or ``residue``.
        atom_name (array_like, optional): atom names in a residue of shape :math:`(|V|,)`
        atom_resi (array_like, optional): atom id to residue id mapping of shape :math:`(|V|,)`
        resi_feat (array_like, optional): residue features of shape :math:`(|V_{res}|, ...)`
        is_hetero_atom (array_like, optional): hetero atom indicators of shape :math:`(|V|,)`
        occupancy (array_like, optional): occupancy of shape :math:`(|V|,)`
        b_factor (array_like, optional): temperature factors of shape :math:`(|V|,)`
        resi_number (array_like, optional): residue numbers of shape :math:`(|V_{res}|,)`
        insertion_code (array_like, optional): insertion codes of shape :math:`(|V_{res}|,)`
        chain_id (array_like, optional): chain ids of shape :math:`(|V_{res}|,)`
    """

    dummy_protein = Chem.MolFromSequence("G")
    dummy_atom = dummy_protein.GetAtomWithIdx(0)

    # TODO: rdkit isn't compatible with X in the sequence
    _resi2id = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                "ARG": 17, "TYR": 18, "TRP": 19}
    _atom2id = {"C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
                "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
                "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
                "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
                "OH": 33, "OXT": 34, "SD": 35, "SG": 36, "UNK": 37}
    _chain2id = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10,
                 "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20,
                 "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}
    _id2resi = {v: k for k, v in _resi2id.items()}
    _id2resi_abbr = {0: "G", 1: "A", 2: "S", 3: "P", 4: "V", 5: "T", 6: "C", 7: "I", 8: "L", 9: "N",
                     10: "D", 11: "Q", 12: "K", 13: "E", 14: "M", 15: "H", 16: "F", 17: "R", 18: "Y", 19: "W"}
    _resi_abbr2id = {v: k for k, v in _id2resi_abbr.items()}
    _id2atom = {v: k for k, v in _atom2id.items()}
    _id2chain = {v: k for k, v in _chain2id.items()}
    _caches = {'atom_': 'ndata', 'bond_': 'edata', 'mol_': 'gdata', 'resi_': 'rdata',
               'node_': 'ndata', 'edge_': 'edata', 'graph_': 'gdata'}

    def __init__(self,
                 edges,
                 resi_type,
                 all_atom=True,
                 **kwargs):
        super().__init__(edges, resi_type, **kwargs)
        self.resi_type = resi_type
        self.n_resi = len(resi_type)
        self.view = 'atom' if all_atom else 'resi'

    def __getitem__(self, index):
        # why do we check tuple?
        # case 1: x[0, 1] is parsed as (0, 1)
        # case 2: x[[0, 1]] is parsed as [0, 1]
        if not isinstance(index, tuple):
            index = (index,)

        if len(index) > 1:
            raise ValueError(f"Protein has only 1 axis, but {len(index)} axis is indexed")

        return self.resi_mask(index[0], compact=True)

    def __repr__(self):
        fields = [f"n_atom={self.n_node}, n_bond={self.n_edge}, n_resi={self.n_resi}"]
        return f"{self.__class__.__name__}({fields})"

    @property
    def node_feat(self):
        """node_feat"""
        if getattr(self, "view", "atom") == "atom":
            return self.atom_feat
        return self.resi_feat

    @property
    def resi2graph(self):
        """Residue id to graph id mapping."""
        return np.zeros(self.n_resi, dtype=np.int64)

    @property
    def connected_component_id(self):
        """Connected component id of each residue."""
        node_in, node_out = self.edges.t()[:2]
        resi_in, resi_out = self.atom_resi[node_in], self.atom_resi[node_out]
        mask = resi_in != resi_out
        resi_in, resi_out = resi_in[mask], resi_out[mask]
        order = np.arange(self.n_resi)
        resi_in, resi_out = np.concatenate([resi_in, resi_out, order]), \
            np.concatenate([resi_out, resi_in, order])

        min_neighbor = np.arange(self.n_resi)
        last = np.zeros_like(min_neighbor)
        while not np.equal(min_neighbor, last):
            last = min_neighbor
            min_neighbor = scatter_min(min_neighbor[resi_out], resi_in, n_axis=self.n_resi)[0]
        cc_id = np.unique(min_neighbor, return_inverse=True)[1]
        return cc_id

    @classmethod
    # pylint: disable=W0221
    def from_molecule(cls, mol, atom_feat="default", bond_feat="default", resi_feat="default",
                      mol_feat=None, kekulized=False):
        """
        Create a protein from an RDKit object.

        Parameters:
            mol (rdchem.Mol): molecule
            atom_feat (str or list of str, optional): atom feats to extract
            bond_feat (str or list of str, optional): bond feats to extract
            resi_feat (str, list of str, optional): residue feats to extract
            mol_feat (str or list of str, optional): molecule feats to extract
            kekulized (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edges``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        protein = Molecule.from_molecule(mol, node_feat=atom_feat, edge_feat=bond_feat,
                                         graph_feat=mol_feat, add_hs=False, kekulized=kekulized)
        resi_feat = cls._standarize_option(resi_feat)

        if kekulized:
            Chem.Kekulize(mol)

        resi_type = []
        atom_name = []
        atom_is_hetero = []
        atom_occupancy = []
        atom_bfactor = []
        atom_resi = []
        resi_number = []
        resi_insertion_code = []
        resi_chain = []
        resi_feat_ = []
        last_resi = None
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
        for atom in atoms:
            pdbinfo = atom.GetPDBResidueInfo()
            number = pdbinfo.GetResidueNumber()
            code = pdbinfo.GetInsertionCode()
            rtype = pdbinfo.GetResidueName().strip()
            canonical_resi = (number, code, rtype)
            if canonical_resi != last_resi:
                last_resi = canonical_resi
                if rtype not in cls._resi2id:
                    warnings.warn(f"Unknown residue `{rtype}`. Treat as glycine")
                    rtype = "GLY"
                resi_type.append(cls._resi2id.get(rtype))
                resi_number.append(number)
                if pdbinfo.GetInsertionCode() not in cls._chain2id or pdbinfo.GetChainId() not in cls._chain2id:
                    return None
                resi_insertion_code.append(cls._chain2id.get(pdbinfo.GetInsertionCode()))
                resi_chain.append(cls._chain2id.get(pdbinfo.GetChainId()))
                feat = []
                for name in resi_feat:
                    func = R.get(f"feature.residue.{name}")
                    feat += func(pdbinfo)
                resi_feat_.append(feat)
            name = pdbinfo.GetName().strip()
            if name not in cls._atom2id:
                name = "UNK"
            atom_name.append(cls._atom2id.get(name))
            atom_is_hetero.append(pdbinfo.GetIsHeteroAtom())
            atom_occupancy.append(pdbinfo.GetOccupancy())
            atom_bfactor.append(pdbinfo.GetTempFactor())
            atom_resi.append(len(resi_type) - 1)
        protein.resi_type = np.array(resi_type)
        protein.atom_name = np.array(atom_name)
        protein.atom_is_hetero = np.array(atom_is_hetero)
        protein.atom_occupancy = np.array(atom_occupancy)
        protein.atom_bfactor = np.array(atom_bfactor)
        protein.atom_resi = np.array(atom_resi)
        protein.resi_number = np.array(resi_number)
        protein.resi_insertion_code = np.array(resi_insertion_code)
        protein.resi_chain = np.array(resi_chain)
        if resi_feat:
            protein.resi_feat = np.array(resi_feat_)

        return protein

    @classmethod
    def resi_from_sequence(cls, sequence):
        """resi_from sequence"""
        resi_type = []
        resi_feat = []
        for resi in sequence:
            if resi not in cls._resi_abbr2id:
                warnings.warn(f"Unknown residue symbol `{resi}`. Treat as glycine")
                resi = "G"
            resi_type.append(cls._resi_abbr2id.get(resi))
            resi_feat.append(feature.onehot(resi, cls._resi_abbr2id, allow_unknown=True))

        resi_feat = np.array(resi_feat)

        return cls(edges=None, n_node=0, resi_type=resi_type, resi_feat=resi_feat)

    @classmethod
    def from_sequence(cls, sequence, atom_feat="default", bond_feat="default", resi_feat="default",
                      mol_feat=None, kekulized=False):
        """
        Create a protein from a sequence.

        .. note::

            It takes considerable time to construct proteins with a large number of atoms and bonds.
            If you only need residue information, you may speed up the construction by setting
            ``atom_feat`` and ``bond_feat`` to ``None``.

        Parameters:
            sequence (str): protein sequence
            atom_feat (str or list of str, optional): atom features to extract
            bond_feat (str or list of str, optional): bond features to extract
            resi_feat (str, list of str, optional): residue features to extract
            mol_feat (str or list of str, optional): molecule features to extract
            kekulized (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edges``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if atom_feat is None and bond_feat is None and resi_feat == "default":
            return cls.resi_from_sequence(sequence)

        mol = Chem.MolFromSequence(sequence)
        if mol is None:
            raise ValueError(f"Invalid sequence `{sequence}`")

        return cls.from_molecule(mol, atom_feat, bond_feat, resi_feat, mol_feat, kekulized)

    @classmethod
    def from_pdb(cls, pdb_file, atom_feat="default", bond_feat="default", resi_feat="default",
                 mol_feat=None, kekulized=False):
        """
        Create a protein from a PDB file.

        Parameters:
            pdb_file (str): file name
            atom_feat (str or list of str, optional): atom features to extract
            bond_feat (str or list of str, optional): bond features to extract
            resi_feat (str, list of str, optional): residue features to extract
            mol_feat (str or list of str, optional): molecule features to extract
            kekulized (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edges``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"No such file `{pdb_file}`")
        mol = Chem.MolFromPDBFile(pdb_file)
        if mol is None:
            raise ValueError(f"RDKit cannot read PDB file `{pdb_file}`")
        return cls.from_molecule(mol, atom_feat, bond_feat, resi_feat, mol_feat, kekulized)

    @classmethod
    def pack(cls, graphs):
        """pack"""
        edges = []
        edge_weight = []
        n_nodes = []
        n_edges = []
        n_resis = []
        cum_node = 0
        cum_edge = 0
        cum_resi = 0
        n_graph = 0
        kwargs = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        view = graphs[0].view
        for graph in graphs:
            edges.append(graph.edges)
            edge_weight.append(graph.edge_weight)
            n_nodes.append(graph.n_node)
            n_edges.append(graph.n_edge)
            n_resis.append(graph.n_resi)
            for k, v in graph.kwargs.items():
                for mdtype in meta_dict[k]:
                    if mdtype == "graph":
                        v = v.expand_dims(0)
                    elif mdtype == "node reference":
                        v = np.where(v != -1, v + cum_node, -1)
                    elif mdtype == "edge reference":
                        v = np.where(v != -1, v + cum_edge, -1)
                    elif mdtype == "residue reference":
                        v = np.where(v != -1, v + cum_resi, -1)
                    elif mdtype == "graph reference":
                        v = np.where(v != -1, v + n_graph, -1)
                kwargs[k].append(v)
            cum_node += graph.n_node
            cum_edge += graph.n_edge
            cum_resi += graph.n_resi
            n_graph += 1

        edges = np.concatenate(edges)
        edge_weight = np.concatenate(edge_weight)
        kwargs = {k: np.concatenate(v) for k, v in kwargs.items()}

        return cls.batch_type(edges.T, edge_weight=edge_weight, n_relation=graphs[0].n_relation,
                              n_nodes=n_nodes, n_edges=n_edges, n_resis=n_resis, view=view,
                              **kwargs)

    def to_molecule(self):
        """
        Return an RDKit object of this protein.

        Parameters:
            ignore_error (bool, optional): if true, return ``None`` for illegal molecules.
                Otherwise, raise an exception.

        Returns:
            rdchem.Mol
        """
        mol = super().to_molecule()
        if mol is None:
            return mol

        resi_type = self.resi_type.tolist()
        atom_name = self.atom_name.tolist()
        atom_resi = self.atom_resi.tolist()
        is_hetero_atom = self.atom_is_hetero.tolist()
        occupancy = self.atom_occupancy.tolist()
        b_factor = self.atom_bfactor.tolist()
        resi_number = self.resi_number.tolist()
        chain_id = self.resi_chain.tolist()
        insertion_code = self.resi_insertion_code.tolist()
        for i, atom in enumerate(mol.GetAtoms()):
            r = atom_resi[i]
            resi = Chem.AtomPDBResidueInfo()
            resi.SetResidueNumber(resi_number[r])
            resi.SetChainId(self._id2chain.get(chain_id[r]))
            resi.SetInsertionCode(self._id2chain.get(insertion_code[r]))
            resi.SetName(f" {self._id2atom.get(atom_name[i]):-3s}")
            resi.SetResidueName(self._id2resi.get(resi_type[r]))
            resi.SetIsHeteroAtom(is_hetero_atom[i])
            resi.SetOccupancy(occupancy[i])
            resi.SetTempFactor(b_factor[i])
            atom.SetPDBResidueInfo(resi)

        return mol

    def to_sequence(self):
        """
        Return a sequence of this protein.

        Returns:
            str
        """
        resi_type = self.resi_type.tolist()
        cc_id = self.connected_component_id.tolist()
        sequence = []
        for i in range(self.n_resi):
            if i > 0 and cc_id[i] > cc_id[i - 1]:
                sequence.append(".")
            sequence.append(self._id2resi_abbr.get(resi_type[i]))
        return "".join(sequence)

    def to_pdb(self, pdb_file):
        """
        Write this protein to a pdb file.

        Parameters:
            pdb_file (str): file name
        """
        mol = self.to_molecule()
        Chem.MolToPDBFile(mol, pdb_file, flavor=10)

    def split(self, node2graph):
        """split"""
        # coalesce arbitrary graph IDs to [0, n)
        _, node2graph = np.unique(node2graph, return_inverse=True)
        n_graph = node2graph.max() + 1
        index = node2graph.argsort()
        mapping = np.zeros_like(index)
        mapping[index] = np.arange(len(index))

        node_in, node_out = self.edges
        edge_mask = node2graph[node_in] == node2graph[node_out]
        edge2graph = node2graph[node_in]
        edge_index = edge2graph.argsort()
        edge_index = edge_index[edge_mask[edge_index]]

        prepend = np.array([-1])
        is_first_node = np.diff(node2graph[index], prepend=prepend) > 0
        graph_index = self.node2graph[index[is_first_node]]

        # a residue can be split into multiple graphs
        max_n_node = node2graph.bincount(minlength=n_graph).max()
        key = node2graph[index] * max_n_node + self.atom_resi.get(index)
        key_set, atom_resi = key.unique(return_inverse=True)
        resi_index = key_set % max_n_node

        edges = self.edges.clone()
        edges = mapping[edges]

        n_nodes = node2graph.bincount(minlength=n_graph)
        n_edges = edge2graph[edge_index].bincount(minlength=n_graph)
        cum_resis = scatter_max(atom_resi, node2graph[index], n_axis=n_graph)[0] + 1
        prepend = np.array([0])
        n_resis = np.diff(cum_resis, prepend=prepend)

        cum_nodes = n_nodes.cumsum(0)
        offsets = (cum_nodes - n_nodes)[edge2graph[edge_index]]

        kwargs = self.data_mask(index, edge_index, resi_index, graph_index)

        return self.batch_type(edges[:, edge_index], edge_weight=self.edge_weight[edge_index],
                               n_nodes=n_nodes, n_edges=n_edges, n_resis=n_resis, view=self.view,
                               offsets=offsets, atom_resi=atom_resi, **kwargs)

    def repeat(self, count):
        """repeat"""
        edges = self.edges.repeat(count, 1)
        edge_weight = self.edge_weight.repeat(count)
        n_nodes = [self.n_node] * count
        n_edges = [self.n_edge] * count
        n_resis = [self.n_resi] * count
        n_relation = self.n_relation

        kwargs = {}
        for k, v in self.kwargs.items():
            if "graph" in self.meta_dict[k]:
                v = v.expand_dims(0)
            shape = [1] * v.ndim
            shape[0] = count
            length = len(v)
            v = v.repeat(shape)
            for mdtype in self.meta_dict[k]:
                if mdtype == "node reference":
                    offsets = np.arange(count) * self.n_node
                    v += offsets.preat(length)
                elif mdtype == "edge reference":
                    offsets = np.arange(count) * self.n_edge
                    v = v + offsets.preat(length)
                elif mdtype == "resi reference":
                    offsets = np.arange(count) * self.n_resi
                    v = v + offsets.preat(length)
                elif mdtype == "graph reference":
                    offsets = np.arange(count)
                    v = v + offsets.preat(length)
            kwargs[k] = v

        return self.batch_type(edges, edge_weight=edge_weight,
                               n_nodes=n_nodes, n_edges=n_edges, n_resis=n_resis, view=self.view,
                               n_relation=n_relation, **kwargs)

    def resi2atom(self, resi_index):
        """Map residue ids to atom ids."""
        resi_index = self._standarize_index(resi_index, self.n_resi)
        if not hasattr(self, "node_inverted_index"):
            setattr(self, 'node_inverted_index', self._build_node_inverted_index())
        inverted_range, order = self.node_inverted_index
        starts, ends = inverted_range[resi_index].t()
        n_match = ends - starts
        offsets = n_match.cumsum(0) - n_match
        ranges = np.arange(n_match.sum())
        ranges = ranges + (starts - offsets).preat(n_match)
        index = order[ranges]
        return index

    def data_mask(self, node_idx=None, edge_idx=None, resi_idx=None, graph_idx=None):
        "data mask"
        kwargs = super().data_mask(node_idx, edge_idx, graph_idx=graph_idx)
        if resi_idx is not None:
            resi_map = self._get_mapping(resi_idx, self.n_resi)
            kwargs['rdata'] = {key: value[resi_map] for key, value in self.rdata.items()}
        else:
            kwargs['rdata'] = self.rdata
        return kwargs

    def resi_mask(self, index, compact=False):
        """
        Return a masked protein based on the specified residues.

        Note the compact option is applied to both residue and atom ids.

        Parameters:
            index (array_like): residue index
            compact (bool, optional): compact residue ids or not

        Returns:
            Protein
        """
        index = self._standarize_index(index, self.n_resi)
        if (np.diff(index) <= 0).any():
            warnings.warn("`resi_mask()` is called to re-order the residues. This will change the protein sequence. "
                          "If this is not desired, you might have passed a wrong index to this function.")
        resi_mapping = -np.ones(self.n_resi, dtype=np.int64)
        resi_mapping[index] = np.arange(len(index))

        node_index = resi_mapping[self.atom_resi] >= 0
        node_index = self._standarize_index(node_index, self.n_node)
        mapping = -np.ones(self.n_node, dtype=np.int64)
        if compact:
            mapping[node_index] = np.arange(len(node_index))
            n_node = len(node_index)
        else:
            mapping[node_index] = node_index
            n_node = self.n_node

        edges = self.edges.clone()
        edges[:, :2] = mapping[edges[:, :2]]
        edge_index = (edges[:, :2] >= 0).all(axis=-1)
        edge_index = self._standarize_index(edge_index, self.n_edge)

        if compact:
            kwargs = self.data_mask(node_index, edge_index, resi_idx=index)
        else:
            kwargs = self.data_mask(edge_idx=edge_index)

        return type(self)(edges[edge_index], edge_weight=self.edge_weight[edge_index], n_node=n_node,
                          view=self.view, **kwargs)

    def subresi(self, index):
        """
        Return a subgraph based on the specified residues.
        Equivalent to :meth:`resi_mask(index, compact=True) <resi_mask>`.

        Parameters:
            index (array_like): residue index

        Returns:
            Protein

        See also:
            :meth:`Protein.resi_mask`
        """
        return self.resi_mask(index, compact=True)

    def _check_attribute(self, key, value):
        """_checck_attribute"""
        super()._check_attribute(key, value)
        for dtype in self._meta_contexts:
            if dtype == "resi":
                if len(value) != self.n_resi:
                    raise ValueError(f"Expect residue attribute `{key}` to have shape"
                                     f"({self.n_resi}, *), but found {value.shape}")
            elif dtype == "residue reference":
                is_valid = (value >= -1) & (value < self.n_resi)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError(f"Expect residue reference in [-1, {self.n_residue}), but found {error_value[0]}")

    def _build_node_inverted_index(self):
        """build node inverted index"""
        keys = self.atom_resi
        order = keys.argsort()
        keys_set, n_keys = keys.unique(return_counts=True)
        ends = n_keys.cumsum(0)
        starts = ends - n_keys
        ranges = np.stack([starts, ends], axis=-1)
        inverted_range = dict(zip(keys_set, ranges))
        return inverted_range, order


class ProteinBatch(MoleculeBatch, Protein):
    """
    Container for proteins with variadic sizes.
    Support both residue-level and atom-level operations and ensure consistency between two views.

    .. warning::

        Edges of the same graph are guaranteed to be consecutive in the edge list.
        The order of residues must be the same as the protein sequence.
        However, this class doesn't enforce any order on nodes or edges.
        Nodes may have a different order with residues.

    Parameters:
        edges (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        resi_type (array_like, optional): residue types of shape :math:`(|V_{res}|,)`
        view (str, optional): default view for this protein. Can be ``atom`` or ``residue``.
        n_nodes (array_like, optional): number of nodes in each graph
            By default, it will be inferred from the largest id in `edges`
        n_edges (array_like, optional): number of edges in each graph
        n_resis (array_like, optional): number of residues in each graph
        offsets (array_like, optional): node id offsets of shape :math:`(|E|,)`.
            If not provided, nodes in `edges` should be relative index, i.e., the index in each graph.
            If provided, nodes in `edges` should be absolute index, i.e., the index in the packed graph.
    """

    _check_attribute = Protein._check_attribute

    def __init__(self, edges=None, atom_type=None, bond_type=None, resi_type=None, view=None, n_nodes=None,
                 n_edges=None, n_resis=None, offsets=None, **kwargs):
        super().__init__(edges=edges, n_nodes=n_nodes, n_edges=n_edges,
                         offsets=offsets, atom_type=atom_type, bond_type=bond_type,
                         resi_type=resi_type, view=view, **kwargs)

        cum_resis = n_resis.cumsum(0)

        self.n_resis = n_resis
        self.cum_resis = cum_resis

    def __repr__(self):
        fields = f"batch_size={self.batch_size}, " + \
                 f"n_atoms={pretty.long_array(self.n_nodes.tolist())}, " + \
                 f"n_bonds={pretty.long_array(self.n_edges.tolist())}, " + \
                 f"n_resis={pretty.long_array(self.n_resis.tolist())}, "
        return f"{self.__class__.__name__}({fields})"

    @property
    def resi2graph(self):
        """Residue id to graph id mapping."""
        order = np.arange(self.batch_size)
        resi2graph = order.preat(self.n_resis)
        return resi2graph

    @property
    def connected_component_id(self):
        """connected component id"""
        cc_id = super().connected_component_id
        cc_id_offsets = scatter_min(cc_id, self.resi2graph, n_axis=self.n_resi)[0][self.resi2graph]
        cc_id = cc_id - cc_id_offsets
        return cc_id

    @classmethod
    # pylint: disable=W0221
    def from_molecule(cls, mols, atom_feat="default", bond_feat="default", resi_feat="default",
                      mol_feat=None, kekulized=False):
        """
        Create a packed protein from a list of RDKit objects.

        Parameters:
            mols (list of rdchem.Mol): molecules
            atom_feat (str or list of str, optional): atom features to extract
            bond_feat (str or list of str, optional): bond features to extract
            resi_feat (str or list of str, optional): residue features to extract
            mol_feat (str or list of str, optional): molecule features to extract
            kekulized (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edges``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        protein = MoleculeBatch.from_molecule(mols, node_feat=atom_feat, edge_feat=bond_feat,
                                              graph_feat=mol_feat, add_hs=False, kekulized=kekulized)
        resi_feat = cls._standarize_option(resi_feat)

        resi_type = []
        atom_name = []
        is_hetero_atom = []
        occupancy = []
        b_factor = []
        atom_resi = []
        resi_number = []
        insertion_code = []
        chain_id = []
        resi_feat_ = []
        last_resi = None
        n_resis = []
        cum_resi = 0

        mols = mols + [cls.dummy_protein]
        for mol in mols:
            if mol is None:
                mol = cls.empty_mol

            if kekulized:
                Chem.Kekulize(mol)

            for atom in mol.GetAtoms():
                resi = atom.GetPDBResidueInfo()
                number = resi.GetResidueNumber()
                code = resi.GetInsertionCode()
                r_type = resi.GetResidueName().strip()
                canonical_resi = (number, code, r_type)
                if canonical_resi != last_resi:
                    last_resi = canonical_resi
                    if r_type not in cls._resi2id:
                        warnings.warn(f"Unknown residue `{type}`. Treat as glycine")
                        r_type = "GLY"
                    resi_type.append(cls._resi2id.get(r_type))
                    resi_number.append(number)
                    insertion_code.append(cls._chain2id.get(resi.GetInsertionCode()))
                    chain_id.append(cls._chain2id.get(resi.GetChainId()))
                    feat = []
                    for name in resi_feat:
                        func = R.get("feature.residue." + name)
                        feat += func(resi)
                    resi_feat_.append(feat)
                name = resi.GetName().strip()
                if name not in cls._atom2id:
                    name = "UNK"
                atom_name.append(cls._atom2id.get(name))
                is_hetero_atom.append(resi.GetIsHeteroAtom())
                occupancy.append(resi.GetOccupancy())
                b_factor.append(resi.GetTempFactor())
                atom_resi.append(len(resi_type) - 1)

            n_resis.append(len(resi_type) - cum_resi)
            cum_resi = len(resi_type)

        resi_type = np.array(resi_type)[:-1]
        atom_name = np.array(atom_name)[:-5]
        is_hetero_atom = np.array(is_hetero_atom)[:-5]
        occupancy = np.array(occupancy)[:-5]
        b_factor = np.array(b_factor)[:-5]
        atom_resi = np.array(atom_resi)[:-5]
        resi_number = np.array(resi_number)[:-1]
        insertion_code = np.array(insertion_code)[:-1]
        chain_id = np.array(chain_id)[:-1]
        if resi_feat:
            resi_feat_ = np.array(resi_feat_)[:-1]
        else:
            resi_feat_ = None

        n_resis = n_resis[:-1]

        return cls(protein.edges, resi_type=resi_type,
                   n_nodes=protein.n_nodes, n_edges=protein.n_edges, n_resis=n_resis,
                   atom_name=atom_name, atom_resi=atom_resi, resi_feat=resi_feat_,
                   is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                   resi_number=resi_number, insertion_code=insertion_code, chain_id=chain_id,
                   offsets=protein.offsets, meta_dict=protein.meta_dict, **protein.caches())

    @classmethod
    def resi_from_sequence(cls, sequence):
        n_resis = []
        resi_type = []
        resi_feat = []
        sequence = sequence + ["G"]
        for seq in sequence:
            for resi in seq:
                if resi not in cls._resi_abbr2id:
                    warnings.warn(f"Unknown residue symbol `{resi}`. Treat as glycine")
                    resi = "G"
                resi_type.append(cls._resi_abbr2id[resi])
                resi_feat.append(feature.onehot(resi, cls._resi_abbr2id, allow_unknown=True))
            n_resis.append(len(seq))

        resi_type = resi_type[:-1]
        resi_feat = np.array(resi_feat)[:-1]

        edges = np.zeros((0, 3), dtype=np.int64)
        n_nodes = [0] * (len(sequence) - 1)
        n_edges = [0] * (len(sequence) - 1)
        n_resis = n_resis[:-1]

        return cls(edges=edges, atom_type=[], bond_type=[], resi_type=resi_type,
                   n_nodes=n_nodes, n_edges=n_edges, n_resis=n_resis,
                   resi_feat=resi_feat)

    @classmethod
    def from_sequence(cls, sequence, atom_feat="default", bond_feat="default", resi_feat="default",
                      mol_feat=None, kekulized=False):
        """
        Create a packed protein from a list of sequences.

        .. note::

            It takes considerable time to construct proteins with a large number of atoms and bonds.
            If you only need residue information, you may speed up the construction by setting
            ``atom_feat`` and ``bond_feat`` to ``None``.

        Parameters:
            sequence (str): list of protein sequences
            atom_feat (str or list of str, optional): atom features to extract
            bond_feat (str or list of str, optional): bond features to extract
            resi_feat (str or list of str, optional): residue features to extract
            mol_feat (str or list of str, optional): molecule features to extract
            kekulized (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edges``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if atom_feat is None and bond_feat is None and resi_feat == "default":
            return cls.resi_from_sequence(sequence)

        mols = []
        for seq in sequence:
            mol = Chem.MolFromSequence(seq)
            if mol is None:
                raise ValueError(f"Invalid sequence `{seq}`")
            mols.append(mol)

        return cls.from_molecule(mols, atom_feat, bond_feat, resi_feat, mol_feat, kekulized)

    @classmethod
    def from_pdb(cls, pdb_file, atom_feat="default", bond_feat="default", resi_feat="default",
                 mol_feat=None, kekulized=False):
        """
        Create a protein from a list of PDB files.

        Parameters:
            pdb_file (str): list of file names
            atom_feat (str or list of str, optional): atom features to extract
            bond_feat (str or list of str, optional): bond features to extract
            resi_feat (str, list of str, optional): residue features to extract
            mol_feat (str or list of str, optional): molecule features to extract
            kekulized (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edges``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        mols = []
        for pfile in pdb_file:
            mol = Chem.MolFromPDBFile(pfile)
            mols.append(mol)

        return cls.from_molecule(mols, atom_feat, bond_feat, resi_feat, mol_feat, kekulized)

    def to_molecule(self):
        """to_molecule"""
        mols = super().to_molecule()

        resi_type = self.resi_type.tolist()
        atom_name = self.atom_name.tolist()
        atom_resi = self.atom_resi.tolist()
        is_hetero_atom = self.atom_is_hetero.tolist()
        occupancy = self.atom_occupancy.tolist()
        b_factor = self.atom_bfactor.tolist()
        resi_number = self.resi_number.tolist()
        chain_id = self.resi_chain.tolist()
        insertion_code = self.resi_insertion_code.tolist()
        cum_nodes = [0] + self.cum_nodes.tolist()

        for i, mol in enumerate(mols):
            for j, atom in enumerate(mol.GetAtoms(), cum_nodes[i]):
                r = atom_resi[j]
                resi = Chem.AtomPDBResidueInfo()
                resi.SetResidueNumber(resi_number[r])
                resi.SetChainId(self._id2chain[chain_id[r]])
                resi.SetInsertionCode(self._id2chain[insertion_code[r]])
                resi.SetName(f" {self._id2atom[atom_name[j]]:-3s}")
                resi.SetResidueName(self._id2resi[resi_type[r]])
                resi.SetIsHeteroAtom(is_hetero_atom[j])
                resi.SetOccupancy(occupancy[j])
                resi.SetTempFactor(b_factor[j])
                atom.SetPDBResidueInfo(resi)

        return mols

    def to_sequence(self):
        """
        Return a list of sequences.

        Returns:
            list of str
        """
        resi_type = self.resi_type.tolist()
        cc_id = self.connected_component_id.tolist()
        cum_resis = [0] + self.cum_resis.tolist()
        sequences = []
        for i in range(self.batch_size):
            sequence = []
            for j in range(cum_resis[i], cum_resis[i + 1]):
                if j > cum_resis[i] and cc_id[j] > cc_id[j - 1]:
                    sequence.append(".")
                sequence.append(self._id2resi_abbr.get(resi_type[j]))
            sequence = "".join(sequence)
            sequences.append(sequence)
        return sequences

    def to_pdb(self, pdb_file):
        """
        Write this packed protein to several pdb files.

        Parameters:
            pdb_files (list of str): list of file names
        """
        mols = self.to_molecule()
        for mol, pfile in zip(mols, pdb_file):
            Chem.MolToPDBFile(mol, pfile, flavor=10)

    def merge(self, graph2graph):
        """merge"""
        # coalesce arbitrary graph IDs to [0, n)
        _, graph2graph = np.unique(graph2graph, return_inverse=True)

        graph_key = graph2graph * self.batch_size + np.arange(self.batch_size)
        graph_index = graph_key.argsort()
        graph = self.subbatch(graph_index)
        graph2graph = graph2graph[graph_index]

        n_graph = graph2graph[-1] + 1
        n_nodes = scatter_add(graph.n_nodes, graph2graph, n_axis=n_graph)
        n_edges = scatter_add(graph.n_edges, graph2graph, n_axis=n_graph)
        n_resis = scatter_add(graph.n_resis, graph2graph, n_axis=n_graph)
        offsets = self._get_offsets(n_nodes, n_edges)

        return type(self)(graph.edges, edge_weight=graph.edge_weight, n_nodes=n_nodes,
                          n_edges=n_edges, n_resis=n_resis, view=self.view, offsets=offsets)

    def repeat(self, count):
        """repeat"""
        n_nodes = self.n_nodes.repeat(count)
        n_edges = self.n_edges.repeat(count)
        n_resis = self.n_resis.repeat(count)
        offsets = self._get_offsets(n_nodes, n_edges)
        edges = self.edges.repeat(count, 1)
        edges[:, :2] += (offsets - self._offsets.repeat(count)).expand_dims(-1)

        kwargs = {}
        for k, v in self.caches().items():
            shape = [1] * v.ndim
            shape[0] = count
            length = len(v)
            v = v.repeat(shape)
            for type_ in self.meta_dict[k]:
                if type_ == "node reference":
                    pack_offsets = np.arange(count) * self.n_node
                    v = v + pack_offsets.preat(length)
                elif type_ == "edge reference":
                    pack_offsets = np.arange(count) * self.n_edge
                    v = v + pack_offsets.preat(length)
                elif type_ == "resi reference":
                    pack_offsets = np.arange(count) * self.n_resi
                    v = v + pack_offsets.preat(length)
                elif type_ == "graph reference":
                    pack_offsets = np.arange(count) * self.batch_size
                    v = v + pack_offsets.preat(length)
            kwargs[k] = v

        return type(self)(edges, edge_weight=self.edge_weight.repeat(count),
                          n_nodes=n_nodes, n_edges=n_edges, n_resis=n_resis, view=self.view,
                          n_relation=self.n_relation, offsets=offsets, **kwargs)

    def repeat_interleave(self, repeats):
        """repeat interleave"""
        if repeats.numel() == 1:
            repeats = repeats * np.ones(self.batch_size, dtype=np.int64)
        n_nodes = self.n_nodes.preat(repeats)
        n_edges = self.n_edges.preat(repeats)
        n_resis = self.n_resis.preat(repeats)
        cum_nodes = n_nodes.cumsum(0)
        cum_edges = n_edges.cumsum(0)
        cum_resis = n_resis.cumsum(0)
        n_node = n_nodes.sum()
        n_edge = n_edges.sum()
        n_resi = n_resis.sum()
        batch_size = repeats.sum()
        n_graphs = np.ones(batch_size)

        # special case 1: graphs[i] may have no node or no edge
        # special case 2: repeats[i] may be 0
        cum_repeats_shifted = repeats.cumsum(0) - repeats
        graph_mask = cum_repeats_shifted < batch_size
        cum_repeats_shifted = cum_repeats_shifted[graph_mask]

        index = cum_nodes - n_nodes
        index = np.concatenate([index, index[cum_repeats_shifted]])
        value = np.concatenate([-n_nodes, self.n_nodes[graph_mask]])
        mask = index < n_node
        node_index = scatter_add(value[mask], index[mask], n_axis=n_node)
        node_index = (node_index + 1).cumsum(0) - 1

        index = cum_edges - n_edges
        index = np.concatenate([index, index[cum_repeats_shifted]])
        value = np.concatenate([-n_edges, self.n_edges[graph_mask]])
        mask = index < n_edge
        edge_index = scatter_add(value[mask], index[mask], n_axis=n_edge)
        edge_index = (edge_index + 1).cumsum(0) - 1

        index = cum_resis - n_resis
        index = np.concatenate([index, index[cum_repeats_shifted]])
        value = np.concatenate([-n_resis, self.n_resis[graph_mask]])
        mask = index < n_resi
        resi_index = scatter_add(value[mask], index[mask], n_axis=n_resi)
        resi_index = (resi_index + 1).cumsum(0) - 1

        graph_index = np.preat(repeats)

        offsets = self._get_offsets(n_nodes, n_edges)
        edges = self.edges[edge_index]
        edges[:, :2] += (offsets - self._offsets[edge_index]).expand_dims(-1)

        node_offsets = None
        edge_offsets = None
        resi_offsets = None
        graph_offsets = None
        kwargs = {}
        for k, v in self.caches().items():
            n_xs = None
            pack_offsets = None
            for type_ in self.meta_dict[k]:
                if type_ == "node":
                    v = v[node_index]
                    n_xs = n_nodes
                elif type_ == "edge":
                    v = v[edge_index]
                    n_xs = n_edges
                elif type_ == "resi":
                    v = v[resi_index]
                    n_xs = n_resis
                elif type_ == "graph":
                    v = v[graph_index]
                    n_xs = n_graphs
                elif type_ == "node reference":
                    if node_offsets is None:
                        node_offsets = self._get_repeat_pack_offsets(self.n_nodes, repeats)
                    pack_offsets = node_offsets
                elif type_ == "edge reference":
                    if edge_offsets is None:
                        edge_offsets = self._get_repeat_pack_offsets(self.n_edges, repeats)
                    pack_offsets = edge_offsets
                elif type_ == "resi reference":
                    if resi_offsets is None:
                        resi_offsets = self._get_repeat_pack_offsets(self.n_resis, repeats)
                    pack_offsets = resi_offsets
                elif type_ == "graph reference":
                    if graph_offsets is None:
                        graph_offsets = self._get_repeat_pack_offsets(n_graphs, repeats)
                    pack_offsets = graph_offsets
            # add offsets to make references point to indexes in their own graph
            if n_xs is not None and pack_offsets is not None:
                v = v + pack_offsets.preat(n_xs)
            kwargs[k] = v

        return type(self)(edges, edge_weight=self.edge_weight[edge_index],
                          n_nodes=n_nodes, n_edges=n_edges, n_resis=n_resis, view=self.view,
                          n_relation=self.n_relation, offsets=offsets, **kwargs)

    def undirected(self, add_inverse=True):
        """undirected"""
        undirected = MoleculeBatch.undirected(self, add_inverse=add_inverse)

        return type(self)(undirected.edges, edge_weight=undirected.edge_weight,
                          n_nodes=undirected.n_nodes, n_edges=undirected.n_edges,
                          n_resis=self.n_resis, view=self.view, n_relation=undirected.n_relation,
                          offsets=undirected.offsets, **undirected.caches())

    def node_mask(self, index, compact=True):
        """node_mask"""
        index = self._standarize_index(index, self.n_node)
        mapping = -np.ones(self.n_node, dtype=np.int64)
        if compact:
            mapping[index] = np.arange(len(index))
            n_nodes = self._get_n_xs(index, self.cum_nodes)
            offsets = self._get_offsets(n_nodes, self.n_edges)
        else:
            mapping[index] = index
            n_nodes = self.n_nodes
            offsets = self._offsets

        edges = self.edges.clone()
        edges[:, :2] = mapping[edges[:, :2]]
        edge_index = (edges[:, :2] >= 0).all(axis=-1)
        n_edges = self._get_n_xs(edge_index, self.cum_edges)

        if compact:
            kwargs = self.data_mask(index, edge_index)
        else:
            kwargs = self.data_mask(edge_idx=edge_index)

        return type(self)(edges[edge_index], edge_weight=self.edge_weight[edge_index],
                          n_nodes=n_nodes, n_edges=n_edges, n_resis=self.n_resis,
                          view=self.view, n_relation=self.n_relation, offsets=offsets[edge_index],
                          **kwargs)

    def edge_mask(self, index):
        """edge_mask"""
        index = self._standarize_index(index, self.n_edge)
        kwargs = self.data_mask(edge_idx=index)
        n_edges = self._get_n_xs(index, self.cum_edges)

        return type(self)(self.edges[index], edge_weight=self.edge_weight[index],
                          n_nodes=self.n_nodes, n_edges=n_edges, n_resis=self.n_resis,
                          view=self.view, n_relation=self.n_relation, offsets=self._offsets[index],
                          **kwargs)

    def resi_mask(self, index, compact=False):
        """
        Return a masked packed protein based on the specified residues.

        Note the compact option is applied to both residue and atom ids, but not graph ids.

        Parameters:
            index (array_like): residue index
            compact (bool, optional): compact residue ids or not

        Returns:
            PackedProtein
        """
        index = self._standarize_index(index, self.n_resi)
        resi_mapping = -np.ones(self.n_resi, dtype=np.int64)
        resi_mapping[index] = np.arange(len(index))

        node_index = resi_mapping[self.atom_resi] >= 0
        node_index = self._standarize_index(node_index, self.n_node)
        mapping = -np.ones(self.n_node, dtype=np.int64)
        if compact:
            mapping[node_index] = np.arange(len(node_index))
            n_nodes = self._get_n_xs(node_index, self.cum_nodes)
            n_resis = self._get_n_xs(index, self.cum_resis)
        else:
            mapping[node_index] = node_index
            n_nodes = self.n_nodes
            n_resis = self.n_resis

        edges = self.edges.clone()
        edges[:, :2] = mapping[edges[:, :2]]
        edge_index = (edges[:, :2] >= 0).all(axis=-1)
        edge_index = self._standarize_index(edge_index, self.n_edge)
        n_edges = self._get_n_xs(edge_index, self.cum_edges)
        offsets = self._get_offsets(n_nodes, n_edges)

        if compact:
            kwargs = self.data_mask(node_index, edge_index, resi_idx=index)
        else:
            kwargs = self.data_mask(edge_idx=edge_index)

        return type(self)(edges[edge_index], edge_weight=self.edge_weight[edge_index],
                          n_nodes=n_nodes, n_edges=n_edges, n_resis=n_resis,
                          view=self.view, n_relation=self.n_relation, offsets=offsets,
                          **kwargs)

    def graph_mask(self, index, compact=False):
        """graph_mask"""
        index = self._standarize_index(index, self.batch_size)
        graph_mapping = -np.ones(self.batch_size, dtype=np.int64)
        graph_mapping[index] = np.arange(len(index))

        node_index = graph_mapping[self.node2graph] >= 0
        node_index = self._standarize_index(node_index, self.n_node)
        resi_index = graph_mapping[self.resi2graph] >= 0
        resi_index = self._standarize_index(resi_index, self.n_resi)
        mapping = -np.ones(self.n_node, dtype=np.int64)
        if compact:
            key = graph_mapping[self.node2graph[node_index]] * self.n_node + node_index
            order = key.argsort()
            node_index = node_index[order]
            key = graph_mapping[self.resi2graph[resi_index]] * self.n_resi + resi_index
            order = key.argsort()
            resi_index = resi_index[order]
            mapping[node_index] = np.arange(len(node_index))
            n_nodes = self.n_nodes[index]
            n_resis = self.n_resis[index]
        else:
            mapping[node_index] = node_index
            n_nodes = np.zeros_like(self.n_nodes)
            n_nodes[index] = self.n_nodes[index]
            n_resis = np.zeros_like(self.n_resis)
            n_resis[index] = self.n_resis[index]

        edges = self.edges.clone()
        edges[:, :2] = mapping[edges[:, :2]]
        edge_index = (edges[:, :2] >= 0).all(axis=-1)
        edge_index = self._standarize_index(edge_index, self.n_edge)
        if compact:
            key = graph_mapping[self.edge2graph[edge_index]] * self.n_edge + edge_index
            order = key.argsort()
            edge_index = edge_index[order]
            n_edges = self.n_edges[index]
        else:
            n_edges = np.zeros_like(self.n_edges)
            n_edges[index] = self.n_edges[index]
        offsets = self._get_offsets(n_nodes, n_edges)

        if compact:
            kwargs = self.data_mask(node_index, edge_index,
                                    resi_idx=resi_index, graph_idx=index)
        else:
            kwargs = self.data_mask(edge_idx=edge_index)

        return type(self)(edges[edge_index], edge_weight=self.edge_weight[edge_index],
                          n_nodes=n_nodes, n_edges=n_edges, n_resis=n_resis,
                          view=self.view, n_relation=self.n_relation, offsets=offsets,
                          **kwargs)

    def get_item(self, index):
        """get_item"""
        node_index = np.arange(self.cum_nodes[index] - self.n_nodes[index], self.cum_nodes[index],
                               device=self.device)
        edge_index = np.arange(self.cum_edges[index] - self.n_edges[index], self.cum_edges[index],
                               device=self.device)
        resi_index = np.arange(self.cum_resis[index] - self.n_resis[index],
                               self.cum_resis[index])
        graph_index = index
        edges = self.edges[edge_index].clone()
        edges[:, :2] -= self._offsets[edge_index].expand_dims(-1)
        kwargs = self.data_mask(node_index, edge_index,
                                resi_idx=resi_index, graph_idx=graph_index)

        return self.item_type(edges, edge_weight=self.edge_weight[edge_index], n_node=self.n_nodes[index],
                              n_relation=self.n_relation, **kwargs)


Protein.batch_type = ProteinBatch
ProteinBatch.item_type = Protein
