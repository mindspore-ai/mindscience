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
molecule
"""
import warnings
import logging

import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from openbabel import pybel

from ..core import Registry as R
from .graph import GraphBatch, Graph
from .. import data
from .. import util
from ..util.geometry import rigid_transform_kabsch_3d

logger = logging.getLogger(__name__)
plt.switch_backend('agg')


@R.register('data.Molecule')
class Molecule(Graph):
    """_summary_

    Args:
        edges (_type_, optional): _description_. Defaults to None.
        ndata (_type_, optional): _description_. Defaults to None.
        edata (_type_, optional): _description_. Defaults to None.
        gdata (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    _bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
    _atom2val = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 7}
    _bond2val = [1, 2, 3, 1.5]
    _id2bond = {v: k for k, v in _bond2id.items()}
    _empty_mol = Chem.MolFromSmiles("")
    _dummy_mol = Chem.MolFromSmiles("**")
    _dummy_atom = _dummy_mol.GetAtomWithIdx(0)
    _dummy_bond = _dummy_mol.GetBondWithIdx(0)
    _caches = {'atom_': 'ndata', 'bond_': 'edata', 'mol_': 'gdata',
               'node_': 'ndata', 'edge_': 'edata', 'graph_': 'gdata'}

    def __init__(self,
                 edges=None,
                 ndata=None,
                 edata=None,
                 gdata=None,
                 **kwargs) -> None:
        if 'n_relation' not in kwargs:
            kwargs['n_relation'] = len(self._bond2id)
        super().__init__(edges=edges, ndata=ndata, edata=edata, gdata=gdata, **kwargs)

    def __eq__(self, other):
        smiles = self.to_smiles(isomeric=False, atom_map=False, canonical=True)
        other_smiles = other.to_smiles(isomeric=False, atom_map=False, canonical=True)
        return smiles == other_smiles

    @classmethod
    def from_molecule(cls,
                      mol,
                      node_feat=None,
                      edge_feat=None,
                      graph_feat=None,
                      add_hs=False,
                      kekulized=False,
                      with_isotope=False):
        """_summary_

        Args:
            mol (_type_): _description_
            node_feat (_type_, optional): _description_. Defaults to None.
            edge_feat (_type_, optional): _description_. Defaults to None.
            graph_feat (_type_, optional): _description_. Defaults to None.
            add_hs (bool, optional): _description_. Defaults to False.
            kekulized (bool, optional): _description_. Defaults to False.
            with_isotope (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if mol is None:
            mol = cls._empty_mol

        if add_hs:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)
        if kekulized:
            Chem.Kekulize(mol)

        node_feat = cls._standarize_option(node_feat)
        edge_feat = cls._standarize_option(edge_feat)
        graph_feat = cls._standarize_option(graph_feat)

        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
        ndata = {}
        ndata['type'] = np.zeros(len(atoms), dtype=int)
        ndata['charge'] = np.zeros(len(atoms), dtype=int)
        ndata['explicit_h'] = np.zeros(len(atoms))
        ndata['chiral_tag'] = np.zeros(len(atoms), dtype=int)
        ndata['radical_electrons'] = np.zeros(len(atoms))
        ndata['map'] = np.zeros(len(atoms))
        ndata['isotope'] = np.zeros(len(atoms), dtype=int) if with_isotope else None
        node_feat_ = []
        for i, atom in enumerate(atoms):
            ndata['type'][i] = atom.GetAtomicNum()
            ndata['charge'][i] = atom.GetFormalCharge()
            ndata['explicit_h'][i] = atom.GetNumExplicitHs()
            ndata['chiral_tag'][i] = atom.GetChiralTag()
            ndata['radical_electrons'][i] = atom.GetNumRadicalElectrons()
            ndata['map'][i] = atom.GetAtomMapNum()

            if with_isotope:
                ndata['isotope'][i] = atom.GetIsotope()
            feat = []
            for name in node_feat:
                func = R.get("feature.atom." + name)
                feat += func(atom)
            node_feat_.append(feat)

        if node_feat_ and node_feat:
            ndata['feat'] = np.array(node_feat_)

        if mol.GetNumConformers() > 0:
            ndata['coord'] = mol.GetConformer().GetPositions()

        bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())]
        edges = np.zeros((len(bonds) * 2, 2), dtype=int)
        edata = {}
        edata['type'] = np.zeros(len(bonds) * 2, dtype=int)
        edata['stereo'] = np.zeros(len(bonds) * 2, dtype=int)
        edata['dir'] = np.zeros(len(bonds) * 2, dtype=int)
        edata['stereo_atom'] = np.zeros((len(bonds) * 2, 2), dtype=int)
        edge_feat_ = []

        for i, bond in enumerate(bonds):
            btype = str(bond.GetBondType())
            stereo = bond.GetStereo()
            bond_dir = bond.GetBondDir()
            atoms_ = bond.GetStereoAtoms().tolist()
            if len(atoms_) != 2:
                atoms_ = [0, 0]
            assert btype in cls._bond2id
            btype = cls._bond2id.get(btype)
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges[2*i:2*i+2] = [[h, t], [t, h]]
            # always explicitly store aromatic bonds, no matter kekulize or not
            if bond.GetIsAromatic():
                btype = cls._bond2id.get("AROMATIC")
            edata['type'][2*i:2*i+2] = [btype, btype]
            edata['stereo'][2*i:2*i+2] = [stereo, stereo]
            edata['dir'][2*i:2*i+2] = [bond_dir, bond_dir]
            edata['stereo_atom'][2*i:2*i+2] = [atoms_, atoms_]
            feat = []
            for name in edge_feat:
                func = R.get("feature.bond." + name)
                feat += func(bond)
            edge_feat_ += [feat, feat]
        if edge_feat_ and edge_feat:
            edata['feat'] = np.array(edge_feat_)

        graph_feat_ = []
        for name in graph_feat:
            func = R.get("feature.mol." + name)
            graph_feat_ += func(mol)
        if graph_feat_ and graph_feat:
            gdata = {'feat': np.array(graph_feat_)}
        else:
            gdata = None

        n_relation = len(cls._bond2id) - 1 if kekulized else len(cls._bond2id)
        molecule = cls(edges.T, ndata=ndata, edata=edata, gdata=gdata,
                       n_node=mol.GetNumAtoms(), n_relation=n_relation)
        return molecule

    @classmethod
    def rd_from_file(cls, mol_file, sanitize=False, calc_charges=False, hydrogen_mode=False):
        """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.
        Parameters
        ----------
        mol_file : str
            Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
            or ``.pdbqt`` or ``.pdb``.
        sanitize : bool
            Whether sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to False.
        calc_charges : bool
            Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
            ``sanitize`` to be True. Default to False.
        remove_hs : bool
            Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
            slow for large molecules. Default to False.
        use_conformation : bool
            Whether we need to extract molecular conformation from proteins and ligands.
            Default to True.
        Returns
        -------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the loaded molecule.
        coordinates : np.ndarray of shape (N, 3) or None
            The 3D coordinates of atoms in the molecule. N for the number of atoms in
            the molecule. None will be returned if ``use_conformation`` is False or
            we failed to get conformation information.
        """
        if mol_file.endswith('.mol2'):
            mol = Chem.MolFromMol2File(mol_file, sanitize=False)
        elif mol_file.endswith('.sdf'):
            supplier = Chem.SDMolSupplier(mol_file, sanitize=False)
            mol = supplier[0]
        elif mol_file.endswith('.pdbqt'):
            with open(mol_file, 'r') as file:
                lines = file.readlines()
                pdb_block = '\n'.join([line[:66] for line in lines])
            mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False)
        elif mol_file.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(mol_file, sanitize=False)
        else:
            return ValueError(f'Expect the format of the mol_file to be \
                              one of .mol2, .sdf, .pdbqt and .pdb, got {mol_file}')

        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)
        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except RuntimeError:
                warnings.warn('Unable to compute charges for the molecule.')
        if hydrogen_mode in ['none', None]:
            pass
        elif hydrogen_mode == 'remove':
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
        elif hydrogen_mode == 'polar':
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1 and [x.GetAtomicNum() for x in atom.GetNeighbors()] == [6]:
                    atom.SetAtomicNum(0)
            mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#0]'))
        elif hydrogen_mode == 'all':
            mol = Chem.AddHs(mol)
        else:
            raise ValueError(
                f'Unexpected hydrogen mode {hydrogen_mode}, the supported modes contains: all, polar, none, remove')
        Chem.SanitizeMol(mol)
        return mol

    @classmethod
    def ob_from_file(cls, mol_file, hydrogen_mode=False):
        """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.
        Parameters
        ----------
        mol_file : str
            Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
            or ``.pdbqt`` or ``.pdb``.
        sanitize : bool
            Whether sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to False.
        calc_charges : bool
            Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
            ``sanitize`` to be True. Default to False.
        remove_hs : bool
            Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
            slow for large molecules. Default to False.
        use_conformation : bool
            Whether we need to extract molecular conformation from proteins and ligands.
            Default to True.
        Returns
        -------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the loaded molecule.
        coordinates : np.ndarray of shape (N, 3) or None
            The 3D coordinates of atoms in the molecule. N for the number of atoms in
            the molecule. None will be returned if ``use_conformation`` is False or
            we failed to get conformation information.
        """
        fmt = mol_file.split('.')[-1]
        mol = next(pybel.readfile(fmt, mol_file))
        if hydrogen_mode == 'none' or hydrogen_mode is None:
            pass
        elif hydrogen_mode == 'remove':
            mol.removeh()
        elif hydrogen_mode == 'polar':
            mol.addh()
            for atom in mol.atoms:
                if atom.atomicnum == 1:
                    mol.OBMol.DeleteAtom(atom.OBAtom)
        elif hydrogen_mode == 'all':
            mol.addh()
        else:
            raise ValueError(
                f'Unexpected hydrogen mode {hydrogen_mode}, the supported modes contains: all, polar, none, remove')
        return mol

    @classmethod
    def from_smiles(cls,
                    smiles,
                    node_feat='default',
                    edge_feat='default',
                    graph_feat=None,
                    add_hs=False,
                    kekulized=False):
        """_summary_

        Args:
            smiles (_type_): _description_
            node_feat (str, optional): _description_. Defaults to 'default'.
            edge_feat (str, optional): _description_. Defaults to 'default'.
            graph_feat (_type_, optional): _description_. Defaults to None.
            add_hs (bool, optional): _description_. Defaults to False.
            kekulized (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        return cls.from_molecule(mol, node_feat, edge_feat, graph_feat, add_hs, kekulized)

    @classmethod
    def rd_conformer_sampling(cls, mol, n_conf=10, seed=None):
        """_summary_

        Args:
            mol (_type_): _description_
            n_conf (int, optional): _description_. Defaults to 10.
            seed (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        coords = []
        params = AllChem.ETKDGv3()
        warn_msg = 'Warning: rdkit coords could not be generated without using random coords. using random coords now.'
        if seed is not None:
            params.randomSeed = seed
        for _ in range(n_conf):
            m_id = AllChem.EmbedMolecule(mol, params)
            if m_id == -1:
                print(warn_msg)
                params.useRandomCoords = True
                params.useBasicKnowledge = False
                m_id = AllChem.EmbedMolecule(mol, params=params)
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
            coords.append(mol.GetConformer().GetPositions())
        return np.array(coords)

    @classmethod
    def rd_conformer_graph(cls, mol, radius=20, max_neighbors=None, use_rdkit_coord=False, n_confs=10):
        """_summary_

        Args:
            mol (_type_): _description_
            radius (int, optional): _description_. Defaults to 20.
            max_neighbors (_type_, optional): _description_. Defaults to None.
            use_rdkit_coord (bool, optional): _description_. Defaults to False.
            n_confs (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        true_coord = mol.GetConformer().GetPositions()
        all_lig_coord = cls.rd_conformer_sampling(mol, n_conf=n_confs)
        lig_graphs = []
        for i in range(n_confs):
            rot, trans = rigid_transform_kabsch_3d(all_lig_coord[i].T, true_coord.T)
            lig_coord = ((rot @ (all_lig_coord[i]).T).T + trans.squeeze())
            assert lig_coord.shape[1] == 3

            graph = Graph.knn_graph(lig_coord, cutoff=radius, max_neighbors=max_neighbors)
            graph.coord = np.array(true_coord, dtype=np.float32)
            if use_rdkit_coord:
                graph.new_coord = lig_coord
            lig_graphs.append(graph)
        return lig_graphs

    @classmethod
    def ob_conformer_sampling(cls, mol, n_conf=10):
        """_summary_

        Args:
            mol (_type_): _description_
            n_conf (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        coords = []
        ff = ob.OBForceField.FindForceField('uff')
        builder = ob.OBBuilder()
        builder.Build(mol.OBMol)
        ff.Setup(mol.OBMol)
        for _ in range(n_conf):
            ff.RandomRotorSearch(1)
            ff.SteepestDescent(100)
            ff.GetCoordinates(mol.OBMol)
            coords.append([atom.coords for atom in mol.atoms])
        return np.array(coords)

    @classmethod
    def ob_conformer_graph(cls, mol, radius=20, max_neighbors=None, use_ori_coord=False, n_confs=10):
        """_summary_

        Args:
            mol (_type_): _description_
            radius (int, optional): _description_. Defaults to 20.
            max_neighbors (_type_, optional): _description_. Defaults to None.
            use_ori_coord (bool, optional): _description_. Defaults to False.
            n_confs (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        true_coord = np.array([atom.coords for atom in mol.atoms])
        all_lig_coord = cls.ob_conformer_sampling(mol, n_conf=n_confs)
        lig_graphs = []
        for i in range(n_confs):
            rot, trans = rigid_transform_kabsch_3d(all_lig_coord[i].T, true_coord.T)
            lig_coord = ((rot @ (all_lig_coord[i]).T).T + trans.squeeze())
            print('kabsch RMSD between rdkit ligand and true ligand is ',
                  np.sqrt(np.sum((lig_coord - true_coord) ** 2, axis=1).mean()).item())
            assert lig_coord.shape[1] == 3

            graph = Graph.knn_graph(lig_coord, cutoff=radius, max_neighbors=max_neighbors)
            graph.coord = np.array(true_coord, dtype=np.float32)
            if use_ori_coord:
                graph.new_coord = lig_coord
            lig_graphs.append(graph)
        return lig_graphs

    @classmethod
    def is_aromatic_ring(cls, mol, bond_ring):
        """_summary_

        Args:
            mol (_type_): _description_
            bond_ring (_type_): _description_

        Returns:
            _type_: _description_
        """
        for b_id in bond_ring:
            if not mol.GetBondWithIdx(b_id).GetIsAromatic():
                return False
        return True

    @classmethod
    def get_aromatics_rings(cls, mol, atom_idx=None):
        """_summary_

        Args:
            mol (_type_): _description_
            atom_idx (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        atom_rings = mol.GetRingInfo().AtomRings()
        bond_rings = mol.GetRingInfo().BondRings()
        rings = []
        for i, atom_ring in enumerate(atom_rings):
            if not cls.is_aromatic_ring(mol, bond_rings[i]):
                continue
            if atom_idx is None or atom_idx in atom_ring:
                rings.append(atom_idx)
        return rings

    @classmethod
    def geometry_graph(cls, mol, include_rings=True):
        """_summary_

        Args:
            mol (_type_): _description_
            include_rings (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        coord = mol.GetConformer().GetPositions()
        edges = []
        for atom in mol.GetAtoms():
            src_idx = atom.GetIdx()
            neighbors = atom.GetNeighbors().tolist()
            neighbor_idx = [neighbor.GetIdx() for neighbor in neighbors]
            for neighbor in neighbors:
                for hop_neighbor in neighbor.GetNeighbors():
                    neighbor_idx.append(hop_neighbor.GetIdx())
            if include_rings:
                aromatic_rings = cls.get_aromatics_rings(mol, src_idx)
                neighbor_idx += aromatic_rings
            all_dst_idx = list(set(neighbor_idx))
            if not all_dst_idx:
                continue
            all_dst_idx.remove(src_idx)
            edges += [[src_idx, dst] for dst in all_dst_idx]
        graph = Graph(edges=np.array(edges).T, n_node=mol.GetNumAtoms())
        graph.edata = {'_': np.linalg.norm(coord[graph.edges[0]] - coord[graph.edges[1]], axis=1)}
        return graph

    @classmethod
    def masked_angle_graph(cls, mol):
        """_summary_

        Args:
            mol (_type_): _description_

        Returns:
            _type_: _description_
        """
        coord = mol.GetConformer().GetPositions()
        weights = []
        for atom in mol.GetAtoms():
            weights.append(atom.GetAtomicNum())
        weights = np.array(weights)
        mask = []
        angles = []
        edges = []
        distances = []
        for bond in mol.GetBonds():
            b_type = bond.GetBondType()
            src_idx = bond.GetBeginAtomIdx()
            dst_idx = bond.GetEndAtomIdx()
            src = mol.GetAtomWithIdx(src_idx)
            dst = mol.GetAtomWithIdx(dst_idx)
            src_neighbors = [atom.GetIdx() for atom in list(src.GetNeighbors())]
            src_neighbors.remove(dst_idx)
            src_weights = weights[src_neighbors]
            dst_neighbors = [atom.GetIdx() for atom in list(dst.GetNeighbors())]
            dst_neighbors.remove(src_idx)
            dst_weights = weights[dst_neighbors]
            src_to_dst = coord[dst_idx] - coord[src_idx]
            if not (src_neighbors and dst_neighbors) or b_type != Chem.BondType.SINGLE or bond.IsInRing():
                edges.append([src_idx, dst_idx])
                distances.append(np.linalg.norm(src_to_dst))
                mask.append(0)
                angles.append(-1)
                edges.append([dst_idx, src_idx])
                distances.append(np.linalg.norm(src_to_dst))
                mask.append(0)
                angles.append(-1)
                continue
            src_neighbor_coord = coord[src_neighbors]
            dst_neighbor_coord = coord[dst_neighbors]
            src_mean_vec = np.mean(src_neighbor_coord * np.array(src_weights)[:, None] - coord[src_idx], axis=0)
            dst_mean_vec = np.mean(dst_neighbor_coord * np.array(dst_weights)[:, None] - coord[dst_idx], axis=0)
            normal = src_to_dst / np.linalg.norm(src_to_dst)
            src_mean_projection = src_mean_vec - src_mean_vec.dot(normal) * normal
            dst_mean_projection = dst_mean_vec - dst_mean_vec.dot(normal) * normal
            cos_dihedral = src_mean_projection.dot(dst_mean_projection) / (
                np.linalg.norm(src_mean_projection) * np.linalg.norm(dst_mean_projection))
            dihedral_angle = np.arccos(cos_dihedral)
            edges.append([src_idx, dst_idx])
            mask.append(1)
            distances.append(np.linalg.norm(src_to_dst))
            angles.append(dihedral_angle)
            edges.append([dst_idx, src_idx])
            distances.append(np.linalg.norm(src_to_dst))
            mask.append(1)
            angles.append(dihedral_angle)
        return mask, angles

    @classmethod
    def _standarize_option(cls, option):
        """_summary_

        Args:
            option (_type_): _description_

        Returns:
            _type_: _description_
        """
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    def to_molecule(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        mol = Chem.RWMol()
        bond_stereo_atom = self.bond_stereo_atom.tolist()
        if self.coord() is not None:
            coord = self.coord().tolist()
            conformer = Chem.Conformer()
        else:
            conformer = None
        for i in range(self.n_node):
            atom = Chem.Atom(int(self.atom_type[i]))
            atom.SetFormalCharge(int(self.atom_charge[i]))
            atom.SetNumExplicitHs(int(self.atom_explicit_h[i]))
            atom.SetChiralTag(Chem.ChiralType(self.atom_chiral_tag[i]))
            atom.SetNumRadicalElectrons(int(self.atom_radical_electrons[i]))
            atom.SetNoImplicit(bool(self.atom_explicit_h[i] > 0 or self.atom_radical_electrons[i] > 0))
            atom.SetAtomMapNum(int(self.atom_map[i]))
            if self.atom_isotope is not None:
                atom.SetIsotope(int(self.atom_isotope[i]))
            if conformer:
                conformer.SetAtomPosition(i, coord[i])
                mol.AddConformer(conformer)
            mol.AddAtom(atom)

        for i in range(self.n_edge):
            h, t = self.edges[:, i]
            etype = self.edge_type[i]
            if h < t:
                j = mol.AddBond(int(h), int(t), Chem.BondType.names[self._id2bond.get(etype)])
                bond = mol.GetBondWithIdx(j - 1)
                bond.SetIsAromatic(bool(self.bond_type[i] == self._bond2id.get('AROMATIC')))
                bond.SetStereo(Chem.BondStereo(self.bond_stereo[i]))
                bond.SetBondDir(Chem.BondDir(int(self.bond_dir[i])))
        j = 0
        for i in range(self.n_edge):
            h, t = self.edges[:, i]
            etype = self.edge_type[i]
            if h < t:
                if self.bond_stereo[i]:
                    bond = mol.GetBondWithIdx(j)
                    bond.SetStereoAtoms(*bond_stereo_atom[i])
                j += 1
            mol.UpdatePropertyCache()
            Chem.AssignStereochemistry(mol)
            mol.ClearComputedProps()
            mol.UpdatePropertyCache()
        return mol

    def to_smiles(self, isomeric=True, atom_map=True, canonical=False):
        """_summary_

        Args:
            isomeric (bool, optional): _description_. Defaults to True.
            atom_map (bool, optional): _description_. Defaults to True.
            canonical (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        mol = self.to_molecule()
        if not atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
        if canonical:
            smiles_set = set()
            while smiles not in smiles_set:
                smiles_set.add(smiles)
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
        return smiles

    def ion_to_molecule(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        atom_charge = self.ndata.pop('charge')
        atom_explicit_h = self.ndata.pop('explicit_h')
        atom_radical_electrons = self.ndata.pop('radical_elections')
        pos_nitrogen = (self.atom_type == 7) & (self.explicit_val() > 3)
        atom_charge = pos_nitrogen.astype(int)
        atom_explicit_h = np.zeros_like(atom_explicit_h)
        atom_radical_electrons = np.zeros_like(atom_radical_electrons)
        return Molecule(self.edges, n_node=self.n_node, n_relation=self.n_relation,
                        ndata=self.ndata, edata=self.edata, gdata=self.gdata,
                        atom_charge=atom_charge, atom_explicit_h=atom_explicit_h,
                        atom_radical_electrons=atom_radical_electrons)

    def explicit_val(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        explicit_val = util.scatter_add(self._bond2val.get(self.edge_type), self.edges[0], n_axis=self.n_node)
        return np.round(explicit_val).astype(int)

    def is_valid(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        atom2val = np.full(data.NUM_ATOM, np.NaN)
        for k, v in self._atom2val.items():
            atom2val[k] = v
        max_atom_val = atom2val[self.atom_type]

        pos_nitrogen = (self.atom_type == 7) & (self.atom_charge == 1)
        max_atom_val[pos_nitrogen] = 4
        # if np.any(np.isnan(max_atom_val)):
        is_valid = (self.explicit_val() <= max_atom_val).all()
        return is_valid

    def is_valid_rdkit(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        try:
            mol = self.to_molecule()
            Chem.SanitizeMol(mol)
            is_valid = True
        except RuntimeError:
            is_valid = False
        return is_valid


@R.register('data.MoleculeBatch')
class MoleculeBatch(GraphBatch, Molecule):
    """_summary_

        Args:
            edges (_type_, optional): _description_. Defaults to None.
            n_nodes (_type_, optional): _description_. Defaults to None.
            n_edges (_type_, optional): _description_. Defaults to None.
            offsets (_type_, optional): _description_. Defaults to None.
    """
    @classmethod
    def from_molecule(cls,
                      mol,
                      node_feat=None,
                      edge_feat=None,
                      graph_feat=None,
                      add_hs=False,
                      kekulized=False,
                      with_isotope=False):
        """_summary_

        Args:
            mols (_type_): _description_

        Returns:
            _type_: _description_
        """
        mol_graphs = [super().from_molecule(m, node_feat=node_feat, edge_feat=edge_feat, graph_feat=graph_feat,
                                            add_hs=add_hs, kekulized=kekulized, with_isotope=with_isotope)
                      for m in mol]
        return cls.pack(mol_graphs)

    @classmethod
    def from_smiles(cls, smiles, node_feat=None, edge_feat=None, graph_feat=None,
                    add_hs=False, kekulized=False):
        """
        Create a packed molecule from a list of SMILES strings.

        Parameters:
            smiles_list (str): list of SMILES strings
            node_feat (str or list of str, optional): atom features to extract
            edge_feat (str or list of str, optional): bond features to extract
            graph_feat (str or list of str, optional): molecule features to extract
            add_hs (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulized (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edges``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        mols = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            assert mol is not None
            mols.append(mol)
        return cls.from_molecule(mols, node_feat=node_feat, edge_feat=edge_feat,
                                 graph_feat=graph_feat, add_hs=add_hs, kekulized=kekulized)

    def ion_to_molecule(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        atom_charge = self.ndata.pop('charge')
        atom_explicit_h = self.ndata.pop('explicit_h')
        atom_radical_electrons = self.ndata.pop('radical_electrons')
        three = self.explicit_val() > 3
        seven = self.atom_type == 7
        pos_nitrogen = three & seven
        atom_charge = pos_nitrogen.astype(int)
        atom_explicit_h = np.zeros_like(atom_explicit_h)
        atom_radical_electrons = np.zeros_like(atom_radical_electrons)

        return GraphBatch(self.edges, ndata=self.ndata, edata=self.edata, gdata=self.gdata,
                          n_edges=self.n_nodes, n_relation=self.n_relation,
                          offsets=self.offsets, atom_charge=atom_charge, atom_explicit_h=atom_explicit_h,
                          atom_radical_electrons=atom_radical_electrons)

    def is_valid(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        atom2val = np.full(data.NUM_ATOM, np.NaN)
        for k, v in self._atom2val.items():
            atom2val[k] = v
        max_atom_val = atom2val[self.atom_type]
        pos_nitrogen = (self.atom_type == 7) & (self.atom_charge == 1)
        max_atom_val[pos_nitrogen] = 4
        assert not np.any(np.isnan(max_atom_val))
        is_valid = self.explicit_val() <= max_atom_val
        is_valid = is_valid.astype(int)
        is_valid = util.scatter_min(is_valid, self.node2graph, n_axis=self.batch_size)[0].astype(np.bool8)
        return is_valid

    def is_valid_rdkit(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return np.concatenate([mol.is_valid_rdkit for mol in self])

    def to_molecule(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        cum_nodes = [0] + self.cum_nodes.tolist()
        cum_edges = [0] + self.cum_edges.tolist()
        edges = self.edges.copy()
        edges -= self.offsets

        mols = []
        for i in range(self.batch_size):
            mol = Chem.RWMol()
            conformer = None if self.coord() is None else Chem.Conformer()

            for j in range(cum_nodes[i], cum_nodes[i + 1]):
                atom = Chem.Atom(int(self.atom_type[j]))
                atom.SetFormalCharge(int(self.atom_charge[j]))
                atom.SetNumExplicitHs(int(self.atom_explicit_h[j]))
                atom.SetChiralTag(int(self.Chem.ChiralType(self.atom_chiral_tag[j])))
                atom.SetNumRadicalElectrons(int(self.atom_radical_electrons[j]))
                atom.SetNoImplicit(bool(self.atom_explicit_h[j] > 0 or self.atom_radical_electrons[j] > 0))
                atom.SetAtomMapNum(int(self.atom_map[j]))
                if self.atom_isotope is not None:
                    atom.SetIsotope(int(self.atom_isotope[j]))
                if conformer:
                    conformer.SetAtomPosition(j - self.cum_nodes[i], self.coord()[j])
                mol.AddAtom(atom)
            if conformer:
                mol.AddConformer(conformer)

            for j in range(cum_nodes[i], cum_edges[i+1]):
                h, t = edges[:, j]
                e_type = self.edge_type[j]
                if h < t:
                    k = mol.AddBond(h, t, Chem.BondType.names[self._id2bond.get(e_type)])
                    bond = mol.GetBondWithIdx(k-1)
                    bond.SetIsAromatic(self.bond_type[j] == self._bond2id.get('AROMATIC'))
                    bond.SetSterreo(Chem.BondStereo(int(self.bond_stereo[j])))
                    bond.SetBondDir(Chem.BondDir(int(self.bond_dir[j])))
            k = 0
            for j in range(cum_edges[i], cum_edges[i+1]):
                h, t, e_type = edges[:, j]
                if h < t:
                    if self.bond_stereo[j]:
                        bond.SetStereoAtoms(*self.bond_stereo_atom[j])
                    k += 1
            mol.UpdatePropertyCache()
            Chem.AssignStereochemistry(mol)
            mol.ClearComputedProps()
            mol.UpdatePropertyCache()
            mols.append(mol)
        return mols


Molecule.batch_type = MoleculeBatch
MoleculeBatch.item_type = Molecule
