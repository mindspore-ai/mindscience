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
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import mindspore as ms
from mindspore import ops
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from openbabel import pybel

from ..features import mol_feat as feature
from ..configs import Registry as R
from .graph import GraphBatch, Graph
from .. import features
from .. import utils

logger = logging.getLogger(__name__)
plt.switch_backend('agg')

ALIAS = {'atom': 'node_', 'bond': 'edge_', 'mol': 'graph_'}


@R.register('data.Molecule')
@dataclass
class Molecule(Graph):
    """"
    Data class of Molecule inherited from Graph

    Args:
        node_charge (array_like):           Formal charge for each atom. Shape: (N, )
        node_explicit_h (array_like):       number of explicit hydrogens for each node. Shape: (N, )
        node chiral_tag (array_like):       Chiral tag for each atom. Shape: (N, )
        node_radical_electrons (array_like): Number of radical electrons for each atom. Shape: (N, )
        node_map (array_like):              Number of atom map for each atom. Shape (N, )
        edge_stereo (array_like):           If each edge is stereo (i.e. 1) or not (i.e. 0). Shape: (E, )
        edge_dir (array_like):              Direction of each Bond (for chirality). Shape (E, )
        edge_stereo_atom (array_like):      Stereo atoms for each bond. Shape: (E, 2)
    """
    node_charge: np.ndarray = None
    node_explicit_h: np.ndarray = None
    node_chiral_tag: np.ndarray = None
    node_radical_electrons: np.ndarray = None
    node_map: np.ndarray = None
    edge_stereo: np.ndarray = None
    edge_dir: np.ndarray = None
    edge_stereo_atom: np.ndarray = None

    def __eq__(self, other):
        smiles = self.to_smiles(isomeric=False, atom_map=False, canonical=True)
        other_smiles = other.to_smiles(isomeric=False, atom_map=False, canonical=True)
        return smiles == other_smiles

    def __getattr__(self, key):
        if key not in self.__dict__ and key.startswith('__'):
            return super().__getattr__(key)
        keys = key.split('_')
        prefix = keys[0]
        if prefix in ALIAS:
            key = ALIAS[prefix] + '_'.join(keys[1:])
        value = self.__dict__[key]
        return value

    @classmethod
    def from_molecule(cls,
                      mol,
                      atom_feat=None,
                      bond_feat=None,
                      mol_feat=None,
                      add_hs=False,
                      kekulized=False):
        """
        Build the Molecule object with RDKit.

        Args:
            mol (Chem.Mol):                         Molecule data construct by RDkit.
            atom_feat (str, Callable, optional):    The method to extract atom features. Defaults to None.
            bond_feat (_type_, optional):           The method to extract bond features. Defaults to None.
            mol_feat (_type_, optional):            The method to extract molecule features. Defaults to None.
            add_hs (bool, optional):                If True, all of the implicit hydrogens will be added.
                                                    Defaults to False.
            kekulized (bool, optional):             If True, all of aromatic bonds will be transformed to single
                                                    or double bond. Defaults to False.

        Returns:
            Molecule
        """
        if mol is None:
            mol = feature.empty_mol

        if add_hs:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)
        if kekulized:
            Chem.Kekulize(mol)
        n_relation = len(feature.bond2id) - 1 if kekulized else len(feature.bond2id)
        atom_feat = cls._standarize_option(atom_feat)
        bond_feat = cls._standarize_option(bond_feat)
        mol_feat = cls._standarize_option(mol_feat)

        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
        atom_type = np.zeros(len(atoms), dtype=int)
        atom_charge = np.zeros(len(atoms), dtype=int)
        atom_explicit_h = np.zeros(len(atoms))
        atom_chiral_tag = np.zeros(len(atoms), dtype=int)
        atom_radical_electrons = np.zeros(len(atoms))
        atom_map = np.zeros(len(atoms))
        atom_feat_ = []
        for i, atom in enumerate(atoms):
            atom_type[i] = atom.GetAtomicNum()
            atom_charge[i] = atom.GetFormalCharge()
            atom_explicit_h[i] = atom.GetNumExplicitHs()
            atom_chiral_tag[i] = atom.GetChiralTag()
            atom_radical_electrons[i] = atom.GetNumRadicalElectrons()
            atom_map[i] = atom.GetAtomMapNum()

            feat = []
            for name in atom_feat:
                func = R.get("feature.atom." + name)
                feat += func(atom)
            atom_feat_.append(feat)

        if atom_feat_ and atom_feat:
            atom_feat = np.array(atom_feat_)
        else:
            atom_feat = None

        if mol.GetNumConformers() > 0:
            atom_coord = mol.GetConformer().GetPositions()
        else:
            atom_coord = None
        bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())]
        edges = np.zeros((len(bonds) * 2, 2), dtype=np.int32)
        bond_type = np.zeros(len(bonds) * 2, dtype=np.int32)
        bond_stereo = np.zeros(len(bonds) * 2, dtype=np.int32)
        bond_dir = np.zeros(len(bonds) * 2, dtype=np.int32)
        bond_stereo_atom = np.zeros((len(bonds) * 2, 2), dtype=np.int32)
        bond_feat_ = []

        for i, bond in enumerate(bonds):
            btype = str(bond.GetBondType())
            stereo = int(bond.GetStereo())
            bond_dir_ = int(bond.GetBondDir())
            atoms_ = [a for a in bond.GetStereoAtoms()]
            if len(atoms_) != 2:
                atoms_ = [0, 0]
            assert btype in feature.bond2id
            btype = feature.bond2id.get(btype)
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges[2*i:2*i+2] = [[h, t], [t, h]]

            bond_type[2*i:2*i+2] = [btype, btype]
            bond_stereo[2*i:2*i+2] = [stereo, stereo]
            bond_dir[2*i:2*i+2] = [bond_dir_, bond_dir_]
            bond_stereo_atom[2*i:2*i+2] = [atoms_, atoms_]
            feat = []
            for name in bond_feat:
                func = R.get("feature.bond." + name)
                feat += func(bond)
            bond_feat_ += [feat, feat]
        if bond_feat_ and bond_feat:
            bond_feat = np.array(bond_feat_)
        else:
            bond_feat = None

        mol_feat_ = []
        for name in mol_feat:
            func = R.get("feature.mol." + name)
            mol_feat_ += func(mol)
        if mol_feat_ and mol_feat:
            mol_feat = np.array(mol_feat_)
        else:
            mol_feat = None

        molecule = cls(edges=edges.T, node_charge=atom_charge, n_node=mol.GetNumAtoms(),
                       node_chiral_tag=atom_chiral_tag, node_coord=atom_coord, node_explicit_h=atom_explicit_h,
                       node_feat=atom_feat, node_map=atom_map, n_relation=n_relation,
                       node_radical_electrons=atom_radical_electrons, node_type=atom_type,
                       edge_feat=bond_feat, edge_dir=bond_dir, edge_stereo=bond_stereo,
                       edge_stereo_atom=bond_stereo_atom, edge_type=bond_type, graph_feat=mol_feat)
        return molecule

    @classmethod
    def rd_from_file(cls, mol_file, sanitize=False, calc_charges=False, hydrogen_mode=False):
        """
        Load a molecule with RDKit from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

        Args:
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
        """
        Load a molecule with Openbabel from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

        Args:
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
                    atom_feat='default',
                    bond_feat='default',
                    mol_feat=None,
                    add_hs=False,
                    kekulized=False):
        """
        Build the Molecule object from SMILES string with RDKit.

        Args:
            smiles (str):                           SMILES string
            atom_feat (str, Callable, optional):    The method to extract atom features. Defaults to None.
            bond_feat (_type_, optional):           The method to extract bond features. Defaults to None.
            mol_feat (_type_, optional):            The method to extract molecule features. Defaults to None.
            add_hs (bool, optional):                If True, all of the implicit hydrogens will be added.
                                                    Defaults to False.
            kekulized (bool, optional):             If True, all of aromatic bonds will be transformed to single
                                                    or double bond. Defaults to False.

        Returns:
            _type_: _description_
        """
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        return cls.from_molecule(mol, atom_feat, bond_feat, mol_feat, add_hs, kekulized)

    @classmethod
    def rd_conformer_sampling(cls, mol, n_conf: int = 10, seed: int = None):
        """
        Sampling the conformer of the molecule with RDKit.

        Args:
            mol (Chem.Mol):         Molecule data constructed with RDKit
            n_conf (int, optional): number of conformations to be sampled. Defaults to 10.
            seed (int, optional):   random seed. Defaults to None.

        Returns:
            coords (array_like):    coordinate matrix for each atom in the molecule. Shape: (N, 3).
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
            coords = np.array(coords)
        return coords

    @classmethod
    def ob_conformer_sampling(cls, mol, n_conf=10):
        """
        Sampling the conformer of the molecule with Openbabel.

        Args:
            mol (OBMol):            Molecule data constructed with Openbabel
            n_conf (int, optional): number of conformations to be sampled. Defaults to 10.
            seed (int, optional):   random seed. Defaults to None.

        Returns:
            coords (array_like):    coordinate matrix for each atom in the molecule. Shape: (N, 3).
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
            coords = np.array(coords)
        return coords

    @classmethod
    def is_aromatic_ring(cls, mol, bond_ring):
        """_summary_

        Args:
            mol (Chem.Mol):         Molecule data with RDKit
            bond_ring (array_like): Index of the bond.

        Returns:
            flag (bool): If the bond is aromatic, return Ture, otherwith False.
        """
        for b_id in bond_ring:
            if not mol.GetBondWithIdx(b_id).GetIsAromatic():
                return False
        return True

    @classmethod
    def get_aromatics_rings(cls, mol, atom_idx=None):
        """_summary_

        Args:
            mol (Chem.Mol):             Molecule data with RDKit
            atom_idx (int, optional):   index of atom. Defaults to None.

        Returns:
            rings (list): a index list of atoms that belong to aromatic rings.
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
        """
        Construct the geometric graph for given molecule.

        Args:
            mol (Chem.Mol):                 Molecule data with RDKit
            include_rings (bool, optional): If True, the atoms in the same aromatic ring will be connected with an edge.
                                            Defaults to True.

        Returns:
            graph (Graph): Geometric graph
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
        edges = np.array(edges).T
        edge_feat = np.linalg.norm(coord[edges[0]] - coord[edges[1]], axis=1)
        graph = Graph(edges=edges, n_node=mol.GetNumAtoms(), edge_feat=edge_feat)
        return graph

    @classmethod
    def masked_angle_graph(cls, mol):
        """
        Generate the angle of ecah rotatable bond for the given molecule.

        Args:
             mol (Chem.Mol):        Molecule data with RDKit

        Returns:
            mask (array_like):      If the bond is rotatable the value is 1, otherwise 0. Shape (E, )
            angles (array_like):    the angle of each rotatable bond, if the bond is not rotatable,
                                    the value is -1. Shape (E, )
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
        """
        Sandarize the option. The option with type of ``str`` will be transformed to list.

        Args:
            option (General): option

        Returns:
            option (list): a list type of option.
        """
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    def to_molecule(self):
        """
        Transform itself to editable molecule with RDKit

        Returns:
            mol (Chem.RWMol): Editable molecule with RDKit
        """
        self.to_array()
        mol = Chem.RWMol()

        self._add_atoms(mol)

        if self.coord is not None:
            coord = self.coord.tolist()
            conformer = Chem.Conformer()
            for i, coord in self.coord:
                conformer.SetAtomPosition(i, coord)
            mol.AddConformer(conformer)

        self._add_bonds(mol)
        return mol

    def to_smiles(self, isomeric=True, atom_map=True, canonical=False):
        """
        Transform itself to SMILES string with RDKit

        Args:
            isomeric (bool, optional):  If True, the SMILES will include isomeric information. Defaults to True.
            atom_map (bool, optional):  If True, the SMILES will include atom_map information. Defaults to True.
            canonical (bool, optional): If True, the output will be canonical SMILES, i.e.,
                                        the SMILES is unique for given moleule. Defaults to False.

        Returns:
            smiles (str): SMILES string
        """
        try:
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
        except (RuntimeError, Chem.AtomValenceException):
            smiles = None
        return smiles

    def ion_to_molecule(self):
        """
        Clearing the number of explicit hydrogens and radical_electrons information.
        And the formal charge will be set to 0 except positive nitrogen (set to 1).

        Returns:
            mol (Molecule): transformed molecule
        """
        kwargs = deepcopy(self.__dict__)
        pos_nitrogen = (self.atom_type == 7) & (self.explicit_val() > 3)
        kwargs['node_charge'] = pos_nitrogen.astype(int)
        kwargs['node_explicit_h'] = np.zeros_like(kwargs.get('node_explicit_h'))
        kwargs['node_radical_electrons'] = np.zeros_like(kwargs.get('node_radical_electrons'))
        return Molecule(**kwargs)

    def explicit_val(self):
        """
        Calculate explicit valence of each atom.

        Returns:
            val (array_like): Explicit valence of each atom.
        """
        bond2val = np.array(feature.bond2val) if self.detach else ms.Tensor(feature.bond2val)
        explicit_val = utils.scatter_add(bond2val[self.edge_type], self.edges[0], n_axis=self.n_node)
        val = explicit_val.round().astype(int)
        return val

    def is_valid(self):
        """
        Check if the valence of all atoms are not smaller than maximum of allowable valence.

        Returns:
            is_valid (bool): molecule is valid or not.
        """
        if self.detach:
            atom2val = np.full(features.NUM_ATOM, np.NaN)
        else:
            atom2val = ops.full(features.NUM_ATOM, np.NaN)
        for k, v in feature.atom2val.items():
            atom2val[k] = v
        max_atom_val = atom2val[self.node_type]

        pos_nitrogen = (self.node_type == 7) & (self.node_charge == 1)
        max_atom_val[pos_nitrogen] = 4
        # if np.any(np.isnan(max_atom_val)):
        is_valid = (self.explicit_val() <= max_atom_val).all()
        return is_valid

    def is_valid_rdkit(self):
        """
        Check if the molecule could be parsed by RDKit correctly.

        Returns:
            is_valid (bool): molecule is valid or not.
        """
        try:
            mol = self.to_molecule()
            Chem.SanitizeMol(mol)
            is_valid = True
        except RuntimeError:
            is_valid = False
        return is_valid

    def _add_atoms(self, mol):
        """adding all atoms to the given molecule with the properties reserved in itself

        Args:
            mol (Chem.RWMol): editable molecule with RDKit format
        """
        for i in range(self.n_node):
            atom = Chem.Atom(int(self.atom_type[i]))
            if self.node_charge is not None:
                atom.SetFormalCharge(int(self.node_charge[i]))
            if self.node_explicit_h is not None:
                atom.SetNumExplicitHs(int(self.node_explicit_h[i]))
            if self.node_chiral_tag is not None:
                atom.SetChiralTag(Chem.ChiralType(self.node_chiral_tag[i]))
            if self.node_radical_electrons is not None:
                atom.SetNumRadicalElectrons(int(self.node_radical_electrons[i]))
            if self.node_radical_electrons is not None and self.node_explicit_h is not None:
                atom.SetNoImplicit(bool(self.node_explicit_h[i] > 0 or self.node_radical_electrons[i] > 0))
            if self.node_map is not None:
                atom.SetAtomMapNum(int(self.node_map[i]))
            mol.AddAtom(atom)

    def _add_bonds(self, mol):
        """adding all bonds to the given molecule with the properties reserved in itself

        Args:
            mol (Chem.RWMol): editable molecule with RDKit format
        """
        for i in range(self.n_edge):
            h, t = self.edges
            etype = self.bond_type[i]
            if h >= t:
                continue
            j = mol.AddBond(int(h), int(t), Chem.BondType.names[feature.id2bond.get(etype)])
            bond = mol.GetBondWithIdx(j - 1)
            if self.edge_type is not None:
                bond.SetIsAromatic(bool(self.edge_type[i] == feature.bond2id.get('AROMATIC')))
            if self.edge_stereo is not None:
                bond.SetStereo(Chem.BondStereo(self.edge_stereo[i]))
            if self.edge_dir is not None:
                bond.SetBondDir(Chem.BondDir(int(self.edge_dir[i])))
        j = 0
        for i in range(self.n_edge):
            h, t = self.edges
            etype = self.bond_type[i]
            if h < t:
                if self.edge_stereo is not None and self.edge_stereo[i]:
                    bond = mol.GetBondWithIdx(j)
                    bond.SetStereoAtoms(*self.bond_stereo_atom[i].tolist())
                j += 1
            mol.UpdatePropertyCache()
            Chem.AssignStereochemistry(mol)
            mol.ClearComputedProps()
            mol.UpdatePropertyCache()


@R.register('data.MoleculeBatch')
@dataclass
class MoleculeBatch(GraphBatch, Molecule):
    """Data class of a batch molecule inherited from GraphBath and Molecule"""
    @classmethod
    def from_molecule(cls,
                      mol,
                      atom_feat=None,
                      bond_feat=None,
                      mol_feat=None,
                      add_hs=False,
                      kekulized=False):
        """Inherited from the class of Molecule"""
        mol_graphs = [super().from_molecule(m, atom_feat=atom_feat, bond_feat=bond_feat, mol_feat=mol_feat,
                                            add_hs=add_hs, kekulized=kekulized)
                      for m in mol]
        return cls.pack(mol_graphs)

    @classmethod
    def from_smiles(cls, smiles, atom_feat=None, bond_feat=None, mol_feat=None,
                    add_hs=False, kekulized=False):
        """
        Create a packed molecule from a list of SMILES strings.

        Args:
            smiles (str): list of SMILES strings
            atom_feat (str or List[str], optional): atom features to extract
            bond_feat (str or List[str], optional): bond features to extract
            mol_feat (str or List[str], optional):  molecule features to extract
            add_hs (bool, optional):                store hydrogens in the molecule graph.
                                                    By default, hydrogens are dropped
            kekulized (bool, optional):             convert aromatic bonds to single/double bonds.
                                                    Note this only affects the relation in ``edges``.
                                                    For ``bond_type``, aromatic bonds are always stored explicitly.
                                                    By default, aromatic bonds are stored.

        Returns:
            mols (MoleculeBatch): MoleculeBatch generated from a list of SMILES
        """
        mols = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            assert mol is not None
            mols.append(mol)
        return cls.from_molecule(mols, atom_feat=atom_feat, bond_feat=bond_feat,
                                 mol_feat=mol_feat, add_hs=add_hs, kekulized=kekulized)

    def is_valid(self):
        """
        Check if the valence of all atoms are not smaller than maximum of allowable valence for each molecules.

        Returns:
            is_valid (List[bool]): each molecule is valid or not.
        """
        atom2val = np.full(features.NUM_ATOM, np.NaN, dtype=np.float32)
        if not self.detach:
            atom2val = ms.Tensor.from_numpy(atom2val)
        for k, v in feature.atom2val.items():
            atom2val[k] = v
        max_atom_val = atom2val[self.atom_type]
        pos_nitrogen = (self.atom_type == 7) & (self.node_charge == 1)
        max_atom_val[pos_nitrogen] = 4

        is_valid = self.explicit_val() <= max_atom_val
        is_valid = is_valid.astype(float)
        bool_ = np.bool_ if self.detach else ms.bool_
        is_valid = utils.scatter_min(is_valid, self.node2graph, n_axis=self.batch_size).astype(bool_)
        return is_valid

    def is_valid_rdkit(self):
        """
        Check if each molecule could be parsed by RDKit correctly in itself.

        Returns:
            is_valid (bool): each molecule is valid or not.
        """
        return np.concatenate([mol.is_valid_rdkit for mol in self])

    def to_molecule(self):
        """
        Transform itself to a list of editable molecules with RDKit

        Returns:
            mols (List[Chem.RWMol]): a list of editable molecules with RDKit
        """
        cum_nodes = [0] + self.cum_nodes.tolist()
        cum_edges = [0] + self.cum_edges.tolist()
        edges = self.edges.copy()
        edges -= self.offsets

        mols = []
        for i in range(self.batch_size):
            mol = Chem.RWMol()
            conformer = None if self.coord is None else Chem.Conformer()

            for j in range(cum_nodes[i], cum_nodes[i + 1]):
                atom = Chem.Atom(int(self.atom_type[j]))
                atom.SetFormalCharge(int(self.node_charge[j]))
                atom.SetNumExplicitHs(int(self.node_explicit_h[j]))
                atom.SetChiralTag(int(self.Chem.ChiralType(self.node_chiral_tag[j])))
                atom.SetNumRadicalElectrons(int(self.node_radical_electrons[j]))
                atom.SetNoImplicit(bool(self.node_explicit_h[j] > 0 or self.node_radical_electrons[j] > 0))
                atom.SetAtomMapNum(int(self.node_map[j]))
                if conformer:
                    conformer.SetAtomPosition(j - self.cum_nodes[i], self.coord[j])
                mol.AddAtom(atom)
            if conformer:
                mol.AddConformer(conformer)

            for j in range(cum_nodes[i], cum_edges[i+1]):
                h, t = edges[:, j]
                e_type = self.bond_type[j]
                if h < t:
                    k = mol.AddBond(h, t, Chem.BondType.names[feature.id2bond.get(e_type)])
                    bond = mol.GetBondWithIdx(k-1)
                    bond.SetIsAromatic(self.bond_type[j] == feature.bond2id.get('AROMATIC'))
                    bond.SetSterreo(Chem.BondStereo(int(self.edge_stereo[j])))
                    bond.SetBondDir(Chem.BondDir(int(self.edge_dir[j])))
            k = 0
            for j in range(cum_edges[i], cum_edges[i+1]):
                h, t, e_type = edges[:, j]
                if h < t:
                    if self.edge_stereo[j]:
                        bond.SetStereoAtoms(*self.edge_stereo_atom[j])
                    k += 1
            mol.UpdatePropertyCache()
            Chem.AssignStereochemistry(mol)
            mol.ClearComputedProps()
            mol.UpdatePropertyCache()
            mols.append(mol)
        return mols


Molecule.batch_type = MoleculeBatch
MoleculeBatch.item_type = Molecule
