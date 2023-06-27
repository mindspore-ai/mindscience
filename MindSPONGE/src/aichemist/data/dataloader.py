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
DataLoader
"""

import warnings
import os
import pickle

import numpy as np
from numpy import random
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from scipy import spatial
from Bio.PDB import PDBParser

from ..core import Registry as R
from ..core import DataLoader
from ..util.geometry import random_rotation_translation
from .molecule import Molecule
from . import feature
from .graph import Graph


mol_passers = {'smiles': Chem.MolFromSmiles, 'smarts': Chem.MolFromSmiles, 'inchi': Chem.MolFromInchi}


@R.register('dataset.MolSet')
class MolSet(DataLoader):
    """
    MolSet
    """
    _caches = ['data', 'label']

    def __init__(self,
                 batch_size: int = 128,
                 lazy=False,
                 shuffle=False,
                 length_unit=None,
                 energy_unit=None,
                 info=None,
                 verbose=0,
                 **kwargs) -> None:
        super().__init__(batch_size, verbose, shuffle, **kwargs)
        self.lazy = lazy
        self.length_unit = length_unit or 'nm'
        self.energy_unit = energy_unit or 'kcal'
        if isinstance(info, str):
            info = [info]
        self.info = info or ['graph']
        self.data = []
        self.label = []

    def __repr__(self):
        lines = f"#sample: {len(self)}\n" + \
                f"#task: {len(self.task_list)}"
        return f"{self.__class__.__name__}(\n  {lines}\n)"

    def __next__(self):
        data, label = super().__next__()
        data = Molecule.pack(data)
        output = []
        for info in self.info:
            if info == 'graph':
                output.append(data)
            else:
                value = getattr(data, info)
                output.append(value)
        return output + [label]

    @property
    def dim_node_feat(self):
        """Dimension of node feats."""
        return self.data[0].node_feat.shape[-1]

    @property
    def dim_edge_feat(self):
        """Dimension of edge features."""
        return self.data[0].edge_feat.shape[-1]

    @property
    def n_atom_type(self):
        """Number of different atom types."""
        return len(self.atom_types)

    @property
    def n_bond_type(self):
        """Number of different bond types."""
        return len(self.bond_types)

    @property
    def atom_types(self):
        """All atom types."""
        atom_types = set()

        if getattr(self, "lazy", False):
            warnings.warn(
                "Calling this function for dataset with lazy=True may take a large amount of time.")
            for smiles in self.mol:
                graph = Molecule.from_smiles(smiles, **self.kwargs)
                atom_types.update(graph.atom_type.tolist())
        else:
            for graph in self.data:
                atom_types.update(graph.atom_type.tolist())

        return sorted(atom_types)

    @property
    def bond_types(self):
        """All bond types."""
        bond_types = set()

        if getattr(self, "lazy", False):
            warnings.warn(
                "Calling this function for dataset with lazy=True may take a large amount of time.")
            for smiles in self.mol:
                graph = Molecule.from_smiles(smiles, **self.kwargs)
                bond_types.update(graph.edges[:, 2].tolist())
        else:
            for graph in self.data:
                bond_types.update(graph.edges[:, 2].tolist())

        return sorted(bond_types)

    def load_file(self, fname, fmt='smiles', mol_field='smiles', transform=None, max_len=None, **kwargs):
        """
        Load the dataset from a csv file.

        Parameters:
            fname (str): file name
            smiles_field (str, optional): name of SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            verbose (int, optional): output verbose level
            **kwargs
        """
        if fname.split('.')[-1] in self._seps:
            sep = self._seps.get(fname.split('.')[-1])
            df = pd.read_table(fname, sep=sep)
            self.label = df[self.task_list].values
            if fmt is not None:
                passer = mol_passers.get(fmt)
                if mol_field is None:
                    mol_field = fmt
                if mol_field in df:
                    setattr(self, 'mol', [passer(s) for s in df[mol_field]])
        elif fmt == 'sdf':
            assert os.path.exists(fname), f'Error: The file {fname} does not exist!'
            setattr(self, 'mol', Chem.SDMolSupplier(fname, True, True, False))
        else:
            raise TypeError(f'The iput file format \"{fmt}\" is not support')

        data = []
        labels = []
        if not self.lazy and hasattr(self, 'mol'):
            indexes = enumerate(self.mol)
            if self.verbose:
                indexes = tqdm(indexes, "Constructing molecules", total=len(self))
            for i, mol in indexes:
                if mol is not None:
                    d = Molecule.from_molecule(mol, **kwargs)
                    data.append(d)
                    if hasattr(self, 'label') and self.label is not None:
                        labels.append(self.label[i])
                    else:
                        label = [mol.GetProp(task) if mol.HasProp(task) else None
                                 for task in self.task_list]
                        labels.append(label)
                if max_len is not None and i >= max_len:
                    break
            assert len(labels) == len(data)
        self.initialize(data=data, label=np.stack(labels), transform=transform, **kwargs)
        return self


@R.register('dataset.ComplexSet')
class ComplexSet(DataLoader):
    """
    ComplexSet
    """
    _caches = ['lig_graphs', 'rec_graphs', 'pockets', 'rec_subgraphs', 'geometry_graphs']

    def __init__(self,
                 path='pdb/',
                 names=None,
                 category='train',
                 pocket_cutoff=8.0,
                 use_rec_atoms=False,
                 chain_radius=7,
                 ca_max_neighbors=10,
                 lig_max_neighbors=20,
                 max_translation=5.0,
                 lig_graph_radius=30,
                 rec_graph_radius=30,
                 surface_max_neighbors=5,
                 surface_graph_cutoff=5,
                 surface_mesh_cutoff=1.7,
                 hydrogen_mode=None,
                 use_rdkit_coords=False,
                 pocket_mode='match_leaf_atoms',
                 dataset_size=None,
                 rec_subgraph=False,
                 is_train_data=False,
                 min_shell_thickness=2,
                 subgraph_radius=10,
                 subgraph_max_neigbor=8,
                 subgraph_cutoff=4,
                 lig_mode='knn',
                 random_rec_atom_subgraph=False,
                 subgraph_augmentation=False,
                 lig_predictions_name=None,
                 geometry_regularization=False,
                 random_rec_atom_subgraph_radius=10,
                 geometry_regularization_ring=False,
                 n_conf=10,
                 **kwargs) -> None:
        self.chain_radius = chain_radius
        self.hydrogen_mode = hydrogen_mode
        self.path = path
        if isinstance(names, (tuple, list)):
            self.names = names
        elif isinstance(names, str) and os.path.exists(names):
            with open(names) as cnames:
                self.names = cnames.read().split()
        else:
            raise ValueError('Unsported parameters of complex_names')
        self.category = category
        self.pocket_cutoff = pocket_cutoff
        self.use_rec_atoms = use_rec_atoms
        self.lig_graph_radius = lig_graph_radius
        self.rec_graph_radius = rec_graph_radius
        self.surface_max_neighbors = surface_max_neighbors
        self.surface_graph_cutoff = surface_graph_cutoff
        self.surface_mesh_cutoff = surface_mesh_cutoff
        self.dataset_size = dataset_size
        self.ca_max_neighbors = ca_max_neighbors
        self.lig_max_neighbors = lig_max_neighbors
        self.max_translation = max_translation
        self.pocket_mode = pocket_mode
        self.use_ori_coord = use_rdkit_coords
        self.is_train_data = is_train_data
        self.subgraph_augmentation = subgraph_augmentation
        self.min_shell_thickness = min_shell_thickness
        self.rec_subgraph = rec_subgraph
        self.subgraph_radius = subgraph_radius
        self.subgraph_max_neigbor = subgraph_max_neigbor
        self.subgraph_cutoff = subgraph_cutoff
        self.random_rec_atom_subgraph = random_rec_atom_subgraph
        self.lig_mode = lig_mode
        self.random_rec_atom_subgraph_radius = random_rec_atom_subgraph_radius
        self.lig_predictions_name = lig_predictions_name
        self.geometry_regularization = geometry_regularization
        self.geometry_regularization_ring = geometry_regularization_ring
        self.n_conf = n_conf
        self.conformer_id = 0
        self.cache = {}
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        lig_graph = self.lig_graphs[idx][self.conformer_id]
        lig_coord = lig_graph.coord()
        rec_graph = self.rec_graphs[idx]
        rec_coord = rec_graph.coord()
        pocket = self.pockets[idx]

        # Randomly rotate and translate the ligand.
        rot, trans = random_rotation_translation(
            max_translation=self.max_translation)
        lig_coords_to_move = lig_graph.node_new_coord if self.use_ori_coord else lig_coord
        mean_to_remove = lig_coords_to_move.mean(axis=0, keepdims=True)
        lig_graph.node_new_coord = (lig_coords_to_move - mean_to_remove) @ rot.T + trans
        new_pocket = (pocket - mean_to_remove) @ rot.T + trans

        if self.subgraph_augmentation and self.is_train_data:
            if idx in self.cache:
                max_dist, min_dist, dists = self.cache[idx]
            else:
                lig_centroid = lig_coord.mean(dim=0)
                dists = np.norm(rec_coord - lig_centroid, dim=1)
                max_dist = np.max(dists)
                min_dist = np.min(dists)
                self.cache[idx] = (min_dist.item(), max_dist.item(), dists)
            radius = min_dist + self.min_shell_thickness + random.random() * abs((
                max_dist - min_dist - self.min_shell_thickness))
            rec_graph = rec_graph.subgraph(rec_graph, dists <= radius)
            assert rec_graph.n_node > 0
        if self.rec_subgraph:
            rec_graph = self.rec_atom_subgraphs[idx]
            if self.random_rec_atom_subgraph:
                rot, trans = random_rotation_translation(max_translation=2)
                translated_lig_coords = lig_coord + trans
                min_dists, _ = spatial.cdist(
                    rec_graph.coord, translated_lig_coords).min(dim=1)
                rec_graph = rec_graph.subgraph(
                    min_dists < self.random_rec_atom_subgraph_radius)
                assert rec_graph.n_node > 0

        geometry_graph = self.geometry_graphs[idx] if self.geometry_regularization else None
        lig_graph.node_coord = lig_coord
        rec_graph.node_coord = rec_coord
        out_ = [lig_graph, rec_graph, new_pocket, pocket, geometry_graph]
        return out_

    def __next__(self):
        if self.iterator >= len(self):
            raise StopIteration
        end = min(self.iterator + self.batch_size, len(self))
        lig_graphs = []
        rec_graphs = []
        geo_graphs = []
        pockets = []
        new_pockets = []
        for i in range(self.iterator, end):
            batch = self[i]
            lig_graph, rec_graph, new_pocket, pocket, geo_graph = batch
            if lig_graph.n_node != len(lig_graph.node_feat) or rec_graph.n_node != len(rec_graph.node_feat):
                continue
            lig_graphs.append(lig_graph)
            rec_graphs.append(rec_graph)
            new_pockets.append(new_pocket)
            pockets.append(pocket)
            geo_graphs.append(geo_graph)
        lig_graphs = Graph.pack(lig_graphs)
        rec_graphs = Graph.pack(rec_graphs)
        if self.geometry_regularization:
            geo_graphs = Graph.pack(geo_graphs)
        else:
            geo_graphs = None
        self.iterator = end
        out_ = [lig_graphs, rec_graphs, pockets, new_pockets, geo_graphs]
        return out_

    def get_receptor(self, rec_path, lig_coord, cutoff):
        """
        get_receptor
        """
        structure = PDBParser().get_structure('random_id', rec_path)
        rec = structure[0]
        all_coords = []
        min_distances = []
        backbones = []
        valid_chain_ids = []
        for i, chain in enumerate(rec):
            chain_coords = []
            chain_backbones = []
            invalid_res_ids = []
            for res_id, residue in enumerate(chain):
                if residue.get_resname() == 'HOH':
                    invalid_res_ids.append(residue.get_id())
                    continue
                residue_coords = []
                backbone = {'CA': None, 'N': None, 'C': None}
                for atom in residue:
                    coord = list(atom.get_vector())
                    residue_coords.append(coord)
                    # only append residue if it is an amino acid and
                    # not some weird molecule that is part of the complex
                    if atom.name in {'CA', 'C', 'N'}:
                        backbone[atom.name] = coord
                if None not in backbone.values():
                    chain_coords += residue_coords
                    chain_backbones.append(backbone)
                else:
                    invalid_res_ids.append(residue.get_id())
            for res_id in invalid_res_ids:
                chain.detach_child(res_id)
            if chain_coords:
                all_chain_coords = np.array(chain_coords)
                distances = spatial.distance.cdist(lig_coord, all_chain_coords)
                min_distance = distances.min()
            else:
                min_distance = np.inf

            min_distances.append(min_distance)
            all_coords.append(chain_coords)
            backbones.append(chain_backbones)
            if min_distance < cutoff:
                valid_chain_ids.append(chain.get_id())

        if not valid_chain_ids:
            chain_id = np.argmin(min_distances)
            valid_chain_ids.append(rec.get_list()[chain_id].get_id())
        coords = []
        ca_coords = []
        n_coords = []
        c_coords = []
        invalid_chains = []
        for i, chain in enumerate(rec):
            if chain.get_id() in valid_chain_ids:
                coords += all_coords[i]
                ca_coords += [b.get('CA') for b in backbones[i]]
                n_coords += [b.get('N') for b in backbones[i]]
                c_coords += [b.get('C') for b in backbones[i]]
            else:
                invalid_chains.append(chain.get_id())
        for invalid_id in invalid_chains:
            rec.detach_child(invalid_id)

        assert len(ca_coords) == len(n_coords)
        assert len(ca_coords) == len(c_coords)
        out_ = [rec, np.array(coords), np.array(ca_coords), np.array(n_coords), np.array(c_coords)]
        return out_

    def pocket_coord(self, lig, rec_coord, cutoff=5.0, pocket_mode='match_atoms'):
        """
        pocket_coord
        """
        lig_coord = lig.GetConformer().GetPositions()
        if pocket_mode == 'match_leaf_atoms':
            leaf_idx = [i for i, atom in enumerate(
                lig.GetAtoms()) if atom.GetDegree() <= 1]
            lig_coord = lig_coord[leaf_idx]
        lig_rec_dist = spatial.distance.cdist(
            lig_coord, rec_coord) if pocket_mode != 'lig_atoms' else None
        if pocket_mode in ['match_atoms', 'match_leaf_atoms', 'match_atoms_to_lig']:
            min_rec_val = np.min(lig_rec_dist, axis=1)
            lig_pocket = lig_coord[min_rec_val < cutoff]
            if pocket_mode == 'match_atoms_to_lig':
                pocket = lig_pocket
            else:
                min_rec_idx = np.argmin(lig_rec_dist, axis=1)
                rec_pocket = rec_coord[min_rec_idx[min_rec_val < cutoff]]
                pocket = 0.5 * (lig_pocket + rec_pocket)
        elif pocket_mode == 'lig_atoms':
            pocket = lig_coord
        elif pocket_mode == 'radius':
            cutoff_ = cutoff
            size = 0
            while size < 4:
                positive_tuple = np.where(lig_rec_dist < cutoff_)
                active_lig = positive_tuple[0]
                active_rec = positive_tuple[1]
                print(
                    'Increasing pocket cutoff radius by 0.5 because there were less than 4 pocket nodes with radius: ',
                    cutoff_)
                cutoff_ += 0.5
                size = active_lig.size
            lig_pocket = lig_coord[active_lig, :]
            rec_pocket = rec_coord[active_rec, :]
            assert np.max(np.linalg.norm(
                lig_pocket - rec_pocket, axis=1)) <= cutoff_
            pocket = 0.5 * (lig_pocket + rec_pocket)
        else:
            raise ValueError(f'pocket_mode -{pocket_mode}- not supported')
        return pocket

    def rec_ca_graph(self, ca_coords, n_coords, c_coords, cutoff=20, max_neighbor=None):
        """get_ca_graph"""
        n_resi = len(ca_coords)
        assert n_resi > 0, "rec contains only 1 residue!"

        resi_feats = np.zeros([n_resi, 3])
        n_i_feat = np.zeros([n_resi, 3])
        u_i_feat = np.zeros([n_resi, 3])
        v_i_feat = np.zeros([n_resi, 3])
        for i in range(n_resi):
            n_coord = n_coords[i]
            ca_coord = ca_coords[i]
            c_coord = c_coords[i]
            u_i = (n_coord - ca_coord) / np.linalg.norm(n_coord - ca_coord)
            t_i = (c_coord - ca_coord) / np.linalg.norm(c_coord - ca_coord)
            n_i = np.cross(u_i, t_i) / np.linalg.norm(np.cross(u_i, t_i))
            v_i = np.cross(n_i, u_i)

            assert np.fabs(np.linalg.norm(
                v_i) - 1.) < 1e-5, 'protein utils protein_to_graph_dips, v_i norm larger than 1'

            n_i_feat[i] = n_i
            u_i_feat[i] = u_i
            v_i_feat[i] = v_i
            resi_feats[i] = ca_coord

        # Build the k-NN graph
        graph = Graph.knn_graph(ca_coords, cutoff=cutoff,
                                max_neighbors=max_neighbor, divisor=4)

        edge_ori_feat = []
        for i, edge in enumerate(graph.edges):
            src, trg = edge[0], edge[1]
            basis_matrix = np.stack(
                (n_i_feat[trg, :], u_i_feat[trg, :], v_i_feat[trg, :]), axis=0)
            p_ij = basis_matrix @ (resi_feats[src, :] - resi_feats[trg, :])
            q_ij = basis_matrix @ n_i_feat[src, :]  # shape (3,)
            k_ij = basis_matrix @ u_i_feat[src, :]
            t_ij = basis_matrix @ v_i_feat[src, :]
            s_ij = np.concatenate(
                (p_ij, q_ij, k_ij, t_ij), axis=0)  # shape (12,)
            edge_ori_feat.append(s_ij)
        # shape of edge feat is (num_edges, 12)
        edge_ori_feat = np.stack(edge_ori_feat, axis=0)
        graph.edata = np.concatenate(
            [graph.edata, edge_ori_feat], axis=1)  # (num_edges, 17)
        graph.coord = ca_coords
        return graph

    def rec_atom_subgraph(self, lig_coord, rec_coord, cutoff=4, max_neighbor=8, subgraph_radius=7):
        """rec_atom_subgraph"""
        lig_rec_dist = spatial.distance.cdist(lig_coord, rec_coord)
        np.fill_diagonal(lig_rec_dist, np.inf)
        subgraph_idx = np.where(
            np.min(lig_rec_dist, axis=0) < subgraph_radius)[0]
        subgraph_coords = rec_coord[subgraph_idx]
        graph = Graph.knn_graph(
            coord=subgraph_coords, max_neighbors=max_neighbor, cutoff=cutoff, divisor=1)
        return graph

    def process(self):
        """process"""
        lig_graphs = []
        rec_graphs = []
        pockets = []
        rec_subgraphs = []
        geometry_graphs = []
        for key, value in self._caches:
            pkg = f'resources/{key}_{self.category}.pkg'
            if not os.path.exists(pkg):
                vars()[key] = []
            with open(pkg, 'rb') as fr:
                vars()[key] = pickle.load(fr)

        # In this complex's ligand the hydrogens cannot be removed
        removes = ['4acu', '3q4c']
        ob_mol2 = ['2jld', '6bt0', '5jm4', '6nao', '3p3h']
        indices = tqdm(self.names, desc='loading complexes')
        for name in indices:
            if name in removes:  # and self.hydrogen_mode != 'all':
                self.names.remove(name)
                continue
            indices.set_description(f'loading {name}')
            lig_path = os.path.join(self.path, name, f'{name}_ligand.mol2')
            if name in ob_mol2:
                lig_path = os.path.join(self.path, name, f'{name}_ligand.sdf')
            rec_path = os.path.join(
                self.path, name, f'{name}_protein_processed.pdb')

            # Convert ligands to graphs
            lig = Molecule.rd_from_file(
                lig_path, sanitize=True, hydrogen_mode=self.hydrogen_mode)
            if not os.path.exists(f'resources/lig_graph_{self.category}.pkg'):
                lig_graphs.append(self.get_lig_graph(lig))
            # Convert ligands to geometry graph
            if not os.path.exists(f'resources/geometry_graph_{self.category}.pkg'):
                if self.geometry_regularization:
                    geometry_graph = Molecule.geometry_graph(
                        lig, include_rings=self.geometry_regularization_ring)
                    geometry_graphs.append(geometry_graph)

            # Get receptors, filter chains, and get its coordinates
            if not os.path.exists(f'resources/rec_graph_{self.category}.pkg'):
                lig_coord = lig.GetConformer().GetPositions()
                rec, rec_coord, ca_coord, n_coord, c_coord = self.get_receptor(
                    rec_path=rec_path, lig_coord=lig_coord, cutoff=self.chain_radius)

                rec_graph = self.rec_ca_graph(ca_coord, n_coord, c_coord,
                                              cutoff=self.rec_graph_radius,
                                              max_neighbor=self.ca_max_neighbors)
                rec_graph.ndata = feature.rec_residue_featurizer(rec)
                rec_graphs.append(rec_graph)
                # Get Pocket Coordinates
                if not os.path.exists(f'resources/pocket_coord_{self.category}.pkg'):
                    pocket = self.pocket_coord(
                        lig, rec_coord, cutoff=self.pocket_cutoff, pocket_mode=self.pocket_mode)
                    pockets.append(pocket)

            # Get receptor subgraphs
            if self.rec_subgraph and not os.path.exists(f'resources/rec_subgraph_{self.category}.pkg'):
                rec_subgraph = self.rec_atom_subgraph(lig_coord, rec_coord,
                                                      max_neighbor=self.subgraph_max_neigbor,
                                                      subgraph_radius=self.subgraph_radius,
                                                      cutoff=self.subgraph_cutoff)
                rec_subgraphs.append(rec_subgraph)
        self.initialize(lig_graphs=lig_graphs, rec_graphs=rec_graphs, pockets=pockets,
                        rec_subgraphs=rec_subgraphs, geometry_graphs=geometry_graphs)
        for key, value in self._caches:
            self.path = f'resources/{key}_{self.category}.pkg'
            if os.path.exists(self.path):
                continue
            with open(f'resources/{key}_{self.category}.pkg', 'wb') as fw:
                pickle.dump(value, fw)
        return self

    def get_lig_graph(self, lig):
        """_summary_

        Args:
            lig (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.lig_mode == 'knn':
            lig_graphs = Molecule.rd_conformer_graph(lig,
                                                     max_neighbors=self.lig_max_neighbors,
                                                     use_rdkit_coord=self.use_ori_coord,
                                                     radius=self.lig_graph_radius, n_confs=self.n_conf)
        elif self.lig_mode == 'structure':
            lig_graph = Molecule.lig_structure_graph(lig)
            lig_graphs = [lig_graph]
        else:
            raise ValueError(
                f'The graph mode of ligand {self.lig_mode} is not supported!')
        node_feat = feature.lig_atom_featurizer(lig)
        for lig_graph in lig_graphs:
            lig_graph.node_feat = node_feat
        assert len(node_feat) == lig_graph.n_node
        lig_graphs.append(lig_graphs)
        return lig_graph
