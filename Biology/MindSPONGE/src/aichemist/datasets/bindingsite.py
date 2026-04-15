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
BindingSite
"""

import os
import random
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import rdFreeSASA
import openbabel as ob
from openbabel import pybel
import skimage
import numpy as np
from scipy import ndimage
from Bio import PDB
from Bio.PDB import SASA

from .. import core
from ..features.const import RESI_ATOM_FEAT


class Feature:
    """
    Generate feature for protein, there are 17 channels:
    atom types:     B, C, N, O, P, [S, Se], helogen, metal
    atomic fatures: hybridization, heavy atoms, heteroatoms, partial charges, hydrophobic,
                    aromatic, acceptor, donor, ring.

    Args:
        save_molecule_codes (bool, optional):   If true, the feature contains molecule codes. Defaults to True.
        smarts_properties (List, optional):     SMARTS fragments if molecule contains. Defaults to None.
        smarts_labels (List, optional):         The labels of SMARTS. Defaults to None.

    """

    def __init__(self,
                 save_molecule_codes=True,
                 smarts_properties=None,
                 smarts_labels=None):
        self.feature_names = []
        self.save_molecule_codes = save_molecule_codes
        self.atom_codes = {}
        metals = [3, 4, 11, 12, 13] + list(range(19, 32)) + \
            list(range(37, 51)) + list(range(55, 84)) + list(range(87, 104))
        atom_classes = [(5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'),
                        ([16, 34], 'S'),
                        ([9, 17, 35, 53], 'halogen'),
                        (metals, 'metal')]
        for code, (atom, name) in enumerate(atom_classes):
            if isinstance(atom, list):
                for a in atom:
                    self.atom_codes[a] = code
            else:
                self.atom_codes[atom] = code
            self.feature_names.append(name)
        self.num_atom_classes = len(atom_classes)

        self.named_props = ['hyb', 'heavydegree', 'heterodegree', 'partialcharge']
        self.feature_names += self.named_props

        if save_molecule_codes:
            # remember if an atom belongs to the ligand or to the protein
            self.feature_names.append('molcode')
        self.callables = []

        if smarts_properties is None:
            self.smarts = ['[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                           '[a]',
                           '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                           '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                           '[r]']
            smarts_labels = ['hydrophobic', 'aromatic',
                             'acceptor', 'donor', 'ring']
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.smarts = smarts_properties

        smarts_labels = [f'smarts{i}' for i in range(len(self.smarts))]
        self._patterns = [pybel.Smarts(smarts) for smarts in self.smarts]
        self.feature_names += smarts_labels

    def encode_num(self, atomic_num):
        """Encoding the atom types for the first eight channels

        Args:
            atomic_num (int): Atomic number of the first eight channels.

        Raises:
            TypeError: Atomic number must be int.

        Returns:
            (np.ndarray): shape (number of atoms in system, 8).
        """
        if not isinstance(atomic_num, int):
            raise TypeError(f'Atomic number must be int, {type(atomic_num)} was given')
        encoding = np.zeros(self.num_atom_classes)
        try:
            encoding[self.atom_codes.get(atomic_num)] = 1.0
        except RuntimeError:
            pass
        return encoding

    def find_smarts(self, molecule: pybel.Molecule):
        """Encoding the atomic properties of 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring'

        Args:
            molecule (pybel.Molecule): Pybel.Molecule object.

        Raises:
            TypeError: Molecule must be pybel.Molecule object.

        Returns:
            (np.ndarray): shape (number of atoms in system, 5).
        """
        if not isinstance(molecule, pybel.Molecule):
            raise TypeError(f'molecule must be pybel.Molecule object, {type(molecule)}was given')
        features = np.zeros((len(molecule.atoms), len(self._patterns)))
        for (pattern_id, pattern) in enumerate(self._patterns):
            atoms_with_prop = pattern.findall(molecule)
            if atoms_with_prop:
                atoms_with_prop = np.array([a for atoms in atoms_with_prop for a in atoms]) - 1
                features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_features(self, molecule: pybel.Molecule, molcode=None):
        """Generate the coordinate and features

        Args:
            molecule (pybel.Molecule): Pybel.Molecule object.
            molcode (float, optional): Record if an atom belongs to the ligand or to the protein. Defaults: None

        Raises:
            ValueError:     Save_molecule_codes is set to True, specify code for the molecule should be provided.
            TypeError:      Motlype must be float.
            RuntimeError:   Got NaN when calculating features.

        Returns:
            coordinates (np.ndarray): Shape (number of atoms in system, 3)
            representation (np.ndarray):  Shape (number of atoms in system, 17)
        """
        if molcode is None:
            if self.save_molecule_codes:
                raise ValueError(
                    'save_molecule_codes is set to True, you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError(f'motlype must be float, {type(molcode)} was given')
        coords = []
        features = []
        heavy_atoms = []
        # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
        for i, atom in enumerate(molecule):
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)
                feature = self.encode_num(atom.atomicnum).tolist() + \
                    [getattr(atom, prop) for prop in self.named_props] + \
                    [func(atom) for func in self.callables]
                features.append(feature)
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack(
                (features, molcode * np.ones((len(features), 1))))
        features = np.hstack(
            [features, self.find_smarts(molecule)[heavy_atoms]])
        if np.isnan(features).any():
            raise RuntimeError('got NaN when calculating features')
        return coords, features


class Rotator:
    """
    Randomly rotate the protein in the train step, refer to PUResNet
    """

    def __init__(self):
        # create matrices for all possible 90* rotations of a box
        self.rotations = [self._rotation_matrix([1, 1, 1], 0)]
        for a1 in range(3):  # about X, Y and Z - 9 rotations
            for t in range(1, 4):
                axis = np.zeros(3)
                axis[a1] = 1
                theta = t * np.pi / 2.0
                self.rotations.append(self._rotation_matrix(axis, theta))
        for (a1, a2) in combinations(range(3), 2):  # about each face diagonal - 6 rotations
            axis = np.zeros(3)
            axis[[a1, a2]] = 1.0
            theta = np.pi
            self.rotations.append(self._rotation_matrix(axis, theta))
            axis[a2] = -1.0
            self.rotations.append(self._rotation_matrix(axis, theta))
        for t in [1, 2]:  # about each space diagonal - 8 rotations
            theta = t * 2 * np.pi / 3
            axis = np.ones(3)
            self.rotations.append(self._rotation_matrix(axis, theta))
            for a1 in range(3):
                axis = np.ones(3)
                axis[a1] = -1
                self.rotations.append(self._rotation_matrix(axis, theta))

    @staticmethod
    def _rotation_matrix(axis, theta):
        """counterclockwise rotation about a given axis by theta radians"""
        if not isinstance(axis, (np.ndarray, list, tuple)):
            raise TypeError('axis must be an array of floats of shape (3,)')
        axis = np.asarray(axis, dtype=float)

        if axis.shape != (3,):
            raise ValueError('axis must be an array of floats of shape (3,)')

        if not isinstance(theta, (float, int)):
            raise TypeError('theta must be a float')

        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def rotate(self, coords, rotation):
        """Rotate with repsect of given coordinates

        Args:
            coords (array_like):    3D coordinate matrix. Shape (N, 3)
            rotation (array_like):  rotation matrix. Shape (3, 3)

        Raises:
            TypeError:  The type of coords must be ndarray, list or tuple
            ValueError: The shape of coords must be (N, 3)
            ValueError: Invalid rotation number
            ValueError: Invalid rotation

        Returns:
            coords: coordinate matrix after random rotation
        """
        if not isinstance(coords, (np.ndarray, list, tuple)):
            raise TypeError('coords must be an array of floats of shape (N, 3)')
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError('coords must be an array of floats of shape (N, 3)')
        if isinstance(rotation, int):
            if rotation < 0 or rotation >= len(self.rotations):
                raise ValueError(f'Invalid rotation number {rotation}!')
            return np.dot(coords, self.rotations[rotation])
        if isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            return np.dot(coords, rotation)
        raise ValueError(f'Invalid rotation {rotation}!')


class BindingSite(core.BaseDataset):
    """Class to make 3D grid for the binding site prediction

    Args:
        data_path(str):                     Pathway of database.
        site_file(str):                     Coordinate file of ligand. Default: 'ligand.mol2'
        protein_cavity_same_feature(bool):  If the cavity use the same feature as protein. Default: ``False``.
        resolution_scale(float):            Resolution of grid. Default: 0.5
        max_distance(int):                  Max number of grids. Default: 35
        max_translation(int or float):      Max distance of translation. Default: 3
        footprint(int or np.ndarray):       If margin for pocket based on ligand structure. Default: ``None``.
        fmt(str):                           File form of ligand. Default:'mol2'
        surface_scale(float):               Cutoff to search the protein surface atoms using SASA. Default:1.4
        is_training(bool):                  If train the model. Default:False

    """
    _caches = ['protein_coords', 'protein_features', 'names']
    columns = ['protein_grids', 'names']

    def __init__(self,
                 data_path,
                 site_file='ligand.mol2',
                 protein_cavity_same_feature=False,
                 resolution_scale=0.5,
                 max_distance=35,
                 max_translation=3,
                 footprint=None,
                 fmt='mol2',
                 surface_scale=1.4,
                 is_training=False,
                 **kwargs):
        super().__init__(**kwargs)
        if is_training:
            self._caches = ['protein_coords', 'protein_features', 'cavity_coords', 'cavity_features']
            self.columns = ['protein_grids', 'cavity_grids']
            self.y_channels = None
        self.data_path = data_path
        self.is_training = is_training
        self.resolution_scale = resolution_scale
        self.surface_scale = surface_scale
        self.max_distance = max_distance
        self.max_translation = max_translation
        self.site_file = site_file
        self.protein_cavity_same_feature = protein_cavity_same_feature
        self.fmt = fmt
        # for cavity6.mol2  #dim should be 18 for Unet
        box_size = int(np.ceil(2 * max_distance * resolution_scale + 1))  # TODO +2?
        if footprint is not None:  # footprint: margin for pocket based on ligand structure
            if isinstance(footprint, int):
                if footprint == 0:
                    footprint = np.ones([1] * 5)
                elif footprint < 0:
                    raise ValueError('footprint cannot be negative')
                elif (2 * footprint + 3) > box_size:
                    raise ValueError('footprint cannot be bigger than box')
                else:
                    footprint = skimage.draw.ellipsoid(
                        footprint, footprint, footprint)
                    footprint = footprint.reshape((1, *footprint.shape, 1))
            elif isinstance(footprint, np.ndarray):
                if not ((footprint.ndim == 5) and (len(footprint) == 1) and (footprint.shape[-1] == 1)):
                    raise ValueError(f'footprint shape should be (1, N, M, L, 1), got {footprint.shape} instead')
            else:
                raise TypeError(f'footprint should be either int or np.ndarray of shape (1, N, M, L, 1), got \
                    {type(footprint)} instead')
            self.footprint = footprint
        else:
            footprint = skimage.draw.ellipsoid(2, 2, 2)
            self.footprint = footprint.reshape((*footprint.shape, 1))

        self.names = sorted(os.listdir(data_path))
        self.transform_random = 1
        self.protein_coords = []
        self.protein_features = []
        self.cavity_coords = []
        self.cavity_features = []

    def __getitem__(self, indices):
        """Convert atom coordinates and features into a fixed-sized 3D grids

        Returns:
            3D grids and system names.
        """
        inputs = super().__getitem__(indices)
        if self.is_training:
            protein_coord, protein_feature, cavity_coord, cavity_feature = inputs
        else:
            protein_coord, protein_feature, name = inputs

        if self.is_training:
            rotation = random.choice(range(0, 24)) if self.transform_random else 0
            translation = self.max_translation * np.random.rand(1, 3) if self.transform_random else (0, 0, 0)
            protein_coord = Rotator().rotate(protein_coord, rotation) + translation
        protein_grid = self.make_grid(protein_coord, protein_feature,
                                      grid_resolution=1.0 / self.resolution_scale)

        if self.is_training:
            pocket_coords = Rotator().rotate(cavity_coord, rotation) + translation
            # Convert atom coordinates and features represented as 2D arrays into a fixed-sized 3D box
            cavity_grid = self.make_grid(pocket_coords, cavity_feature)

            margin = ndimage.maximum_filter(cavity_grid, footprint=self.footprint)
            cavity_grid += margin
            cavity_grid = cavity_grid.clip(0, 1)
            zoom = protein_grid.shape[1] / cavity_grid.shape[1]
            cavity_grid = np.stack(
                [ndimage.zoom(cavity_grid[..., i], zoom) for i in range(self.y_channels)], -1)
            # cavity_grid = np.expand_dims(cavity_grid, 0)
            return protein_grid, cavity_grid
        return protein_grid, name

    def process(self, max_len=None):
        """data processing

        Args:
            maximum of dataset

        Returns:
            self
        """
        feature = Feature(save_molecule_codes=False)
        for i, name in enumerate(self.names):
            if max_len and i >= max_len:
                self.names = self.names[:i]
                break
            print(f'load {i} / {len(self.names)} {name} ...')
            pdb_path = os.path.join(self.data_path, name, name + '_protein.pdb')
            if not os.path.exists(pdb_path):
                pdb_path = os.path.join(self.data_path, name, name + '_protein.pdbqt')
            protein_coord, protein_feature = self.protein_biopy_featurizer(pdb_path)
            centroid = protein_coord.mean(axis=0)
            protein_coord -= centroid

            self.protein_coords.append(protein_coord)
            self.protein_features.append(protein_feature)

            if not self.is_training:
                continue
            cavity = next(pybel.readfile(self.fmt, os.path.join(self.data_path, name, name + '_ligand.' + self.fmt)))
            if self.protein_cavity_same_feature:
                cavity_coord, cavity_feature = feature.get_features(cavity)
            else:
                cavity_coord = np.zeros([len(cavity.atoms), 3])
                for j, atom in enumerate(cavity.atoms):
                    cavity_coord[j] = atom.coords
                cavity_feature = np.ones((len(cavity_coord), 1))
            cavity_coord -= centroid
            self.cavity_coords.append(cavity_coord)
            self.cavity_features.append(cavity_feature)
            if self.y_channels is None:
                self.y_channels = cavity_feature.shape[1]  # cavity_features  #36
        return self

    # output data shape is [G,G,G,F]
    def make_grid(self, coords, features, grid_resolution=1.0/2):
        """Make grid for protein

        Args:
            coords (np.ndarray):                3D coordinate of protein or cavity.
            features (np.ndarray):              Feature of each protein heavy atoms.
            grid_resolution (float, optional):  Resolution of grid. Defaults to 1.0/2.

        Raises:
            ValueError:     Coords must be an array of floats of shape (number of atoms in system, 3).
            ValueError:     Features must be an array of floats of shape (number of atoms in system, number of feature).
            TypeError:      Grid_resolution must be float.
            ValueError:     Grid_resolution must be positive.
            TypeError:      Max_dist must be float.
            ValueError:     Max_dist must be positive.

        Returns:
            grid: Tensor of shape(36, 36, 36, 18)
        """
        coords = np.asarray(coords, dtype=float)
        c_shape = coords.shape
        if len(c_shape) != 2 or c_shape[1] != 3:
            raise ValueError('coords must be an array of floats of shape (N, 3)')
        n_atoms = len(coords)
        features = np.asarray(features, dtype=float)
        f_shape = features.shape
        if len(f_shape) != 2 or f_shape[0] != n_atoms:
            raise ValueError('features must be an array of floats of shape (N, F)')
        if not isinstance(grid_resolution, (float, int)):
            raise TypeError('grid_resolution must be float')
        if grid_resolution <= 0:
            raise ValueError('grid_resolution must be positive')
        if not isinstance(self.max_distance, (float, int)):
            raise TypeError('max_dist must be float')
        if self.max_distance <= 0:
            raise ValueError('max_dist must be positive')
        num_features = f_shape[1]
        box_size = int(np.ceil(2. * self.max_distance / grid_resolution + 1))
        # move all atoms to the nearest grid point
        grid_coords = ((coords + self.max_distance + 0.) / grid_resolution).round().astype(int)
        in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)  # remove atoms outside the box
        grid = np.zeros((box_size, box_size, box_size, num_features), dtype=np.float32)
        for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
            grid[x, y, z, :] += f
        return grid

    def protein_rdkit_featurizer(self, pdb_path):
        """Use RDKit to generate protein surface feature

        Args:
            pdb_path (str):                 Path of protein pdb file.

        Returns:
            coordinates (np.ndarray):       shape (number of atoms in system, 3)
            representation (np.ndarray):    shape (number of atoms in system, 18).
        """
        mol = Chem.MolFromPDBFile(pdb_path)
        coords = mol.GetConformer().GetPositions()
        feature = []
        radii = rdFreeSASA.classifyAtoms(mol)
        if radii:
            rdFreeSASA.CalcSASA(mol, radii)
        for atom in mol.GetAtoms():
            name = atom.GetPDBResidueInfo().GetName().strip()
            resi = atom.GetPDBResidueInfo().GetResidueName().strip()
            key = resi + '_' + name
            if key not in RESI_ATOM_FEAT:
                continue
            if radii:
                surface = float(atom.GetProp("SASA")) > self.surface_scale
            else:
                surface = 0
            feature.append(RESI_ATOM_FEAT.get(key) + [surface])
        return coords, np.array(feature)

    def protein_biopy_featurizer(self, pdb_path):
        """Use Biopython to generate protein surface feature

        Args:
            pdb_path (str):                 Path of protein pdb file.

        Returns:
            Returns:
            coordinates (np.ndarray):       shape (number of atoms in system, 3)
            representation (np.ndarray):    shape (number of atoms in system, 18).
        """
        sr = SASA.ShrakeRupley(n_points=100)

        rec = PDB.PDBParser().get_structure('random_id', pdb_path)[0]
        sr.compute(rec, level="A")
        coords = []
        features = []
        for atom in rec.get_atoms():
            coord = list(atom.get_vector())
            name = atom.name
            resi = atom.get_parent().get_resname()
            key = resi + '_' + name
            if key not in RESI_ATOM_FEAT:
                continue
            features.append(RESI_ATOM_FEAT.get(key) + [atom.sasa])
            coords.append(coord)
        return np.array(coords), np.array(features)

    def get_pockets_segmentation(self, density, name, centroid, site_path,
                                 threshold=0.05, min_size=2, site_format='mol2'):
        """Transform the npz data to mol2 file

        Args:
            density (array_like):           npz data
            name (str):                     name of the molecule
            centroid (array_like):          centroid
            site_path (str):                file path to save the output
            threshold (float, optional):    threshold. Defaults to 0.05.
            min_size (int, optional):       minimum size. Defaults to 2.
            site_format (str, optional):    file format of output. Defaults to 'mol2'.

        Raises:
            ValueError: segmentation of more than one pocket is not supported

        Returns:
            site_name (str): The path of output file.
        """
        os.makedirs(os.path.dirname(site_path), exist_ok=True)

        if len(density) != 1:
            raise ValueError('segmentation of more than one pocket is not supported')
        print('name=', name)
        print('density:', np.shape(density))  # (-1, 36, 36, 36, 1)  #MUST '-1' dimension is 1
        print('name:', np.shape(name))  # (3, 1)
        print('centroid:', np.shape(centroid))  # (3, 1)

        origin = (centroid - self.max_distance)
        step = np.array([1.0 / self.resolution_scale] * 3)

        print('origin:', np.shape(origin))  # (3, 1)
        print('step:', np.shape(step))  # (3,)

        voxel_size = (1 / self.resolution_scale) ** 3
        bw = skimage.morphology.closing((density[0] > threshold).any(axis=-1))
        cleared = skimage.segmentation.clear_border(bw)
        label_image, num_labels = skimage.measure.label(cleared, return_num=True)
        for i in range(1, num_labels + 1):
            pocket_idx = (label_image == i)
            pocket_size = pocket_idx.sum() * voxel_size
            if pocket_size < min_size:
                label_image[np.where(pocket_idx)] = 0

        print('label_image:', np.shape(label_image))  # (36, 36, 36)
        pockets = label_image
        print('pockets.max():', pockets.max())  # (36, 36, 36)
        score_list = []
        for pocket_label in range(1, pockets.max() + 1):
            site_indx = np.argwhere(pockets == pocket_label)
            indices = np.argwhere(pockets == pocket_label).astype('float32')
            score_all = []
            for one in site_indx:
                score_all.append(density[0][one[0]][one[1]][one[2]])
            if indices.shape[0] == 0:
                score_avg = 0
            else:
                score_avg = float((sum(score_all) / len(score_all))[0])
            score_list.append(score_avg)
            indices *= step
            indices += origin[0]  # John
            mol = ob.OBMol()
            for idx in indices:
                a = mol.NewAtom()
                a.SetVector(float(idx[0]), float(idx[1]), float(idx[2]))
            p_mol = pybel.Molecule(mol)
            site_file = os.path.join(site_path, name[0]+'-site-'+str(pocket_label) +
                                     'score-' + "%.2f" % score_avg+'.'+site_format)
            site_name = os.path.join(name[0]+'-site-'+str(pocket_label)+'score-' + "%.2f" % score_avg+'.'+site_format)
            os.makedirs(os.path.dirname(site_file), exist_ok=True)
            p_mol.write(site_format, site_file, overwrite=True)
        return site_name
