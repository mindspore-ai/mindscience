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
Dataset
"""

import warnings
import os

import numpy as np
from rdkit import Chem
import pandas as pd
from tqdm import tqdm


from ..configs import Registry as R
from ..core import BaseDataset
from ..data.molecule import Molecule, MoleculeBatch


mol_passers = {'smiles': Chem.MolFromSmiles, 'smarts': Chem.MolFromSmiles, 'inchi': Chem.MolFromInchi}


@R.register('dataset.MolSet')
class MolSet(BaseDataset):
    """
    Molecular dataset. Each sample contains a molecule graph, and any number of prediction targets.

    Args:
            lazy (bool, optional):          if True, the molecules are processed in the dataloader. This may slow
                                            down the data loading process, but save a lot of CPU memory and dataset
                                            loading time. Default: ``False``.
            verbose (int, optional):        verbose (int, optional): output verbose level. Defaults to 0.
            transform (Callable, optional): transform (Callable, optional): data transformation function.
                                            Defaults to None.
    """
    _caches = ['data', 'label']

    def __init__(self,
                 lazy=False,
                 verbose=0,
                 transform=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self.lazy = lazy
        self.columns = ['graph.' + key for key in MoleculeBatch.keys()]
        self.columns += self._caches[1:]
        self.transform = transform or Molecule.from_molecule
        self.data = []
        self.label = []

    def __repr__(self):
        lines = f"#sample: {len(self)}\n" + \
                f"#task: {len(self.task_list)}"
        return f"{self.__class__.__name__}(\n  {lines}\n)"

    def __getitem__(self, index):
        params = list(super().__getitem__(index))
        if self.lazy:
            data = []
            for mol in params[0]:
                mol = Molecule.from_molecule(mol)
                data.append(mol)
            params[0] = data
        data = Molecule.pack(params[0]).to_dict()
        output = [data.get(k.split('.')[-1]) for k in self.columns if '.' in k]
        return output + params[1:]

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
                bond_types.update(graph.edge_type.tolist())
        else:
            for graph in self.data:
                bond_types.update(graph.edge_type.tolist())

        return sorted(bond_types)

    def load_file(self, fname, fmt='smiles', mol_field='smiles', max_len=None, **kwargs):
        """
        Load the dataset from a csv file.

        Args:
            fname (str):                    input file name
            fmt (str):                      format of the input file
            smiles_field (str, optional):   name of SMILES column in the table.
                                            Use ``None`` if there is no SMILES column.
            verbose (int, optional):        output verbose level

        Returns:
            self
        """
        assert os.path.exists(fname), f'Error: The file {fname} does not exist!'
        if fname.split('.')[-1] in self._seps and fmt in mol_passers.keys():
            sep = self._seps.get(fname.split('.')[-1])
            df = pd.read_table(fname, sep=sep)
            self.label = df[self.task_list].values
            if mol_field is None:
                mol_field = fmt
            assert mol_field in df
            setattr(self, 'mol', df[mol_field].values)
        elif fmt == 'sdf':
            setattr(self, 'mol', Chem.SDMolSupplier(fname, True, True, False))
        else:
            raise TypeError(f'The iput file format \"{fmt}\" is not support')

        if hasattr(self, 'mol'):
            return self.load_mol(max_len=max_len, fmt=fmt, **kwargs)
        return self

    def load_mol(self, max_len=None, fmt='smiles', **kwargs):
        """
        Transform the molecule to Graph data.

        Args:
            max_len (int, optional): maximum length of the molecules. Defaults to None.

        Returns:
            self
        """
        labels = []
        if not hasattr(self, 'mol') or self.lazy:
            return self

        indexes = enumerate(self.mol)
        if self.verbose:
            indexes = tqdm(indexes, "Constructing molecules", total=len(self.mol))
        for i, mol in indexes:
            if mol is None:
                continue
            if fmt != 'sdf':
                passer = mol_passers.get(fmt)
                mol = passer(mol)
            if not self.lazy:
                mol = self.transform(mol, **kwargs)
            self.data.append(mol)
            if 'label' in self._caches:
                label = [mol.GetProp(task) for task in self.task_list] if self.label is None else self.label[i]
                labels.append(label)
            if max_len and len(self.data) >= max_len:
                break
        if 'label' in self._caches:
            self.label = np.stack(labels).astype(np.float32)
        return self
