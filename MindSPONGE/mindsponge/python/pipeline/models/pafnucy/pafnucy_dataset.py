# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
"""pafnucy data"""
import os
import warnings

import numpy as np
from mindspore.dataset import GeneratorDataset
from openbabel import openbabel as ob
from openbabel import pybel
from sklearn.utils import shuffle

from ...dataset import PDBBind
from .pafnucy_data import (extrct2013ids, parseandclean, extractfeature,
                           preprocess)

ob.obErrorLog.SetOutputLevel(0)


class PAFNUCYDataSet(PDBBind):
    """pafnucy dataset"""
    def __init__(self, config):
        self.config = config
        self.is_training = self.config.is_training
        self.std = 0.19213134 # given by shuffle seed 123
        self.data_size = 0
        self.pdbs = {"general": [], "refined": [], "core": []}
        self.labels = {}
        self.schemalist = ["coords_feature", "affinity", "rot"]
        self.general_data_src = ""
        self.refine_data_src = ""
        self.general_pdbids = []
        self.refine_pdbids = []
        self.training_pdbids = []
        self.training_size = 0
        super().__init__()


    def __getitem__(self, idx):
        data, label = self.data_parse(idx=idx)
        rot = False
        if self.is_training:
            assert self.training_size != 0
            rotation = idx // self.training_size
            if rotation >= self.config.rotations:
                rotation = 0
                rot = True
            else:
                rot = False
                rotation = rotation.item()
        else:
            rotation = 0
        features = self.process(data, label, rotation, rot)
        tuple_feature = tuple([features.get(key) for key in self.schemalist])
        return tuple_feature


    def __len__(self):
        data_len = self.training_size * (self.config.rotations + 1)
        return data_len


    def get_path(self, pdbid, pdbset):
        "get path"
        if pdbset == "general":
            ligand_path = self.general_data_src + pdbid + f"/{pdbid}_ligand.mol2"
            if os.path.exists(self.general_data_src + pdbid + f"/{pdbid}_pocket.mol2"):
                pocket_path = self.general_data_src + pdbid + f"/{pdbid}_pocket.mol2"
            else:
                pocket_path = self.general_data_src + pdbid + f"/{pdbid}_pocket.pdb"
                molfile = pocket_path.replace(".pdb", ".mol2")
                command = "obabel -i pdb %s -o mol2 -O %s" % (pocket_path, molfile)
                os.system(command)
                pocket_path = molfile
        else:
            ligand_path = self.refine_data_src + pdbid + f"/{pdbid}_ligand.mol2"
            if os.path.exists(self.refine_data_src + pdbid + f"/{pdbid}_pocket.mol2"):
                pocket_path = self.refine_data_src + pdbid + f"/{pdbid}_pocket.mol2"
            else:
                pocket_path = self.refine_data_src + pdbid + f"/{pdbid}_pocket.pdb"
                molfile = pocket_path.replace(".pdb", ".mol2")
                command = "obabel -i pdb %s -o mol2 -O %s" % (pocket_path, molfile)
                os.system(command)
                pocket_path = molfile
        return ligand_path, pocket_path

    # pylint: disable=arguments-differ
    def process(self, data, label=None, rotation=0, rot=False):
        """data process"""
        assert len(data) == 2
        pocket = data[0]
        ligand = data[1]

        feature = extractfeature(pocket, ligand)
        coords = feature[:, :3]
        features = feature[:, 3:]
        coords_feature = preprocess(coords, features, self.config, self.std, rotation=rotation)
        coords_feature = np.array(coords_feature, dtype=np.float32)
        if label is not None:
            affinity = label
        else:
            affinity = -1
        return {"coords_feature": coords_feature, "affinity": affinity, "rot": rot}


    def download(self, path=None):
        pass


    def data_parse(self, input_data=None, idx=0):
        """data parse"""
        if input_data is None:
            pdbid = self.training_pdbids[idx][0]
            pdbset = self.training_pdbids[idx][1]
        else:
            pdbid = input_data[0]
            pdbset = input_data[1]
        assert pdbset in ["general", "refined"]
        ligand_path, pocket_path = self.get_path(pdbid, pdbset)
        ligand = next(pybel.readfile('mol2', ligand_path))
        try:
            pocket = next(pybel.readfile('mol2', pocket_path))
        except ValueError:
            warnings.warn('no pocket available.')
        label = self.labels.get(pdbid)
        data = [pocket, ligand]
        return data, label


    def set_training_data_src(self, data_src=None):
        """set training data src"""
        if data_src is None:
            data_src = self.cache
        cmd = "cp {data_src}/index/INDEX_core_data.2016 {data_src}/PDBbind_2016_plain_text_index/index/"
        os.system(cmd)
        print("Start preprocessing PDBBind data ... ")
        if not os.path.exists(os.path.join(data_src, 'PDBbind_2016_plain_text_index/index/INDEX_general_PL_data.2016')):
            raise IOError("INDEX_general_PL_data.2016 file doesn't exit!")
        if not os.path.exists(os.path.join(data_src, 'PDBbind_2016_plain_text_index/index/INDEX_core_data.2016')):
            raise IOError("INDEX_core_data.2016 file doesn't exit!")
        if not os.path.exists(os.path.join(data_src, 'PDBbind_2016_plain_text_index/index/INDEX_refined_data.2016')):
            raise IOError("INDEX_refined_data.2016 file doesn't exit!")
        if os.path.exists(os.path.join(data_src, 'core_pdbbind2013.ids')):
            print("Remove Exist core_pdbbind2013.ids file.")
            os.remove(os.path.join(data_src, 'core_pdbbind2013.ids'))

        self.general_data_src = data_src + "general-set-except-refined/"
        self.refine_data_src = data_src + "refined-set/"

        extrct2013ids(data_src)
        affinity_data = parseandclean(data_src)
        self.data_size = len(affinity_data)
        for i in range(self.data_size):
            pdbid = affinity_data.iloc[i, 0]
            pdbset = affinity_data.iloc[i, 3]
            ligand_path, pocket_path = self.get_path(pdbid, pdbset)
            ligand = next(pybel.readfile('mol2', ligand_path))
            try:
                pocket = next(pybel.readfile('mol2', pocket_path))
            except ValueError:
                print(ValueError)
                continue
            if ligand is None or pocket is None:
                continue

            if affinity_data.iloc[i, 2]:
                self.pdbs[pdbset].append([pdbid, pdbset])
            else:
                self.pdbs["core"].append([pdbid, "refined"])
            self.labels[pdbid] = affinity_data.iloc[i, 1]

        self.general_pdbids = self.pdbs.get("general")
        self.refine_pdbids = self.pdbs.get("refined")

        refined_shuffled = shuffle(self.refine_pdbids, random_state=123)
        self.training_pdbids = self.general_pdbids + refined_shuffled[self.config.size_val:]
        self.training_size = len(self.training_pdbids)
        self.training_pdbids *= (self.config.rotations + 1)


    def create_iterator(self, num_epochs):
        dataset = GeneratorDataset(source=self, column_names=self.schemalist,
                                   num_parallel_workers=4, shuffle=False, max_rowsize=16)
        dataset = dataset.batch(batch_size=20, drop_remainder=True)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
        return iteration
