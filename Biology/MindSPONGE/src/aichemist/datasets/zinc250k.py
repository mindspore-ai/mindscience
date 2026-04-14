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
zinc250k
"""


import os
from . import dataset
from .. import utils


class ZINC250k(dataset.MolSet):
    """
    Subset of ZINC compound database for virtual screening.

    Statistics:
        - #Molecule: 498,910
        - #Regression task: 2

    Args:
        path (str):                 path to store the dataset
        verbose (int, optional):    output verbose level
        **kwargs
    """

    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/" \
          "250k_rndm_zinc_drugs_clean_3.csv"
    md5 = "b59078b2b04c6e9431280e3dc42048d5"
    task_list = ["logP", "qed"]
    _caches = ['data']

    def __init__(self, path, verbose=1, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def process(self, **kwargs):
        """data processing"""
        file_name = utils.download(self.url, self.path, md5=self.md5)
        self.load_file(file_name, mol_field="smiles", **kwargs)
        return self
