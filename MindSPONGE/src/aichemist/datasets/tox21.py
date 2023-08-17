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
Tox21
"""

import os
from . import dataset
from .. import utils


class Tox21(dataset.MolSet):
    """
    Qualitative toxicity measurements on 12 biological targets, including nuclear receptors
    and stress response pathways.

    Statistics:
        - #Molecule: 7,831
        - #Classification task: 12

    Args:
        path (str):                 path to store the dataset
        verbose (int, optional):    output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    md5 = "2882d69e70bba0fec14995f26787cc25"
    task_list = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
                 "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

    def __init__(self, path, verbose=1, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def process(self, **kwargs):
        """data processing"""
        zip_file = utils.download(self.url, self.path, md5=self.md5)
        csv_file = utils.extract(zip_file)
        self.load_file(csv_file, mol_field="smiles", **kwargs)
        return self
