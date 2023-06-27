
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
from .. import data
from .. import util


class ZINC250k(data.MolSet):
    """
    Subset of ZINC compound database for virtual screening.

    Statistics:
        - #Molecule: 498,910
        - #Regression task: 2

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/" \
          "250k_rndm_zinc_drugs_clean_3.csv"
    md5 = "b59078b2b04c6e9431280e3dc42048d5"
    target_fields = ["logP", "qed"]

    def __init__(self, path, batch_size=128, verbose=1, **kwargs):
        super().__init__(batch_size=batch_size)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = util.download(self.url, path, md5=self.md5)

        self.load_csv(file_name, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)
