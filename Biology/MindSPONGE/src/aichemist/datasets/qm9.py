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
qm9
"""


import os
from . import dataset
from .. import utils


class QM9(dataset.MolSet):
    """
    Geometric, energetic, electronic and thermodynamic properties of DFT-modeled small molecules.

    Statistics:
        - #Molecule: 133,885
        - #Regression task: 12

    Args:
        path (str):                 path to store the dataset
        position (bool, optional):  load node position or not.
                                    This will add `position` as a node attribute to each sample.
        verbose (int, optional):    output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
    fname = 'gdb9.tar.gz'
    md5 = "560f62d8e6c992ca0cf8ed8d013f9131"
    task_list = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298", "h298", "g298"]

    def __init__(self, path, verbose=1, info='graph', **kwargs):
        info = ['atom_coord', 'atom_type']
        super().__init__(verbose=verbose, info=info, **kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def process(self, max_len=None):
        """data processing"""
        fname = self.path + '/' + self.fname
        if not os.path.exists(self.fname):
            fname = utils.download(self.url, self.path, md5=self.md5)
        sdf_file = utils.extract(fname, "gdb9.sdf")
        csv_file = utils.extract(fname, "gdb9.sdf.csv")
        self.load_file(csv_file)
        self.load_file(sdf_file, fmt='sdf', max_len=max_len)
        return self
