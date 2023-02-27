# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""PDBBind"""
import os
import tarfile
from tqdm import tqdm
from ..dataset import DataSet


class PDBBind(DataSet):
    """"PDBBind Dataset"""
    def __init__(self):

        self.url = {
            "index": "http://www.pdbbind.org.cn/download/PDBbind_2016_plain_text_index.tar.gz",
            "general": "http://www.pdbbind.org.cn/download/pdbbind_v2016_general-set-except-refined.tar.gz",
            "refined": "http://www.pdbbind.org.cn/download/pdbbind_v2016_refined.tar.gz",
            "pp": "http://www.pdbbind.org.cn/download/pdbbind_v2016_PP.tar.gz",
            "mol2": "http://www.pdbbind.org.cn/download/PDBbind_v2016_mol2.tar.gz",
            "sdf": "http://www.pdbbind.org.cn/download/PDBbind_v2016_sdf.tar.gz",
            "2013": "http://www.pdbbind.org.cn/download/pdbbind_v2013_core_set.tar.gz"
        }

        self.cache = "./PDBBind_data"
        self.in_memory = True
        super().__init__()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download(self, path=None):
        """download"""
        if path is not None:
            self.cache = path
        print("Start download data")
        for _, url in self.url.items():
            command = "wget -P " + self.cache + " " + url
            os.system(command)

        file_list = os.listdir(path)
        tar_gz_list = []
        for val in file_list:
            if val.endswith("tar.gz"):
                tar_gz_list.append(val)

        print("Start uncompression ... ")
        for i in tqdm(range(len(tar_gz_list))):
            val = tar_gz_list[i]
            val_path = os.path.join(path, val)
            if "PDBbind_2016_plain_text_index" in val:
                dir_path = os.path.join(self.cache, "PDBbind_2016_plain_text_index/")
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                tar_file = tarfile.open(val)
                tar_file.extractall(dir_path)
            else:
                tar_file = tarfile.open(val)
                tar_file.extractall(val_path)
        print("Finish uncompression ... ")
        print("PDBBind has been saved in ", self.cache)

    def process(self, data, **kwargs):
        raise NotImplementedError

    def data_parse(self, input_data, idx):
        raise NotImplementedError

    def create_iterator(self, num_epochs, **kwargs):
        raise NotImplementedError
